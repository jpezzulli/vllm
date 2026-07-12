# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-context-parallel helpers for sparse-MLA (DSA) top-k indexers.

Under DCP the KV cache is interleaved across ranks in units of
``cp_kv_cache_interleave_size`` tokens: global token position ``p`` lives on
rank ``(p // interleave) % world`` at local position
``(p // (world * interleave)) * interleave + p % interleave``.

The DSA indexer computes per-query relevance logits against the *local* KV
shard and selects a local top-k. The helpers here lift those local winners
into global coordinates, exchange the (id, score) candidate lists across the
DCP group and reduce them to the *global* top-k, which every rank then filters
back down to its own slots for the sparse attention read.

Scores are raw indexer logits (weighted ReLU dot products) with no
per-sequence normalization, so they are directly comparable across ranks.
"""

import torch

from vllm.distributed.parallel_state import GroupCoordinator

# Candidate ids ride the all-gather as float32 lanes next to their scores.
# float32 represents integers exactly up to 2**24, which bounds the maximum
# indexable token position (1M context is well within it).
MAX_FP32_EXACT_ID = 1 << 24


def dcp_local_count_from_global(
    global_lens: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    interleave: int,
) -> torch.Tensor:
    """Number of tokens of a ``global_lens``-token prefix stored on this rank.

    Works elementwise on any integer tensor (per-request seq_lens or
    per-query-token causal lengths alike).
    """
    unit = dcp_world_size * interleave
    full = global_lens // unit
    rem = global_lens - full * unit
    return full * interleave + (rem - dcp_rank * interleave).clamp_(
        min=0, max=interleave
    )


def dcp_local_count_int(
    global_len: int,
    dcp_world_size: int,
    dcp_rank: int,
    interleave: int,
) -> int:
    """Python-int twin of :func:`dcp_local_count_from_global`."""
    unit = dcp_world_size * interleave
    full, rem = divmod(global_len, unit)
    return full * interleave + min(max(rem - dcp_rank * interleave, 0), interleave)


def dcp_local_pos_to_global(
    local_idx: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    interleave: int,
) -> torch.Tensor:
    """Map local token positions to global positions; ``-1`` stays ``-1``."""
    unit = local_idx // interleave
    global_idx = (
        unit * (dcp_world_size * interleave)
        + dcp_rank * interleave
        + local_idx % interleave
    )
    return torch.where(local_idx < 0, local_idx, global_idx)


def dcp_gather_topk_scores(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    col_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fetch the logit score for every locally selected candidate.

    ``topk_indices`` hold per-request positions (``-1`` = padding). For the
    prefill path the request's K rows start at ``col_offset[row]`` within the
    shared logits workspace; decode logits are already request-relative.
    Padded entries get ``-inf`` so they lose every comparison in the merge.
    """
    idx = topk_indices.clamp(min=0).to(torch.int64)
    if col_offset is not None:
        idx = idx + col_offset.to(torch.int64).unsqueeze(1)
    # Guard the gather against padded rows pointing past the logits width
    # (possible for zero-length rows whose clamp(0) still lands out of range
    # only when the logits tensor is narrower than 1 column - never in
    # practice - but clamp anyway for CUDA-graph padded rows).
    idx = idx.clamp_(max=logits.shape[1] - 1)
    scores = logits.gather(1, idx).to(torch.float32)
    return scores.masked_fill_(topk_indices < 0, float("-inf"))


def dcp_merge_global_topk(
    local_global_ids: torch.Tensor,
    local_scores: torch.Tensor,
    topk: int,
    dcp_group: GroupCoordinator,
) -> torch.Tensor:
    """All-gather per-rank candidates and reduce to the global top-``topk``.

    ``local_global_ids``: int32 ``[rows, topk]`` global positions, ``-1`` pad.
    ``local_scores``: float32 ``[rows, topk]``, ``-inf`` on padding.

    Returns int32 ``[rows, topk]`` global ids (``-1`` pad), identical on every
    rank: candidate sets are disjoint per rank (ownership partitions
    positions), the gathered layout is rank-major on every rank, and topk over
    the same tensor is deterministic.

    Correctness of the reduction: any position in the true global top-k is,
    restricted to its owner rank, within that rank's local top-k, so the union
    of local top-k lists always contains the global top-k.
    """
    rows = local_global_ids.shape[0]
    world = dcp_group.world_size

    # Single collective: scores and ids share one fp32 tensor. ids <= 2**24
    # are exact in fp32 (asserted at engine start via max_model_len).
    packed = torch.empty(
        (rows, 2, topk), dtype=torch.float32, device=local_scores.device
    )
    packed[:, 0].copy_(local_scores)
    packed[:, 1].copy_(local_global_ids.to(torch.float32))

    # [rows, 2, topk] -> [world * rows, 2, topk], rank-major.
    gathered = dcp_group.all_gather(packed, dim=0)
    gathered = gathered.view(world, rows, 2, topk)

    all_scores = (
        gathered[:, :, 0].permute(1, 0, 2).reshape(rows, world * topk)
    )
    all_ids = gathered[:, :, 1].permute(1, 0, 2).reshape(rows, world * topk)

    top_scores, top_pos = torch.topk(all_scores, k=topk, dim=1, sorted=True)
    merged = all_ids.gather(1, top_pos).to(torch.int32)
    # Positions that never existed (fewer than topk real candidates in total)
    # carry -inf scores; restore the -1 padding convention.
    merged.masked_fill_(top_scores == float("-inf"), -1)
    return merged
