# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(SparseMLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    # DCP support: the SM120 sparse kernels emit a per-(token, head) LSE that
    # the MLA layer merges across ranks. NOTE: the kernels store LSE in the
    # log2 domain (log2f(sum) + max, sentinel -1e30 for empty rows); the
    # merge must run with is_lse_base_on_e=False (see returns_base2_lse).
    can_return_lse_for_decode: bool = True
    returns_base2_lse: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype not in ("fp8_ds_mla", "nvfp4"):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"or nvfp4 KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        if self.kv_cache_dtype == "nvfp4":
            # NVFP4 block-16 packed cache; requires the GLM_NSA_NVFP4 model
            # type in FlashInfer's sparse-MLA SM120 kernels.
            self.kv_scale_format = "nvfp4_b16"
        else:
            self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Skip-topk layers are built with indexer=None and get the shared
        # buffer via mla_args instead (cf. FLASHMLA_SPARSE).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

        # DCP: the shared topk buffer holds *global* positions merged by the
        # indexer; forward_mqa filters them down to this rank's local slots.
        self.cp_kv_cache_interleave_size = (
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if kv_cache_dtype == "nvfp4":
            from vllm.v1.attention.backends.mla.nvfp4_ds_mla_cache import (
                concat_and_cache_nvfp4_ds_mla,
            )

            k_pe_2d = k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe
            concat_and_cache_nvfp4_ds_mla(
                kv_c_normed, k_pe_2d, kv_cache, slot_mapping.flatten()
            )
            return
        super().do_kv_cache_update(
            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]
        # Under DCP the MLA layer all-gathers q along the head dim, so this
        # rank attends with every head over its local KV shard; the layer
        # LSE-merges and reduce-scatters the partial outputs afterwards.
        num_heads = q.shape[1] if self.dcp_world_size > 1 else self.num_heads

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # With DCP the buffer holds globally merged positions; entries owned
        # by other ranks resolve to -1 and are skipped by the sparse kernel
        # (interspersed -1 indices are masked, not just tail padding).
        return_lse = self.need_to_return_lse_for_decode
        valid_counts: torch.Tensor | None = None
        conv = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            dcp_world_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            # Rows with no locally owned candidate are skipped by the kernel
            # (their LSE lands at the -1e30 sentinel but the output row may
            # keep stale pool memory) - track them so they can be zeroed
            # before the cross-rank merge.
            return_valid_counts=return_lse,
        )
        if return_lse:
            topk_indices_physical, valid_counts = cast(
                tuple[torch.Tensor, torch.Tensor], conv
            )
        else:
            topk_indices_physical = cast(torch.Tensor, conv)

        output = q.new_empty(
            (num_actual_toks, num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=attn_metadata.topk_tokens,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            kv_scale_format=self.kv_scale_format,
            return_lse=return_lse,
        )
        if return_lse:
            out, lse = out
            out = out.squeeze(1)
            # lse: [num_tokens, num_heads] fp32, log2 domain. Rows where this
            # rank owns no selected candidate report the -1e30 LSE sentinel
            # (zero weight in the merge), but their *output* rows are
            # uninitialized pool memory - zero them so a stray NaN/Inf can't
            # leak through the 0-weight multiply in the LSE correction.
            assert valid_counts is not None
            out.masked_fill_((valid_counts == 0).view(-1, 1, 1), 0.0)
            return out, lse
        return out.squeeze(1), None
