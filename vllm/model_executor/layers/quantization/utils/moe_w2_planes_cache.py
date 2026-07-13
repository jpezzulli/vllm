# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk cache for built 2-bit expert planes (+ optional FP4 delta planes).

The load-time requant (f64 dequant -> sign-symmetric 2-bit -> fragment-major
pack) costs ~9 min per restart on Kimi-K2.7 (23,040 expert-layer pairs).
The result is deterministic given (checkpoint, TP layout, zero mode,
codebook), so it is cached to disk after the first build and streamed back
on later restarts, skipping the requant entirely.

Opt-in via VLLM_MOE_W2_PLANES_CACHE=<dir>. Layout:

  <dir>/tp{W}-rank{R}/meta.json          # cache key for this rank
  <dir>/tp{W}-rank{R}/layer{L}.<part>.bin  # raw u8 tensors

Parts per layer: planes13, sc13, planes2, sc2 and, when the FP4 delta tier
was enabled at build time, fp13, fp2. A layer HITS when meta matches and
every required part exists with the exact expected size (computed from the
layer's weight shapes); anything else is a MISS for that layer and it
rebuilds (and rewrites) from the checkpoint as before. Writes go through a
background thread (tmp file + atomic rename), are best-effort, and never
fail the load. Note the vLLM weight loader still reads the checkpoint
shards on a hit — only the requant is skipped (loader-level skip is a
possible follow-up).

Sizes (Kimi-K2.7 @ TP4): 2-bit ~70 GiB/rank, FP4 ~126 GiB/rank.
"""

import hashlib
import json
import os
import queue
import re
import threading

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_VERSION = "tsym4-fragmajor-v1"
_PARTS_2BIT = ("planes13", "sc13", "planes2", "sc2")
_PARTS_FP4 = ("fp13", "fp2")

_meta_written = False
_writer: "queue.Queue[tuple[str, torch.Tensor] | None] | None" = None
_broken = False


def enabled() -> bool:
    return bool(os.getenv("VLLM_MOE_W2_PLANES_CACHE"))


def layer_idx_from_name(layer_name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", layer_name or "")
    return int(m.group(1)) if m else None


def _tp_ids() -> tuple[int, int]:
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    return get_tensor_model_parallel_world_size(), \
        get_tensor_model_parallel_rank()


def _ckpt_id() -> str:
    """Identity of the checkpoint the planes derive from: model path +
    sha1 of the safetensors index (covers shard layout and tensor set)."""
    from vllm.config import get_current_vllm_config
    model = get_current_vllm_config().model_config.model
    h = hashlib.sha1(model.encode())
    idx = os.path.join(model, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def _meta() -> dict:
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    world, rank = _tp_ids()
    return dict(
        version=_VERSION,
        ckpt_id=_ckpt_id(),
        world=world,
        rank=rank,
        zero_mode=os.getenv("VLLM_MOE_W2_ZERO_MODE", "auto"),
        # split-FP4 stores radix-5 quintal planes in fp13/fp2 (5/8 of the
        # nibble bytes) — a cache built in another mode must MISS
        # wholesale. "w4q" also invalidates caches of the superseded
        # 2-bit-refinement split (stored fp4_split=True).
        fp4_split="w4q" if moe_w2_delta.split_enabled() else False,
    )


def _rank_dir() -> str:
    world, rank = _tp_ids()
    return os.path.join(os.environ["VLLM_MOE_W2_PLANES_CACHE"],
                        f"tp{world}-rank{rank}")


def expected_sizes(E: int, N13: int, K13: int, N2: int, K2: int,
                   want_fp4: bool) -> dict[str, int]:
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    exp = {
        "planes13": E * N13 * K13 // 4,
        "sc13": E * N13 * K13 // 32,
        "planes2": E * N2 * K2 // 4,
        "sc2": E * N2 * K2 // 32,
    }
    if want_fp4:
        if moe_w2_delta.split_enabled():
            exp["fp13"] = E * N13 * K13 * 5 // 16
            exp["fp2"] = E * N2 * K2 * 5 // 16
        else:
            exp["fp13"] = E * N13 * K13 // 2
            exp["fp2"] = E * N2 * K2 // 2
    return exp


def cache_has_layer(layer_idx: int, sizes: dict[str, int]) -> bool:
    """Presence probe for the loader-level skip: would try_load hit, without
    reading a byte of plane data? Same validity rules (meta match + every
    required part at its exact expected size); os.path.getsize only. Never
    raises."""
    if not enabled() or _broken:
        return False
    try:
        d = _rank_dir()
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp) or json.load(open(mp)) != _meta():
            return False
        for part, nbytes in sizes.items():
            p = os.path.join(d, f"layer{layer_idx}.{part}.bin")
            if not os.path.exists(p) or os.path.getsize(p) != nbytes:
                return False
        return True
    except Exception:  # noqa: BLE001 - probe only, staging path still works
        return False


def try_load(layer_idx: int,
             sizes: dict[str, int]) -> dict[str, torch.Tensor] | None:
    """CPU u8 tensors for a cached layer, or None (miss). Never raises."""
    if not enabled() or _broken:
        return None
    try:
        d = _rank_dir()
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp):
            return None
        if json.load(open(mp)) != _meta():
            _mark_broken(f"meta mismatch in {d} (stale cache?) — rebuilding")
            return None
        out = {}
        for part, nbytes in sizes.items():
            p = os.path.join(d, f"layer{layer_idx}.{part}.bin")
            if not os.path.exists(p):
                logger.info("moe_w2 planes cache: MISS layer %d (%s absent)",
                            layer_idx, part)
                return None
            if os.path.getsize(p) != nbytes:
                logger.info(
                    "moe_w2 planes cache: MISS layer %d (%s size %d != %d)",
                    layer_idx, part, os.path.getsize(p), nbytes)
                return None
            out[part] = torch.from_numpy(np.fromfile(p, dtype=np.uint8))
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("moe_w2 planes cache: read failed (%s) — rebuilding",
                       e)
        return None


def _mark_broken(msg: str) -> None:
    global _broken
    if not _broken:
        _broken = True
        logger.warning("moe_w2 planes cache: %s", msg)


def _writer_loop() -> None:
    while True:
        item = _writer.get()
        if item is None:
            return
        path, cpu = item
        try:
            tmp = path + ".tmp"
            cpu.numpy().tofile(tmp)
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001
            _mark_broken(f"write failed for {path}: {e}")


def store(layer_idx: int, tensors: dict[str, torch.Tensor]) -> None:
    """Queue a built layer's planes for background writing. Never raises;
    tensors may live on GPU (copied to CPU here, synchronously)."""
    global _meta_written, _writer
    if not enabled() or _broken:
        return
    try:
        d = _rank_dir()
        os.makedirs(d, exist_ok=True)
        if not _meta_written:
            mp = os.path.join(d, "meta.json")
            if os.path.exists(mp) and json.load(open(mp)) != _meta():
                # stale cache from another checkpoint/config: start over
                for f in os.listdir(d):
                    os.unlink(os.path.join(d, f))
            with open(mp + ".tmp", "w") as f:
                json.dump(_meta(), f)
            os.replace(mp + ".tmp", mp)
            _meta_written = True
        if _writer is None:
            # maxsize bounds the transient host copies (~3 GiB at Kimi TP4
            # layer sizes) and back-pressures the build if the disk lags.
            _writer = queue.Queue(maxsize=2)
            threading.Thread(target=_writer_loop, daemon=True,
                             name="moe-w2-planes-cache").start()
        for part, t in tensors.items():
            if t is None:
                continue
            path = os.path.join(d, f"layer{layer_idx}.{part}.bin")
            _writer.put((path, t.detach().reshape(-1).cpu()))
    except Exception as e:  # noqa: BLE001
        _mark_broken(f"store failed: {e}")
