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

import functools
import hashlib
import json
import os
import queue
import re
import threading
import time

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    MOE_W2_QUANTIZER_ABI,
    zero_mode,
)

logger = init_logger(__name__)

_VERSION = 2
_PARTS_2BIT = ("planes13", "sc13", "planes2", "sc2")
_PARTS_FP4 = ("fp13", "fp2")

_meta_written = False
_writer: "queue.Queue[tuple[str, torch.Tensor] | None] | None" = None
_writer_thread: threading.Thread | None = None
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


@functools.cache
def _ckpt_id() -> str:
    """Cheap but content-sensitive checkpoint identity.

    Reading every byte of a 400-600 GiB checkpoint on each boot would erase
    most of the persistent-cache win.  Instead hash the immutable HF revision
    when available plus the index and a stable manifest of every referenced
    shard (resolved blob id, size and nanosecond timestamps).  Normal shard
    replacement is therefore detected without bulk I/O.  Operators that
    manage an immutable external snapshot can provide its digest explicitly
    through VLLM_MOE_W2_CKPT_ID.
    """
    override = os.getenv("VLLM_MOE_W2_CKPT_ID", "").strip()
    if override:
        return override
    from vllm.config import get_current_vllm_config
    model_config = get_current_vllm_config().model_config
    model = str(model_config.model)
    resolved_revision = getattr(
        getattr(model_config, "hf_config", None), "_commit_hash", None)
    revision = resolved_revision or getattr(model_config, "revision", None)
    h = hashlib.sha256()
    h.update(b"moe-w2-checkpoint-stat-v2\0")
    h.update(model.encode())
    h.update(b"\0")
    h.update(str(revision or "").encode())
    if not os.path.isdir(model):
        immutable_revision = (
            isinstance(revision, str)
            and len(revision) == 40
            and all(c in "0123456789abcdef" for c in revision.lower())
        )
        if not immutable_revision:
            raise RuntimeError(
                "moe_w2 persistent cache needs a local checkpoint path, an "
                "immutable resolved 40-hex model revision, or "
                "VLLM_MOE_W2_CKPT_ID")
        return h.hexdigest()

    config_path = os.path.join(model, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            h.update(b"config.json\0")
            h.update(hashlib.sha256(f.read()).digest())

    shards: set[str] = set()
    for name in ("model.safetensors.index.json",
                 "pytorch_model.bin.index.json"):
        idx = os.path.join(model, name)
        if not os.path.exists(idx):
            continue
        with open(idx, "rb") as f:
            raw = f.read()
        h.update(name.encode())
        h.update(hashlib.sha256(raw).digest())
        try:
            weight_map = json.loads(raw).get("weight_map", {})
            shards.update(str(v) for v in weight_map.values())
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            raise RuntimeError(f"invalid checkpoint index {idx}: {e}") from e
    if not shards:
        shards.update(
            name for name in os.listdir(model)
            if name.endswith((".safetensors", ".bin")))
    if not shards:
        raise RuntimeError(
            f"no checkpoint weight shards found under {model!r}")
    for rel in sorted(shards):
        path = os.path.join(model, rel)
        st = os.stat(path)
        record = {
            "name": rel,
            "resolved_blob": os.path.basename(os.path.realpath(path)),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns,
        }
        h.update(json.dumps(record, sort_keys=True, separators=(",", ":"))
                 .encode())
    return h.hexdigest()


def _meta() -> dict:
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    world, rank = _tp_ids()
    return dict(
        version=_VERSION,
        quantizer_abi=MOE_W2_QUANTIZER_ABI,
        ckpt_id=_ckpt_id(),
        world=world,
        rank=rank,
        zero_mode=zero_mode(),
        # split-FP4 stores radix-5 quintal planes in fp13/fp2 (5/8 of the
        # nibble bytes) — a cache built in another mode must MISS
        # wholesale. "w4q" also invalidates caches of the superseded
        # 2-bit-refinement split (stored fp4_split=True).
        fp4_split="w4q" if moe_w2_delta.split_enabled() else False,
    )


def _rank_dir() -> str:
    world, rank = _tp_ids()
    meta_key = hashlib.sha256(
        json.dumps(_meta(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return os.path.join(os.environ["VLLM_MOE_W2_PLANES_CACHE"], "v2",
                        meta_key, f"tp{world}-rank{rank}")


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
    return cache_layer_files(layer_idx, sizes) is not None


def cache_layer_files(
    layer_idx: int,
    sizes: dict[str, int],
) -> dict[str, str] | None:
    """Return exact-size files for one compatible cached layer.

    This is the cheap eligibility contract used by loader-level skipping:
    cache-key metadata plus existence and exact size for every requested
    part. The payload is deliberately not read or hashed here.
    """
    if not enabled() or _broken:
        return None
    try:
        d = _rank_dir()
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp):
            return None
        with open(mp) as f:
            meta = json.load(f)
        if meta != _meta():
            return None
        files = {}
        for part, nbytes in sizes.items():
            p = os.path.join(d, f"layer{layer_idx}.{part}.bin")
            if not os.path.exists(p) or os.path.getsize(p) != nbytes:
                return None
            files[part] = p
        return files
    except Exception:  # noqa: BLE001 - probe only, staging path still works
        return None


def try_load(layer_idx: int,
             sizes: dict[str, int]) -> dict[str, torch.Tensor] | None:
    """CPU u8 tensors for a cached layer, or None (miss). Never raises."""
    if not enabled() or _broken:
        return None
    try:
        files = cache_layer_files(layer_idx, sizes)
        if files is None:
            return None
        out = {}
        for part, nbytes in sizes.items():
            raw = np.fromfile(files[part], dtype=np.uint8)
            if raw.nbytes != nbytes:
                return None
            out[part] = torch.from_numpy(raw)
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
            tmp = (f"{path}.tmp.{os.getpid()}."
                   f"{threading.get_ident()}")
            with open(tmp, "wb") as f:
                cpu.numpy().tofile(f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001
            _mark_broken(f"write failed for {path}: {e}")


def store(layer_idx: int, tensors: dict[str, torch.Tensor]) -> None:
    """Queue a built layer's planes for background writing. Never raises;
    tensors may live on GPU (copied to CPU here, synchronously)."""
    global _meta_written, _writer, _writer_thread
    if not enabled() or _broken:
        return
    try:
        d = _rank_dir()
        os.makedirs(d, exist_ok=True)
        if not _meta_written:
            mp = os.path.join(d, "meta.json")
            tmp = f"{mp}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(_meta(), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, mp)
            _meta_written = True
        if _writer is None:
            # maxsize bounds the transient host copies (~3 GiB at Kimi TP4
            # layer sizes) and back-pressures the build if the disk lags.
            _writer = queue.Queue(maxsize=2)
            _writer_thread = threading.Thread(
                target=_writer_loop, daemon=True,
                name="moe-w2-planes-cache")
            _writer_thread.start()
        for part, t in tensors.items():
            if t is None:
                continue
            path = os.path.join(d, f"layer{layer_idx}.{part}.bin")
            _writer.put((path, t.detach().reshape(-1).cpu()))
    except Exception as e:  # noqa: BLE001
        _mark_broken(f"store failed: {e}")


def shutdown() -> None:
    """Drain the background writer and reset process-global cache state."""
    global _meta_written, _writer, _writer_thread, _broken
    writer, thread = _writer, _writer_thread
    if writer is not None:
        deadline = time.monotonic() + 60.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "moe_w2 planes-cache writer queue did not drain "
                    "within 60 s")
            try:
                writer.put(None, timeout=min(1.0, remaining))
                break
            except queue.Full:
                continue
    if thread is not None and thread.is_alive():
        thread.join(timeout=max(deadline - time.monotonic(), 0.0)
                    if writer is not None else 60.0)
        if thread.is_alive():
            raise RuntimeError(
                "moe_w2 planes-cache writer did not stop within 60 s")
    _writer = None
    _writer_thread = None
    _meta_written = False
    _broken = False
    _ckpt_id.cache_clear()
