# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts on 2-bit tensor-sym planes (cubit moe_w2) for the
DeepSeek-V4 / GLM-5.x MoE family.

Opt-in via VLLM_MOE_W2=1. Replaces the stock routed-expert GEMM path:

  weights : checkpoint mxfp4 e2m1 codes -> {-4,-1,1,4} 2-bit planes built on
            GPU at load (QUANT_PROBE tensor-sym K=4: acceptance 2.73 vs 2.68
            baseline, 12/12 coherent; the sign-sym finding reproduces on
            GLM-5.2 — internal/glm52-sweep). Block-32 UE8M0 scale bytes
            verbatim. FP8 block-quant checkpoints (DS4-FP8, GLM-5.2-FP8) are
            re-quantized at load via build_layer_planes_fp8.
  compute : cubit `moe_w2_mm` SASS GEMM (M<=4 per pair, PRMT-LUT decode,
            QMMA.SF block-32 sfb, f32 act-scale fold) for BOTH w13 and w2.
  glue    : moe_align_block_size(block=4) pairs, fp8 group-128 activation
            quant, silu*up in torch, weighted scatter-add unpermute. All
            steps are tensor ops or driver launches on the current stream:
            CUDA-graph capturable, registered as one custom op.

VRAM: planes+scales ~1.73 GiB/layer (vs ~3.2 GiB raw fp4) -> 43 layers fit
a single 96 GB SM120 board together with the fp8 dense stack and KV.
The MTP drafter keeps the stock DeepGEMM-MXFP4 path: layer names containing
"mtp" are excluded, matching the QUANT_PROBE protocol (drafter unmodified).
"""

import ctypes
import functools
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    mxfp4_to_codes,
    pack_fragment_major,
    pack_scales,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_KERN = b"moe_w2_mm"
_DIR = os.getenv("VLLM_MOE_W2_CUBIT_DIR", "/cubit-share")
_BLOCK = 4                      # tokens per pair == kernel M limit
_NTHR = 256                     # NWARP=8 (K>=1024)


def _nwarp_for_k(k: int) -> int:
    """Split-K warp count baked into each cubin by gen_moe_w2.py (KSLICE=K/NWARP
    must be a multiple of 128). K>=1024 -> 8 warps; K=512 (the w2 GEMM under TP4)
    shards to 4. The launch block MUST match the cubin or the extra warps index
    past K (KSLICE*wid) and read garbage. Mirrors the generator's `_nwarp`."""
    nb = k // 128
    cap = 8 if k >= 1024 else 4
    for n in range(min(cap, nb), 0, -1):
        if nb % n == 0:
            return n
    return 1

_cu = None
_fns: dict = {}
_state = "uninit"
# PREFILL LEVER (default ON since the mc4afrag cubins ship): fragment-major
# activations so each lane's m16k32 QMMA A-fragment loads in ONE LDG.128 (vs 8
# strided 4-byte loads). Profile showed prefill moe_w2_mm is L1/load-issue bound
# (NOT weight-DRAM bound), so this cuts the dominant load class ~4x at identical
# occupancy -> measured 1.30x (K=4096) / 1.27x (K=2048) on the prefill GEMM.
# Numerics are bit-identical to mc4. Needs moe_w2_mm_mc4afrag_k{K}.cubin present
# (loader degrades to mc4 when missing). Opt out: VLLM_MOE_W2_AFRAG=0.
_AFRAG = os.getenv("VLLM_MOE_W2_AFRAG", "1") == "1"
_afrag_ok = False


def _to_fragment_major(a: torch.Tensor, pairs: int, K: int) -> torch.Tensor:
    """[pairs*16, K] fp8 row-major -> fragment-major per 16-token tile (matches the
    AFRAG kernel layout / tools.moe_w2_prefill_bench.pack_a_fragment_major):
    dims [pair, g2, g, j, quad, t, b] -> [pair, j, g, t, quad, g2, b].

    `a` MUST have EXACTLY pairs*16 rows (complete tiles). Callers pass the
    tile-aligned region ws['a1'][:pairs*16] -- NOT ws['a1'][:slots] (slots is the
    over-allocated, non-16-multiple sorted_ids size)."""
    assert a.shape[0] == pairs * 16, (a.shape, pairs)
    v = a.view(torch.uint8).view(pairs, 2, 8, K // 64, 4, 4, 4)
    v = v.permute(0, 3, 2, 5, 4, 1, 6).reshape(pairs * 16, K)
    return v.contiguous().view(a.dtype)

# layer_key -> dict(planes13, sc13, planes2, sc2, top_k, inter)
_LAYERS: dict[int, dict] = {}
_WS: dict = {}                  # shared workspaces, sized lazily

# ---- adaptive expert top-p (VLLM_MOE_W2_TOPP, colibri's --topp) ----------
# Keep each token's routed experts only up to cumulative router weight p:
# the tail of the top-k carries little output mass but full fetch/compute
# cost. Measured on colibri (GLM-5.2, same sigmoid+norm_topk router family):
# p=0.7 cut expert loads 30-40% and bought 1.6x end-to-end on a cold cache.
# Here it shrinks the per-step expert union — on the BASE cache that is
# fewer misses and fewer replay triggers; GPU-resident it is less HBM
# traffic. 0 (default) = off, exact stock routing.
#   VLLM_MOE_W2_TOPP        cumulative-weight cutoff p in (0,1)
#   VLLM_MOE_W2_TOPP_MIN    experts always kept per token (default 2)
#   VLLM_MOE_W2_TOPP_RENORM 1 (default): renormalize kept weights so the
#                           token's total routed weight is preserved
#                           (colibri semantics for norm_topk models);
#                           0: keep original weights (mass shrinks).
_TOPP = float(os.getenv("VLLM_MOE_W2_TOPP", "0"))
_TOPP_MIN = max(1, int(os.getenv("VLLM_MOE_W2_TOPP_MIN", "2")))
_TOPP_RENORM = os.getenv("VLLM_MOE_W2_TOPP_RENORM", "1") == "1"


def _apply_topp(topk_weights: torch.Tensor, topk_ids: torch.Tensor):
    """Drop each token's routed-weight tail past cumulative fraction _TOPP.

    Dropped entries get weight 0 and their expert id REDIRECTED to the
    token's heaviest expert — the redirected pair never fetches a new
    expert (top-1 is always kept) and its zero weight makes the unpermute
    contribution exactly zero, so the drop needs no kernel changes and is
    invisible to moe_align/desc. Pure static-shape tensor ops:
    CUDA-graph-capture-safe. Returns (weights, ids) untouched when off."""
    k = topk_ids.shape[1]
    if not (0.0 < _TOPP < 1.0) or k <= _TOPP_MIN:
        return topk_weights, topk_ids
    w = topk_weights.float()
    order = torch.argsort(w, dim=1, descending=True)
    w_sorted = w.gather(1, order)
    cum = torch.cumsum(w_sorted, dim=1)
    tot = cum[:, -1:]
    # keep ranks whose PRECEDING cumulative mass is still below p*tot
    # (the first expert crossing the threshold is kept, colibri semantics)
    keep_sorted = (cum - w_sorted) < (_TOPP * tot)
    keep_sorted[:, :_TOPP_MIN] = True
    keep = torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
    if _TOPP_RENORM:
        kept_sum = (w * keep).sum(dim=1, keepdim=True).clamp_min(1e-20)
        w = w * (tot / kept_sum)
    top1 = topk_ids.gather(1, order[:, :1])
    new_ids = torch.where(keep, topk_ids, top1.expand_as(topk_ids))
    new_w = torch.where(keep, w, torch.zeros_like(w)).to(topk_weights.dtype)
    return new_w, new_ids


def enabled() -> bool:
    return os.getenv("VLLM_MOE_W2", "0") == "1"


@functools.cache
def _layer_cutoff() -> int:
    """Main-stack layer count: layers >= this are the MTP drafter. Taken from
    the model config when available (43 for DS4-Flash, 78 for GLM-5.2, 61 for
    Kimi-K2.7); VLLM_MOE_W2_NUM_LAYERS overrides.

    get_text_config() unwraps composite VLM configs (KimiK25Config keeps
    num_hidden_layers on .text_config; a bare hf_config lookup would raise
    and silently fall back to 43, sending layers 43+ down the stock path);
    for text-only configs it returns self."""
    v = os.getenv("VLLM_MOE_W2_NUM_LAYERS")
    if v is not None:
        return int(v)
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config().model_config.hf_config
        cfg = cfg.get_text_config()
        n = cfg.num_hidden_layers
        if n:
            return int(n)
    except Exception:  # noqa: BLE001
        pass
    return 43


def is_w2_layer(layer_name: str) -> bool:
    """Main-model routed experts only. The MTP drafter (layer index >=
    num_hidden_layers, e.g. model.layers.43.* for the 43-layer main stack)
    keeps its original path: QUANT_PROBE's acceptance numbers were
    measured with the drafter unmodified."""
    if not enabled():
        return False
    name = layer_name or ""
    if "mtp" in name:
        return False
    import re
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m is None:
        return False
    return int(m.group(1)) < _layer_cutoff()


def _driver():
    global _cu
    if _cu is None:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p]
        cu.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_void_p, ctypes.c_char_p]
        _cu = cu
    return _cu


def _ck(r, what):
    if r:
        raise RuntimeError(f"moe_w2_cubit: CUDA error {r} in {what}")


def _ensure_ready() -> bool:
    global _state
    if _state == "ready":
        return True
    if _state == "unavailable":
        return False
    try:
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        cu = _driver()
        for tier, kern in (("w2", b"moe_w2_mm"), ("w4", b"moe_w4_mm"),
                           ("w4q", b"moe_w4q_mm"),
                           ("w2mc2", b"moe_w2_mm"), ("w2mc4", b"moe_w2_mm")):
            # GEMM contraction K: gate-up needs K=hidden (4096 DS4-Flash,
            # 6144 GLM-5.x, 7168 Kimi-K2.x); down needs K=I/TP (2048 @ TP1,
            # 1024 @ TP2, 512 @ TP4). Cubins are loaded opportunistically --
            # the plane builders assert the shapes the model actually needs
            # are present (_assert_kernels fails loudly at weight load).
            for k in (7168, 6144, 4096, 2048, 1024, 512):
                if tier in ("w2mc2", "w2mc4"):
                    fname = f"moe_w2_mm_{tier[2:]}_k{k}.cubin"
                else:
                    fname = f"moe_{tier}_mm_k{k}.cubin"
                path = os.path.join(_DIR, fname)
                if not os.path.exists(path):
                    continue
                mod = ctypes.c_void_p()
                _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                    f"cuModuleLoad {path}")
                fn = ctypes.c_void_p()
                _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, kern),
                    "cuModuleGetFunction")
                _fns[(tier, k)] = fn
        global _afrag_ok
        if _AFRAG:
            try:
                for k in (7168, 6144, 4096, 2048, 1024, 512):
                    path = os.path.join(_DIR, f"moe_w2_mm_mc4afrag_k{k}.cubin")
                    if not os.path.exists(path):
                        continue
                    mod = ctypes.c_void_p()
                    _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                        f"cuModuleLoad {path}")
                    fn = ctypes.c_void_p()
                    _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"moe_w2_mm"),
                        "cuModuleGetFunction afrag")
                    _fns[("w2mc4afrag", k)] = fn
                _afrag_ok = True
                logger.info("moe_w2_cubit: AFRAG prefill cubins loaded")
            except Exception as e:  # noqa: BLE001
                logger.warning("moe_w2_cubit: AFRAG unavailable (%s); using mc4", e)
                _afrag_ok = False
        _state = "ready"
        logger.info("moe_w2_cubit: cubins loaded: %s", sorted(_fns))
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("moe_w2_cubit unavailable: %s", e)
        _state = "unavailable"
        return False


# --------------------------------------------------------------------------
# Load-time plane building
# --------------------------------------------------------------------------

def _require_kernels(K13: int, K2: int, need_w4: bool) -> None:
    """Fail loudly at weight load when the cubins this model's shapes need are
    missing from _DIR (they are loaded opportunistically in _ensure_ready)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    need = [("w2", K13), ("w2", K2), ("w2mc4", K13), ("w2mc4", K2)]
    if need_w4:
        w4tier = "w4q" if moe_w2_delta.split_enabled() else "w4"
        need += [(w4tier, K13), (w4tier, K2)]
    missing = [f"{t}_k{k}" for t, k in need if (t, k) not in _fns]
    assert not missing, (
        f"moe_w2_cubit: missing cubins for K13={K13}/K2={K2}: {missing} "
        f"(dir {_DIR}; set VLLM_MOE_W2_CUBIT_DIR)")


def _fp4_tier_for_build(E: int, dev, n13k13: int, n2k2: int):
    """FP4 delta tier sized for this model's PER-RANK shapes (n13k13 =
    N13*K13, n2k2 = N2*K2 elements). Over the base cache the FP4 slots must
    carry their OWN block-32 scale sections ([fp4_13|sc13|fp4_2|sc2]) — the
    base planes, and with them the GPU-resident scale planes the standalone
    delta shares, are host-resident there. Split mode (DELTA_SPLIT): slots
    hold RADIX-5 quintal planes (2.5 bits/elem, bit-exact e2m1 — see
    moe_w2_planes.pack_quintal_fragment_major) and NO scale sections even
    over the base cache — the quintal kernel reads class/sign/scales from
    the base slot the refinement is residency-coupled to."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    split = moe_w2_delta.split_enabled()
    sc13, sc2 = ((n13k13 // 32, n2k2 // 32)
                 if moe_w2_delta.base_enabled() and not split else (0, 0))
    if split:
        return moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                     w13_bytes=n13k13 * 5 // 16,
                                     w2_bytes=n2k2 * 5 // 16)
    return moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                 w13_bytes=n13k13 // 2 + sc13,
                                 w2_bytes=n2k2 // 2 + sc2)


def _stage_fp4_host(tier, layer_key: int, fp13, sc13, fp2, sc2) -> None:
    """Stage a layer's FP4 planes into the tier's pinned host store; over the
    base cache the scale planes ride along inside the slot sections (copied
    section-by-section — no GPU-side cat temporaries). Split slots carry
    refinement only — their scales live in the coupled base slot."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    if moe_w2_delta.base_enabled() and not moe_w2_delta.split_enabled():
        tier.add_layer_host_sections(layer_key, (fp13, sc13), (fp2, sc2))
    else:
        tier.add_layer_host_planes(layer_key, fp13, fp2)


def _pack_fp4_plane(nib):
    """One expert's FP4-tier plane row from its e2m1 nibbles: the full
    fragment-major nibble plane (moe_w4_mm), or — split mode — the radix-5
    QUINTAL plane (moe_w4q_mm reads it alongside the resident base;
    bit-exact e2m1 at 2.5 bits/elem)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        pack_fp4_fragment_major, pack_quintal_fragment_major)
    if moe_w2_delta.split_enabled():
        return pack_quintal_fragment_major(nib)
    return pack_fp4_fragment_major(nib)


def _fp4_plane_nbytes(n: int, k: int) -> int:
    """Per-expert FP4-tier plane bytes for one [n, k] matrix: nibbles
    (n*k/2) or the split quintal plane (n*k*5/16)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    return n * k * 5 // 16 if moe_w2_delta.split_enabled() else n * k // 2


# Loader-level skip (the planes-cache/pack "v1.5 follow-up"): layers the
# pack already serves never need their checkpoint experts in host RAM at
# all. Decided at CREATE time (sidecar probe), executed by stubbing the big
# params + no-op'ing their weight loaders — the vLLM loader then streams
# past those tensors without allocating or copying. Measured motivation
# (GLM-5.2 TP2 boot-from-pack): ~190 s of giant host allocations before the
# shard read, 222 s of staged copies during it, and a ~0.5 TB transient
# that intermittently OOM'd the box when boots overlapped.
_n_created = 0
_skip_logged = False


def _noop_loader(*args, **kwargs):
    """Weight-loader stand-in for pack-skipped params: the expert loading
    loop calls with return_success=True and must see truthy, or it treats
    the shard as unmapped and keeps probing replicas."""
    return True if kwargs.get("return_success") else None


def plan_pack_skip(layer) -> bool:
    """CREATE-time twin of the boot-from-cache paths: assign this layer's
    key (the same build-order counter process_weights_after_loading uses),
    probe whichever store this config will serve from — the pack sidecars
    (BASE cache / host-resident) or the planes cache (GPU-resident) — and
    when the layer is already served, stub the four big params and disarm
    their loaders. Returns True when the layer boots with zero checkpoint
    staging."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as _pc)
    from vllm.model_executor.layers.quantization.utils.moe_w2_store import (
        pack_has_layer)
    global _n_created
    key = _n_created
    _n_created += 1
    layer._moe_w2_create_key = key
    if not enabled():
        return False
    try:
        E, N13, K13h = layer.w13_weight.shape
        _, N2, K2h = layer.w2_weight.shape
    except Exception:  # noqa: BLE001 - unexpected layout: stage as before
        return False
    K13, K2 = K13h * 2, K2h * 2
    c13len, s13len = N13 * K13 // 4, N13 * K13 // 32
    c2len, s2len = N2 * K2 // 4, N2 * K2 // 32
    if moe_w2_delta.base_enabled():
        # host-resident: the pack store serves the base (and the FP4
        # need-pool when configured) — probe the sidecars.
        n_keys = _layer_cutoff() + 1
        if not pack_has_layer("base", key, n_keys, E,
                              c13len + s13len + c2len + s2len):
            return False
        if moe_w2_delta.enabled():
            # over-base FP4 need-pool: mirror _fp4_tier_for_build's sizing
            # and get_tier's pack tag (split quintal slots live in a
            # SEPARATE pack — different geometry, "fp4q")
            if moe_w2_delta.split_enabled():
                ftag = "fp4q"
                fslot = N13 * K13 * 5 // 16 + N2 * K2 * 5 // 16
            else:
                ftag = "fp4"
                fslot = (N13 * K13 // 2 + s13len) + (N2 * K2 // 2 + s2len)
            if not pack_has_layer(ftag, key, n_keys, E, fslot):
                return False
    else:
        # GPU-resident: the planes cache is the only source that can
        # replace the checkpoint requant — probe it (keyed by transformer
        # layer index, sized exactly like process-time try_load).
        lidx = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
        if lidx is None or not _pc.cache_has_layer(
                lidx, _pc.expected_sizes(
                    E, N13, K13, N2, K2, want_fp4=moe_w2_delta.enabled())):
            return False
    for pname in ("w13_weight", "w13_weight_scale",
                  "w2_weight", "w2_weight_scale"):
        p = getattr(layer, pname)
        p.data = torch.empty(0, dtype=p.data.dtype, device="cpu")
        p.weight_loader = _noop_loader
    layer._moe_w2_shapes = (E, N13, K13, N2, K2)
    layer._moe_w2_pack_skip = True
    global _skip_logged
    if not _skip_logged:
        _skip_logged = True
        logger.info(
            "moe_w2 LOADER-SKIP armed: pack-resident expert layers are "
            "neither host-staged nor copied from the checkpoint "
            "(first: key %d)", key)
    logger.debug("moe_w2: layer key %d loader-skipped", key)
    return True


# ---- streaming FIRST boot (VLLM_MOE_W2_STREAM_BUILD, default on) ---------
# With no pack and no planes cache to skip from, the loader used to stage
# the FULL expert checkpoint in host RAM before a single layer was
# requantized — ~400+ GB transient on GLM-5.2, reported as a 4-5 h
# swap-through first boot on a 354 GB host. Streaming build requants each
# layer the moment its LAST expected expert tensor lands (exact per-param
# load counting, no ordering assumptions) and stubs its staging right
# after: peak staging = O(one layer) ≈ 6 GB instead of the checkpoint.
# The GPU is idle during load anyway, so the per-layer requant overlaps
# shard I/O instead of serializing after it. VLLM_MOE_W2_STREAM_BUILD=0
# restores the stage-everything-then-build behaviour.
_STREAM = os.getenv("VLLM_MOE_W2_STREAM_BUILD", "1") == "1"
_stream_logged = False


class _StreamLoader:
    """Per-param weight_loader wrapper: counts SUCCESSFUL (expert, shard)
    loads and triggers the layer build when every big param is complete.
    A load arriving after the build would be silent data loss (the params
    are stubs by then) — fail loudly instead."""

    def __init__(self, layer, pname, inner):
        self._layer = layer
        self._pname = pname
        self._inner = inner

    def __call__(self, param, loaded_weight, *args, **kwargs):
        # LAZY staging: the create hook left this param as a 0-byte stub
        # (allocating every layer's host buffer up front peaked at ~300+ GB
        # before the first shard was even read — measured). Materialize the
        # layer's buffer the moment its first tensor arrives; with a
        # layer-major checkpoint only a couple of layers are ever
        # in-flight, an unordered one merely degrades to the old profile.
        if param.data.numel() == 0:
            shape = self._layer._moe_w2_stream_shapes[self._pname]
            param.data = torch.empty(shape, dtype=param.data.dtype,
                                     device="cpu")
        ret = self._inner(param, loaded_weight, *args, **kwargs)
        ok = (ret is True) if kwargs.get("return_success") else True
        if not ok:
            return ret
        pend = self._layer._moe_w2_pending
        if pend.get(self._pname, 0) <= 0:
            raise RuntimeError(
                f"moe_w2 stream-build: {self._pname} load arrived after "
                f"the layer was already built — more (expert, shard) "
                f"tensors than expected; set VLLM_MOE_W2_STREAM_BUILD=0 "
                f"and report the checkpoint")
        pend[self._pname] -= 1
        if all(v == 0 for v in pend.values()):
            key = self._layer._moe_w2_create_key
            build_layer_planes_nvfp4(self._layer, key)
            # Drop the staging storage IN PLACE on the ORIGINAL Parameter
            # objects. _finish_layer replaced the layer's attributes with
            # stub Parameters, but load_weights' params_dict (built once,
            # up front) still references the originals for the rest of the
            # load — without this, nothing frees until load_weights returns
            # and the "streaming" peak is the whole checkpoint again
            # (measured: 517 GB at 29/47 shards on GLM TP2).
            for p in self._layer._moe_w2_stream_orig:
                p.data = torch.empty(0, dtype=p.data.dtype, device="cpu")
            self._layer._moe_w2_stream_orig = ()
            self._layer._moe_w2_stream_built = True
            logger.debug("moe_w2: layer key %d stream-built during load",
                         key)
        return ret


def arm_stream_build(layer) -> bool:
    """Arm the streaming per-layer build on a layer plan_pack_skip missed
    (first boot, or a store that does not yet hold it). Expected loads per
    param: w13-side shards land twice per expert (w1, w3), w2-side once —
    exact counts over ALL SIX params the requant reads (including scale_2:
    building before they land would bake uninitialized per-tensor scales
    into the planes AND the caches), so completeness needs no
    checkpoint-ordering assumption.

    Staging is LAZY: the four big params become 0-byte stubs here and a
    layer's buffers materialize on its FIRST loaded tensor (the up-front
    create_weights allocation of every layer peaked at ~300 GB before the
    first shard was read — measured), then drop IN PLACE at build (the
    attribute swap alone frees nothing: load_weights' params_dict holds
    the original objects until the load returns — measured 517 GB).
    Returns True when armed (the caller then skips its own staging).
    No-op unless VLLM_MOE_W2_STREAM_BUILD=1 (default)."""
    global _stream_logged
    if not (_STREAM and enabled()):
        return False
    try:
        E = layer.w13_weight.shape[0]
    except Exception:  # noqa: BLE001 - unexpected layout: staged path
        return False
    expected = {"w13_weight": 2 * E, "w13_weight_scale": 2 * E,
                "w13_weight_scale_2": 2 * E,
                "w2_weight": E, "w2_weight_scale": E,
                "w2_weight_scale_2": E}
    big = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    wrappers = {}
    for pname in expected:
        p = getattr(layer, pname, None)
        inner = getattr(p, "weight_loader", None)
        if p is None or inner is None:
            return False                # leave the layer fully staged
        wrappers[pname] = (p, _StreamLoader(layer, pname, inner))
    layer._moe_w2_pending = expected
    # originals of the BIG params: their storage is dropped in place at
    # build time, and their SHAPES feed the lazy materialization
    layer._moe_w2_stream_orig = tuple(getattr(layer, p) for p in big)
    layer._moe_w2_stream_shapes = {
        p: tuple(getattr(layer, p).shape) for p in big}
    for pname in big:                   # lazy: nothing staged until loaded
        p = getattr(layer, pname)
        p.data = torch.empty(0, dtype=p.data.dtype, device="cpu")
    for pname, (p, wrap) in wrappers.items():
        p.weight_loader = wrap
    if not _stream_logged:
        _stream_logged = True
        logger.info(
            "moe_w2 STREAM-BUILD armed: layer staging materializes on its "
            "first loaded tensor and requants on its last (peak staging = "
            "layers in flight, not the checkpoint); "
            "VLLM_MOE_W2_STREAM_BUILD=0 restores the old path")
    return True


def _try_skip_requant(layer, layer_key: int, E: int, N13: int, K13: int,
                      N2: int, K2: int, param_names) -> bool:
    """Boot-from-pack: when every host store this config serves from already
    holds this layer's rows (valid pack written by a previous boot), the
    dequant->requant of the checkpoint experts produces bytes NOBODY reads —
    the base planes live in the pack, and so do the FP4 need-pool sections.
    Skip it: register the layer's slot-layout metadata and the param stubs
    exactly as _finish_layer's base path would, and let the tiers serve
    from the pack. On GLM the requant is the dominant boot cost (NVFP4 ->
    f64 -> 2-bit, hundreds of GiB of transients); with the pack it reduces
    to open+read.

    Only applies over the base cache (base_enabled): the GPU-resident plane
    path needs the planes materialized regardless. A PinnedHostStore never
    contains layers at boot -> configs without VLLM_MOE_W2_STORE_DIR are
    untouched. Layers absent from the pack (e.g. the MTP drafter, or a
    partially written pack) requant as before. When the FP4 tier is enabled
    but ITS pack misses the layer, we also requant (the fp4 sections can
    only be rebuilt from the checkpoint bytes)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    if not moe_w2_delta.base_enabled():
        return False
    dev = torch.device("cuda")
    c13len, s13len = N13 * K13 // 4, N13 * K13 // 32
    c2len, s2len = N2 * K2 // 4, N2 * K2 // 32
    btier = moe_w2_delta.get_base_tier(
        _layer_cutoff() + 1, E, dev,
        w13_bytes=c13len + s13len, w2_bytes=c2len + s2len)
    if layer_key not in btier._store:
        return False
    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    if tier is not None and layer_key not in tier._store:
        return False
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as _pc)
    _LAYERS[layer_key] = dict(
        N13=N13, K13=K13, N2=N2, K2=K2, E=E, base=True,
        tl_idx=_pc.layer_idx_from_name(getattr(layer, "layer_name", "")),
        off_s13=c13len, off_c2=c13len + s13len,
        off_s2=c13len + s13len + c2len,
        off4_s13=2 * c13len, off4_c2=2 * c13len + s13len,
        off4_s2=2 * c13len + s13len + 2 * c2len,
    )
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in param_names:
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info(
        "moe_w2: layer %d requant SKIPPED — %s serving from pack "
        "(boot-from-pack)", layer_key,
        "base+fp4" if tier is not None else "base")
    return True


def build_layer_planes(layer, layer_key: int) -> None:
    """Quantize one FusedMoE layer's experts to 2-bit planes (GPU, chunked).

    Reads the CPU-resident checkpoint params (w13_weight [E,2I,K/2] u8 etc.),
    builds fragment-major code planes + scale planes on the GPU, then
    replaces the originals with empty stubs.
    """
    assert _ensure_ready(), "moe_w2 cubins missing"
    dev = torch.device("cuda")
    w13 = layer.w13_weight.data          # [E, 2I, H/2] u8 (cpu)
    s13 = layer.w13_weight_scale.data    # [E, 2I, H/32] u8
    w2 = layer.w2_weight.data            # [E, H, I/2] u8
    s2 = layer.w2_weight_scale.data      # [E, H, I/32] u8
    E, N13, _ = w13.shape
    _, N2, _ = w2.shape
    K13, K2 = N2, N13 // 2               # H, I (4096/2048 on DS4-Flash TP1)
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())
    if _try_skip_requant(layer, layer_key, E, N13, K13, N2, K2,
                         ("w13_weight", "w13_weight_scale", "w2_weight",
                          "w2_weight_scale")):
        return

    planes13 = torch.empty(E, N13 * K13 // 4, dtype=torch.uint8, device=dev)
    sc13 = torch.empty(E, N13 * K13 // 32, dtype=torch.uint8, device=dev)
    planes2 = torch.empty(E, N2 * K2 // 4, dtype=torch.uint8, device=dev)
    sc2 = torch.empty(E, N2 * K2 // 32, dtype=torch.uint8, device=dev)

    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        mxfp4_to_nibbles, pack_fp4_fragment_major)
    # Pass the PER-RANK FP4 plane sizes (N*K//2 bytes/expert) so the delta tier's
    # slots, host store, and pool indexing match the (TP-sharded) planes. On TP1
    # these equal the module constants -> the single-GPU path is unchanged.
    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    fp13 = fp2 = None
    if tier is not None:
        # full nibble planes (w4) or quintal planes (w4q, split)
        fp13 = torch.empty(E, _fp4_plane_nbytes(N13, K13),
                           dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, _fp4_plane_nbytes(N2, K2),
                          dtype=torch.uint8, device=dev)

    chunk = 32
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = w13[e0:e1].to(dev, non_blocking=True)
        sg = s13[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            nib = mxfp4_to_nibbles(wg[i])
            planes13[e0 + i] = pack_fragment_major(mxfp4_to_codes(wg[i]))
            sc13[e0 + i] = pack_scales(sg[i])
            if fp13 is not None:
                fp13[e0 + i] = _pack_fp4_plane(nib)
        wg = w2[e0:e1].to(dev, non_blocking=True)
        sg = s2[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            nib = mxfp4_to_nibbles(wg[i])
            planes2[e0 + i] = pack_fragment_major(mxfp4_to_codes(wg[i]))
            sc2[e0 + i] = pack_scales(sg[i])
            if fp2 is not None:
                fp2[e0 + i] = _pack_fp4_plane(nib)

    if tier is not None:
        _stage_fp4_host(tier, layer_key, fp13, sc13, fp2, sc2)
        del fp13, fp2
        # (the background manager is started by get_tier when the tier is
        # created; the old "start on layer NUM_LAYERS-1" trigger never fired
        # under PP, where layer_keys are local per rank and never reach 42)

    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", "w13_weight_scale", "w2_weight",
                   "w2_weight_scale"))


def build_layer_planes_fp8(layer, layer_key: int,
                           scale_suffix: str = "weight_scale_inv") -> None:
    """FP8 block-quant checkpoint variant of build_layer_planes (Fp8MoEMethod:
    DS4-Flash-FP8, GLM-5.2-FP8 — models without an FP4 release).

    Reads the CPU-staged fp8 params (w13_weight [E,2I,H] e4m3 +
    w13_weight_scale_inv [E,ceil(2I/128),ceil(H/128)] f32 etc.), re-quantizes
    each expert on GPU to the sweep-validated 2-bit pipeline (block-32 UE8M0 +
    e2m1 snap + tensor-sym {-4,-1,1,4}; internal/glm52-sweep/sweep.py), packs
    fragment-major planes, then replaces the originals with empty stubs. The
    e2m1 nibbles of the same requant feed the optional FP4 delta tier.
    """
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        fp8_block_to_codes_scales, pack_fp4_fragment_major)

    assert _ensure_ready(), "moe_w2 cubins missing"
    dev = torch.device("cuda")
    w13 = layer.w13_weight.data                       # [E, 2I, H] e4m3 (cpu)
    s13 = getattr(layer, f"w13_{scale_suffix}").data  # [E, 2I/128, H/128] f32
    w2 = layer.w2_weight.data                         # [E, H, I] e4m3
    s2 = getattr(layer, f"w2_{scale_suffix}").data    # [E, H/128, I/128] f32
    assert w13.dtype == torch.float8_e4m3fn, w13.dtype
    E, N13, K13 = w13.shape
    _, N2, K2 = w2.shape
    _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())
    if _try_skip_requant(layer, layer_key, E, N13, K13, N2, K2,
                         ("w13_weight", f"w13_{scale_suffix}", "w2_weight",
                          f"w2_{scale_suffix}")):
        return

    planes13 = torch.empty(E, N13 * K13 // 4, dtype=torch.uint8, device=dev)
    sc13 = torch.empty(E, N13 * K13 // 32, dtype=torch.uint8, device=dev)
    planes2 = torch.empty(E, N2 * K2 // 4, dtype=torch.uint8, device=dev)
    sc2 = torch.empty(E, N2 * K2 // 32, dtype=torch.uint8, device=dev)

    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    fp13 = fp2 = None
    if tier is not None:
        # full nibble planes (w4) or quintal planes (w4q, split)
        fp13 = torch.empty(E, _fp4_plane_nbytes(N13, K13),
                           dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, _fp4_plane_nbytes(N2, K2),
                          dtype=torch.uint8, device=dev)

    # fp8 experts are 4x the bytes of the mxfp4 path and the requant makes f32
    # temporaries -> smaller H2D chunks, per-expert quantize.
    chunk = 8
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = w13[e0:e1].to(dev, non_blocking=True)
        sg = s13[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            codes, sbytes, nib = fp8_block_to_codes_scales(
                wg[i], sg[i], want_nibbles=fp13 is not None)
            planes13[e0 + i] = pack_fragment_major(codes)
            sc13[e0 + i] = pack_scales(sbytes)
            if fp13 is not None:
                fp13[e0 + i] = _pack_fp4_plane(nib)
        wg = w2[e0:e1].to(dev, non_blocking=True)
        sg = s2[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            codes, sbytes, nib = fp8_block_to_codes_scales(
                wg[i], sg[i], want_nibbles=fp2 is not None)
            planes2[e0 + i] = pack_fragment_major(codes)
            sc2[e0 + i] = pack_scales(sbytes)
            if fp2 is not None:
                fp2[e0 + i] = _pack_fp4_plane(nib)

    if tier is not None:
        _stage_fp4_host(tier, layer_key, fp13, sc13, fp2, sc2)
        del fp13, fp2

    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", f"w13_{scale_suffix}", "w2_weight",
                   f"w2_{scale_suffix}"))


def _consume_planes_cache(layer, layer_key: int, dev,
                          E: int, N13: int, K13: int, N2: int,
                          K2: int) -> bool:
    """Serve one layer's planes from the planes cache (GPU-resident
    configs). CPU tensors from the cache feed the same _stage_fp4_host/
    _finish_layer sinks as a fresh requant (their copy_ calls are
    device-agnostic). Shared by the staged path (cache hit replaces the
    requant) and the loader-skip path (stubs; the cache is the ONLY
    source). Returns True on a hit."""
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as planes_cache)
    lidx = planes_cache.layer_idx_from_name(getattr(layer, "layer_name", ""))
    if not planes_cache.enabled() or lidx is None:
        return False
    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    cached = planes_cache.try_load(lidx, planes_cache.expected_sizes(
        E, N13, K13, N2, K2, want_fp4=tier is not None))
    if cached is None:
        return False
    planes13 = cached["planes13"].view(E, -1).to(dev)
    sc13 = cached["sc13"].view(E, -1).to(dev)
    planes2 = cached["planes2"].view(E, -1).to(dev)
    sc2 = cached["sc2"].view(E, -1).to(dev)
    if tier is not None:
        _stage_fp4_host(tier, layer_key, cached["fp13"].view(E, -1),
                        sc13, cached["fp2"].view(E, -1), sc2)
    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2,
                  sc2, N13, K13, N2, K2, E,
                  ("w13_weight", "w13_weight_scale", "w2_weight",
                   "w2_weight_scale"))
    logger.info("moe_w2: layer %d planes from cache", lidx)
    return True


def build_layer_planes_nvfp4(layer, layer_key: int) -> None:
    """NVFP4 (modelopt) checkpoint variant of build_layer_planes
    (ModelOptNvFp4FusedMoE: nvidia/GLM-5.2-NVFP4 — e2m1 codes + e4m3
    block-16 scales + per-tensor scale_2).

    Reads the CPU-staged params (w13_weight [E,2I,H/2] u8 packed +
    w13_weight_scale [E,2I,H/16] e4m3 + w13_weight_scale_2 [E,2] f32 etc.),
    dequantizes each expert to f64 on GPU (exact) and re-quantizes to the
    sweep-validated sign-symmetric 2-bit pipeline; the e2m1 nibbles of the
    same requant feed the optional FP4 delta tier. The UE8M0 block-32 output
    scales absorb scale_2, so serving needs no extra per-tensor factor.
    """
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        nvfp4_to_codes_scales, pack_fp4_fragment_major)

    assert _ensure_ready(), "moe_w2 cubins missing"
    dev = torch.device("cuda")
    if getattr(layer, "_moe_w2_pack_skip", False):
        # loader-level skip: the params are 0-byte stubs (plan_pack_skip),
        # shapes travel via the create-time stash. The probed store MUST
        # still serve the layer — there is no checkpoint copy to fall
        # back to.
        E, N13, K13, N2, K2 = layer._moe_w2_shapes
        _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())
        if moe_w2_delta.base_enabled():
            assert _try_skip_requant(
                layer, layer_key, E, N13, K13, N2, K2,
                ("w13_weight", "w13_weight_scale", "w2_weight",
                 "w2_weight_scale")), (
                f"moe_w2: layer {layer_key} was loader-skipped on a pack "
                f"sidecar hit but the pack no longer serves it "
                f"(dir/sidecar changed mid-load?) — restart without the "
                f"stale VLLM_MOE_W2_STORE_DIR state")
            return
        # GPU-resident: materialize the planes from the planes cache
        # (probed at create time; a miss here means the cache dir changed
        # under a live load).
        assert _consume_planes_cache(layer, layer_key, dev,
                                     E, N13, K13, N2, K2), (
            f"moe_w2: layer {layer_key} was loader-skipped on a planes-"
            f"cache hit but the cache no longer serves it — restart "
            f"without the stale VLLM_MOE_W2_PLANES_CACHE state")
        return
    w13 = layer.w13_weight.data                 # [E, 2I, H/2] u8 (cpu)
    s13 = layer.w13_weight_scale.data           # [E, 2I, H/16] e4m3
    s13_2 = layer.w13_weight_scale_2.data       # [E, 2] f32 (w1, w3)
    w2 = layer.w2_weight.data                   # [E, H, I/2] u8
    s2 = layer.w2_weight_scale.data             # [E, H, I/16] e4m3
    s2_2 = layer.w2_weight_scale_2.data         # [E] f32
    assert w13.dtype == torch.uint8 and s13.dtype == torch.float8_e4m3fn, (
        w13.dtype, s13.dtype)
    E, N13, K13h = w13.shape
    K13 = K13h * 2
    _, N2, K2h = w2.shape
    K2 = K2h * 2
    group = K13 // s13.shape[2]                 # 16 for NVFP4
    _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())
    if _try_skip_requant(layer, layer_key, E, N13, K13, N2, K2,
                         ("w13_weight", "w13_weight_scale", "w2_weight",
                          "w2_weight_scale")):
        return

    # Planes cache (VLLM_MOE_W2_PLANES_CACHE): the requant below is
    # deterministic given (checkpoint, TP layout, zero mode), so cached
    # planes can be streamed back instead of rebuilt (~9 min saved on
    # Kimi-K2.7 restarts). Complements the pack store's boot-from-pack
    # above: the cache serves GPU-RESIDENT plane configs (planes must be
    # materialized), the pack store serves host-resident tiers (planes
    # never materialize).
    if _consume_planes_cache(layer, layer_key, dev, E, N13, K13, N2, K2):
        return
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as planes_cache)
    lidx = planes_cache.layer_idx_from_name(getattr(layer, "layer_name", ""))
    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)

    planes13 = torch.empty(E, N13 * K13 // 4, dtype=torch.uint8, device=dev)
    sc13 = torch.empty(E, N13 * K13 // 32, dtype=torch.uint8, device=dev)
    planes2 = torch.empty(E, N2 * K2 // 4, dtype=torch.uint8, device=dev)
    sc2 = torch.empty(E, N2 * K2 // 32, dtype=torch.uint8, device=dev)

    fp13 = fp2 = None
    if tier is not None:
        # full nibble planes (w4) or quintal planes (w4q, split)
        fp13 = torch.empty(E, _fp4_plane_nbytes(N13, K13),
                           dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, _fp4_plane_nbytes(N2, K2),
                          dtype=torch.uint8, device=dev)

    # f64 temporaries are 16x the packed nibbles -> small H2D chunks,
    # per-expert quantize (mirrors the fp8 loader).
    chunk = 8
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = w13[e0:e1].to(dev, non_blocking=True)
        sg = s13[e0:e1].to(dev, non_blocking=True)
        s2g = s13_2[e0:e1].to(dev, non_blocking=True)
        half = N13 // 2                          # rows [0:I]=w1, [I:2I]=w3
        for i in range(e1 - e0):
            s2_row = torch.cat((s2g[i, 0].expand(half), s2g[i, 1].expand(half)))
            codes, sbytes, nib = nvfp4_to_codes_scales(
                wg[i], sg[i], s2_row, group=group,
                want_nibbles=fp13 is not None)
            planes13[e0 + i] = pack_fragment_major(codes)
            sc13[e0 + i] = pack_scales(sbytes)
            if fp13 is not None:
                fp13[e0 + i] = _pack_fp4_plane(nib)
        wg = w2[e0:e1].to(dev, non_blocking=True)
        sg = s2[e0:e1].to(dev, non_blocking=True)
        s2g = s2_2[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            codes, sbytes, nib = nvfp4_to_codes_scales(
                wg[i], sg[i], s2g[i], group=group,
                want_nibbles=fp2 is not None)
            planes2[e0 + i] = pack_fragment_major(codes)
            sc2[e0 + i] = pack_scales(sbytes)
            if fp2 is not None:
                fp2[e0 + i] = _pack_fp4_plane(nib)

    if planes_cache.enabled() and lidx is not None:
        planes_cache.store(lidx, dict(planes13=planes13, sc13=sc13,
                                      planes2=planes2, sc2=sc2,
                                      fp13=fp13, fp2=fp2))

    if tier is not None:
        _stage_fp4_host(tier, layer_key, fp13, sc13, fp2, sc2)
        del fp13, fp2

    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", "w13_weight_scale", "w2_weight",
                   "w2_weight_scale"))


def _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E, param_names) -> None:
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    # transformer layer index of this layer_key (dense-offset models: GLM's
    # first sparse layer 3 -> key 0). LOOKA uses it to pair each key with
    # its transformer layer's router (mlp.gate) weights.
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as _pc)
    _tl = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
    if moe_w2_delta.base_enabled():
        # BASE cache (inverted delta): the 2-bit planes go to PINNED HOST RAM
        # instead of staying GPU-resident; the GPU holds only the base tier's
        # slot pool. Slot layout per expert: [codes13 | sc13 | codes2 | sc2]
        # (the tier's "w13 section" = codes13+sc13, "w2 section" = codes2+sc2,
        # so add_layer_host_planes packs it verbatim).
        c13len, s13len = planes13.shape[1], sc13.shape[1]
        c2len, s2len = planes2.shape[1], sc2.shape[1]
        btier = moe_w2_delta.get_base_tier(
            _layer_cutoff() + 1, E, dev,
            w13_bytes=c13len + s13len, w2_bytes=c2len + s2len)
        btier.add_layer_host_planes(
            layer_key,
            torch.cat((planes13, sc13), dim=1),
            torch.cat((planes2, sc2), dim=1))
        _LAYERS[layer_key] = dict(
            N13=N13, K13=K13, N2=N2, K2=K2, E=E, base=True, tl_idx=_tl,
            off_s13=c13len, off_c2=c13len + s13len,
            off_s2=c13len + s13len + c2len,
            # FP4 need-pool slot sections ([fp4_13|sc13|fp4_2|sc2]; fp4 codes
            # are 2x the 2-bit codes, scale sections identical) — read by the
            # base+delta desc kernel when the FP4 tier coexists.
            off4_s13=2 * c13len, off4_c2=2 * c13len + s13len,
            off4_s2=2 * c13len + s13len + 2 * c2len,
        )
        del planes13, sc13, planes2, sc2
        stub = torch.empty(0, dtype=torch.uint8, device=dev)
        for name in param_names:
            layer.register_parameter(
                name, torch.nn.Parameter(stub, requires_grad=False))
        logger.info("moe_w2: layer %d planes HOST-staged (base cache, "
                    "%.2f GiB pinned)", layer_key,
                    E * btier.slot_bytes / 2**30)
        return

    _LAYERS[layer_key] = dict(
        planes13=planes13, sc13=sc13, planes2=planes2, sc2=sc2,
        N13=N13, K13=K13, N2=N2, K2=K2, E=E, tl_idx=_tl,
    )
    # Release checkpoint copies; keep CUDA stubs so device probes stay happy.
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in param_names:
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info("moe_w2: layer %d planes built (%.2f GiB)", layer_key,
                (planes13.nbytes + sc13.nbytes + planes2.nbytes + sc2.nbytes)
                / 2**30)


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------

def _workspaces(slots: int, tokens: int, dev, inter: int = 2048,
                hidden: int = 4096, n_experts: int = 256) -> dict:
    # `inter` = per-rank expert intermediate size I (2048 on 1 GPU; 1024 @ TP2,
    # 512 @ TP4 as the experts shard). The hidden H (4096 DS4, 6144 GLM-5.x) is
    # NOT sharded, so the A-side (a1), x-quant (xq) and w2 output (c2) buffers
    # stay H-wide; only the gate/up output (c13 = 2I), the intermediate
    # activation (act/a2 = I) and its group-128 scales (as2 = I/128) follow the
    # shard.
    if (_WS.get("slots", 0) < slots or _WS.get("tokens", 0) < tokens
            or _WS.get("inter") != inter or _WS.get("hidden") != hidden
            or _WS.get("n_experts", 0) < n_experts):
        slots = max(slots, _WS.get("slots", 0))
        tokens = max(tokens, _WS.get("tokens", 0))
        n_experts = max(n_experts, _WS.get("n_experts", 0))
        _WS.update(
            slots=slots,
            tokens=tokens,
            inter=inter,
            hidden=hidden,
            n_experts=n_experts,
            # token-side quant buffers; the LAST row is the permanent zero
            # pad row (gather source for filler slots) — quant only ever
            # writes rows [:T].
            xq=torch.zeros(tokens + 1, hidden, dtype=torch.float8_e4m3fn,
                           device=dev),
            xs=torch.zeros(tokens + 1, hidden // 128, dtype=torch.float32,
                           device=dev),
            a1=torch.zeros(slots + 4, hidden, dtype=torch.float8_e4m3fn,
                           device=dev),
            as1=torch.zeros(slots + 4, hidden // 128, dtype=torch.float32,
                            device=dev),
            # zeros, not empty: pad-pair rows are never written by the kernel
            # (early EXIT) yet flow through silu/scatter math with weight 0;
            # uninitialized inf/nan would poison 0*x.
            c13=torch.zeros(slots + 4, 2 * inter, dtype=torch.bfloat16,
                            device=dev),
            act=torch.zeros(slots + 4, inter, dtype=torch.bfloat16, device=dev),
            a2=torch.zeros(slots + 4, inter, dtype=torch.float8_e4m3fn,
                           device=dev),
            as2=torch.zeros(slots + 4, max(inter // 128, 1),
                            dtype=torch.float32, device=dev),
            c2=torch.zeros(slots + 4, hidden, dtype=torch.bfloat16,
                           device=dev),
            desc=torch.empty(4, slots // _BLOCK, 6, dtype=torch.int64,
                             device=dev),
            # split-FP4 (moe_w4q_mm) desc tables: 8 u64 per pair, 64 B ABI
            desc4s=torch.empty(2, slots // _BLOCK, 8, dtype=torch.int64,
                               device=dev),
            # -1 slot row for the tier-less desc path; sized to the MODEL's
            # expert count (256 = DS4 default; 384 Kimi-K2.x reads past a
            # fixed 256-row table).
            no_slots=torch.full((max(n_experts, 256),), -1,
                                dtype=torch.int32, device=dev),
        )
        if _afrag_ok:
            # AFRAG destination buffers: the triton repack streams row-major
            # a1/a2 into these (single pass, no copy-back); the desc tables
            # point the GEMM at them instead of a1/a2.
            _WS.update(
                a1f=torch.zeros(slots + 4, hidden, dtype=torch.float8_e4m3fn,
                                device=dev),
                a2f=torch.zeros(slots + 4, inter, dtype=torch.float8_e4m3fn,
                                device=dev),
            )
    return _WS


import triton
import triton.language as tl


@triton.jit
def _afrag_repack_kernel(src_ptr, dst_ptr, K: tl.constexpr):
    """Row-major fp8 [pairs*16, K] -> AFRAG fragment-major, single pass.

    One program = one (pair, j=k64) 16-row x 64-byte block = 256 u32 words;
    the permutation [pair, g2, g, j, quad, t, b] -> [pair, j, g, t, quad, g2, b]
    lands each program's words in one contiguous 1 KiB dst run. Bit-identical
    to _to_fragment_major (validated), ~3x faster than the torch permute+copy
    and needs no intermediate tensor."""
    p = tl.program_id(0)
    j = tl.program_id(1)
    w = tl.arange(0, 256)
    g2 = w & 1
    quad = (w >> 1) & 3
    t = (w >> 3) & 3
    g = (w >> 5) & 7
    src_off = (p * 16 + g2 * 8 + g) * (K // 4) + j * 16 + quad * 4 + t
    dst_off = p * 16 * (K // 4) + j * 256 + w
    tl.store(dst_ptr + dst_off, tl.load(src_ptr + src_off))


def _afrag_repack(src: torch.Tensor, dst: torch.Tensor, pairs: int, K: int):
    """Repack rows [:pairs*16] of `src` (fp8 row-major) into `dst` (AFRAG)."""
    src32 = src.view(torch.uint8).view(-1).view(torch.int32)
    dst32 = dst.view(torch.uint8).view(-1).view(torch.int32)
    _afrag_repack_kernel[(pairs, K // 64)](src32, dst32, K=K)


@triton.jit
def _desc_build_kernel(
    eids_ptr, npost_ptr, slot_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    p13b, s13b, p2b, s2b, poolb,
    p13s, s13s, p2s, s2s,
    slot_bytes, w13_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """All four moe desc tables in one launch (24 columns per pair).

    d_ptr = [4, cap, 6] i64: 0 = w2-tier w13, 1 = w2-tier w2,
    2 = w4-tier w13, 3 = w4-tier w2. A pair is routed to exactly one tier
    via the m_rows field (the other tier's kernel sees m=0 -> early EXIT).
    slot_ptr = this layer's row of the delta slot table (-1 = base tier);
    poolb = delta pool base (w13 plane at slot start, w2 at +w13_bytes).
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = slot >= 0
    m2 = tl.where(live & ~is4, mblock, 0).to(tl.int64)
    m4 = tl.where(live & is4, mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    slot_c = tl.maximum(slot, 0)
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    bs13 = s13b + e * s13s
    bs2 = s2b + e * s2s
    for gi in tl.static_range(4):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c, m = p13b + e * p13s, bs13, a1, as1, c13, m2
        elif gi == 1:
            b, s, a, as_, c, m = p2b + e * p2s, bs2, a2, as2, c2, m2
        elif gi == 2:
            b, s, a, as_, c, m = (poolb + slot_c * slot_bytes, bs13,
                                  a1, as1, c13, m4)
        else:
            b, s, a, as_, c, m = (poolb + slot_c * slot_bytes + w13_bytes,
                                  bs2, a2, as2, c2, m4)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


@triton.jit
def _desc_build_kernel_w4s(
    eids_ptr, npost_ptr, slot_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    p13b, s13b, p2b, s2b, poolb,
    p13s, s13s, p2s, s2s,
    slot_bytes, w13r_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap8, mblock,
    BLOCK: tl.constexpr,
):
    """Split-FP4 desc tables (moe_w4q_mm, 8 x u64 per pair, 64 B ABI):
    {a, as, base, ref, bs, c, m_rows, pad}. `base`/`bs` point at the
    RESIDENT 2-bit plane / scale rows (exactly the w2 tier's pointers);
    `ref` at the delta slot's quintal sections ([q13 | q2], w13r_bytes =
    q13 section size). Written alongside the main kernel's w2 tables;
    pairs not FP4-resident get m=0 (w4q early-EXITs)."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = slot >= 0
    m4 = tl.where(live & is4, mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    ref = poolb + tl.maximum(slot, 0) * slot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    for gi in tl.static_range(2):
        d = d_ptr + gi * cap8 + p * 8
        if gi == 0:
            bb, rr, ss, a, as_, c = (p13b + e * p13s, ref, s13b + e * s13s,
                                     a1, as1, c13)
        else:
            bb, rr, ss, a, as_, c = (p2b + e * p2s, ref + w13r_bytes,
                                     s2b + e * s2s, a2, as2, c2)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, bb, mask=mask)
        tl.store(d + 3, rr, mask=mask)
        tl.store(d + 4, ss, mask=mask)
        tl.store(d + 5, c, mask=mask)
        tl.store(d + 6, m4, mask=mask)
        tl.store(d + 7, tl.zeros_like(m4), mask=mask)


@triton.jit
def _desc_build_kernel_basecache(
    eids_ptr, npost_ptr, slot_ptr, miss_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    poolb, slot_bytes, off_s13, off_c2, off_s2,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """Base-cache variant of _desc_build_kernel: the 2-bit BASE planes live in
    a GPU pool (slot sections per expert: [codes13 | sc13 | codes2 | sc2]),
    not in resident per-layer planes. A live pair whose expert is NOT resident
    (slot < 0) gets m=0 (the GEMM early-EXITs; its c13/c2 rows stay zero, so
    the pair contributes nothing) and bumps `miss_ptr` — the runner fetches
    the missing experts and replays the step. Only the w2-tier tables d[0]
    (w13 GEMM) and d[1] (w2 GEMM) are written; the w4 tier is not used with
    the base cache."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    hit = slot >= 0
    m = tl.where(live & hit, mblock, 0).to(tl.int64)
    n_miss = tl.sum(tl.where(mask & live & ~hit, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    slot_c = tl.maximum(slot, 0)
    sbase = poolb + slot_c * slot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    for gi in tl.static_range(2):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = sbase, sbase + off_s13, a1, as1, c13
        else:
            b, s, a, as_, c = sbase + off_c2, sbase + off_s2, a2, as2, c2
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


@triton.jit
def _desc_build_kernel_base_delta(
    eids_ptr, npost_ptr, bslot_ptr, fslot_ptr, miss_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    bpoolb, bslot_bytes, off_s13, off_c2, off_s2,
    fpoolb, fslot_bytes, off4_s13, off4_c2, off4_s2,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """Base cache + FP4 need-pool coexistence variant: TWO slot tables with
    priority FP4 > 2-bit base slot > miss. FP4-resident pairs go to the w4
    tier (d[2]/d[3]) reading [fp4_13|sc13|fp4_2|sc2] sections from the FP4
    pool (the slots carry their own scales — no GPU-resident scale planes
    exist with a host-resident base); the rest go to the w2 tier (d[0]/d[1])
    from the base pool. A live pair resident in NEITHER pool gets m=0 in both
    tiers (contributes zero) and bumps `miss_ptr` — same replay contract as
    the plain base-cache kernel. All four desc tables are written."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    bslot = tl.load(bslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    fslot = tl.load(fslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = fslot >= 0
    bhit = bslot >= 0
    m2 = tl.where(live & bhit & ~is4, mblock, 0).to(tl.int64)
    m4 = tl.where(live & is4, mblock, 0).to(tl.int64)
    n_miss = tl.sum(tl.where(mask & live & ~bhit & ~is4, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    bs = bpoolb + tl.maximum(bslot, 0) * bslot_bytes
    fs = fpoolb + tl.maximum(fslot, 0) * fslot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    for gi in tl.static_range(4):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c, m = bs, bs + off_s13, a1, as1, c13, m2
        elif gi == 1:
            b, s, a, as_, c, m = bs + off_c2, bs + off_s2, a2, as2, c2, m2
        elif gi == 2:
            b, s, a, as_, c, m = fs, fs + off4_s13, a1, as1, c13, m4
        else:
            b, s, a, as_, c, m = fs + off4_c2, fs + off4_s2, a2, as2, c2, m4
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


@triton.jit
def _desc_build_kernel_base_delta_split(
    eids_ptr, npost_ptr, bslot_ptr, fslot_ptr, miss_ptr, d_ptr, d4s_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    bpoolb, bslot_bytes, off_s13, off_c2, off_s2,
    fpoolb, fslot_bytes, w13r_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, cap8, mblock,
    BLOCK: tl.constexpr,
):
    """Base cache + SPLIT FP4 need-pool: quintal slots are read AGAINST
    the base pool slot (codes + scales), so a pair routes to the w4q tier
    only when its expert is resident in BOTH slot tables. FP4-mapped but
    base-missing counts as a MISS (contributes zero, bumps miss_ptr — the
    runner's base fetch + replay restores it; the base tier's eviction
    hard-excludes FP4-mapped experts so this is a transient, not a steady
    state). w2 tables (d_ptr[0..1], 6-field) serve base-resident pairs not
    in FP4; w4q tables (d4s_ptr[0..1], 8-field/64 B) carry
    {a, as, base=bslot codes section, ref=fslot section,
    bs=bslot scale section, c, m, pad}."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    bslot = tl.load(bslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    fslot = tl.load(fslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    bhit = bslot >= 0
    is4 = (fslot >= 0) & bhit          # split serve needs BOTH resident
    m2 = tl.where(live & bhit & ~is4, mblock, 0).to(tl.int64)
    m4 = tl.where(live & is4, mblock, 0).to(tl.int64)
    n_miss = tl.sum(tl.where(mask & live & ~bhit, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    bs = bpoolb + tl.maximum(bslot, 0) * bslot_bytes
    fs = fpoolb + tl.maximum(fslot, 0) * fslot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    for gi in tl.static_range(2):      # w2 tables (base pool sections)
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = bs, bs + off_s13, a1, as1, c13
        else:
            b, s, a, as_, c = bs + off_c2, bs + off_s2, a2, as2, c2
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m2, mask=mask)
    for gi in tl.static_range(2):      # w4s tables (base + refinement)
        d = d4s_ptr + gi * cap8 + p * 8
        if gi == 0:
            bb, rr, ss, a, as_, c = bs, fs, bs + off_s13, a1, as1, c13
        else:
            bb, rr, ss, a, as_, c = (bs + off_c2, fs + w13r_bytes,
                                     bs + off_s2, a2, as2, c2)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, bb, mask=mask)
        tl.store(d + 3, rr, mask=mask)
        tl.store(d + 4, ss, mask=mask)
        tl.store(d + 5, c, mask=mask)
        tl.store(d + 6, m4, mask=mask)
        tl.store(d + 7, tl.zeros_like(m4), mask=mask)


def _launch(tier: str, K: int, desc: torch.Tensor, n_rows: int, pairs: int,
            stream):
    fn = _fns[(tier, K)]
    args = [ctypes.c_uint64(desc.data_ptr()),
            ctypes.c_uint32(K),
            ctypes.c_uint32(K // 64),
            ctypes.c_uint32(n_rows * 2),
            ctypes.c_uint32(K // 128)]
    argv = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args])
    _ck(_driver().cuLaunchKernel(fn, n_rows // 16, pairs, 1,
                                 _nwarp_for_k(K) * 32, 1, 1, 0,
                                 stream, argv, None), "launch")


def _moe_w2_forward(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils import prefill_timers
    with prefill_timers.span("moe_w2"):
        return _moe_w2_forward_timed(x, topk_weights, topk_ids, layer_key)


def _moe_w2_forward_timed(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    st = _LAYERS[layer_key]
    T, H = x.shape
    # adaptive expert top-p (env-gated; identity when off). Must run before
    # moe_align/mark_seen/route_log so dropped experts are neither fetched
    # nor counted as routed.
    topk_weights, topk_ids = _apply_topp(topk_weights, topk_ids)
    top_k = topk_ids.shape[1]
    dev = x.device
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    # decode-sized calls use the proven 4-token kernel + delta tier;
    # prefill-sized calls use the MC4 kernel (16 tokens per pair-entry = full
    # QMMA-M, plane reads amortized 4x, ~1.5x over MC2) on the 2-bit base only.
    # 96 = the largest cudagraph capture size: anything above is necessarily a
    # prefill chunk; short tail chunks keep the delta-quality path.
    prefill = T > 96
    mblock = 16 if prefill else _BLOCK
    sorted_ids, expert_blocks, num_post = moe_align_block_size(
        topk_ids, mblock, st["E"])
    slots = sorted_ids.numel()
    pairs = slots // mblock
    # st["K2"] = per-rank expert intermediate I (w2 contraction), st["K13"] =
    # hidden H (w13 contraction) -> size the workspaces for the model's shapes
    # (and correctly under tensor parallelism).
    ws = _workspaces(slots, T, dev, inter=st["K2"], hidden=st["K13"],
                     n_experts=st["E"])

    # ---- activation quant (group-128) into the padded buffer; the buffer's
    # last row is the permanent zero pad row for filler slots.
    xq = ws["xq"]
    pad_row = xq.shape[0] - 1
    _, xs = per_token_group_quant_fp8(x, 128, out_q=xq[:T])
    ws["xs"][:T] = xs
    valid = sorted_ids < T * top_k
    rows = torch.where(valid, sorted_ids // top_k,
                       torch.full_like(sorted_ids, pad_row))
    torch.index_select(xq.view(torch.uint8), 0, rows,
                       out=ws["a1"][:slots].view(torch.uint8))
    torch.index_select(ws["xs"], 0, rows, out=ws["as1"][:slots])

    # ---- desc tables in ONE triton launch
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    base_mode = st.get("base", False)
    # AFRAG (prefill): the GEMM reads fragment-major activations from the
    # dedicated a1f/a2f buffers (filled by the single-pass triton repack
    # below); point the desc 'a' fields there. w4 tables are decode-only,
    # so redirecting the shared base in prefill is safe.
    use_afrag = prefill and _afrag_ok
    a1_base = ws["a1f"] if use_afrag else ws["a1"]
    a2_base = ws["a2f"] if use_afrag else ws["a2"]
    d = ws["desc"]
    cap = d.shape[1]
    miss_rows = None
    if base_mode:
        # BASE cache: 2-bit planes come from the base tier's GPU pool; a live
        # pair with a non-resident expert contributes zero and bumps the miss
        # counter (runner fetches + replays). Prefill fetches its whole layer
        # working set up-front (outside capture) — decode must stay
        # capturable, so misses are handled post-hoc. The FP4 need-pool
        # (delta tier over the base cache, gate-filled) coexists on the
        # decode path: FP4-resident pairs divert to the w4 tier.
        btier = moe_w2_delta._BASE_TIER
        tier = moe_w2_delta._TIER        # FP4 need-pool (None unless opted in)
        if torch.cuda.is_current_stream_capturing():
            btier.notify_capture()
            if tier is not None:
                tier.notify_capture()
        elif prefill:
            btier.ensure_resident(layer_key, topk_ids.view(-1))
        moe_w2_delta.mark_seen(btier.seen[layer_key], topk_ids.view(-1).long())
        if tier is not None:
            # the gate's force_promote reads the FP4 tier's own seen scatter
            moe_w2_delta.mark_seen(tier.seen[layer_key],
                                   topk_ids.view(-1).long())
        if not prefill:
            # LOOKA/PILOT (router-lookahead): score predictors + write the
            # next layer's prediction. Must run BEFORE the route_log
            # overwrite below (predictor [0] reads last step's ids from it).
            # In-graph safe (persistent buffers, static shapes); no-op
            # unless armed.
            from vllm.model_executor.layers.quantization.utils import (
                moe_w2_looka)
            if moe_w2_looka.enabled():
                moe_w2_looka.record(layer_key, x, topk_ids, btier.route_log)
        if not prefill and btier.route_log is not None:
            # per-(token,layer) routing log for the draft-prefetch predictor:
            # a static [n_layers, T_cap, k_cap] buffer the runner reads back
            # post-step (~KBs). In-graph safe: fixed shapes per captured
            # size, static destination. Rows beyond this step's real token
            # count hold stale ids — the host slices by the true T.
            _t = min(topk_ids.shape[0], btier.route_log.shape[1])
            _k = min(topk_ids.shape[1], btier.route_log.shape[2])
            btier.route_log[layer_key, :_t, :_k].copy_(
                topk_ids[:_t, :_k], non_blocking=True)
        if layer_key == 0:
            # per-step counter reset, in-graph (layer 0 runs first each step)
            btier.miss_count.zero_()
        slot_row = btier.slot_table[layer_key]
        use_fp4 = tier is not None and not prefill
        use_w4s_base = use_fp4 and moe_w2_delta.split_enabled()
        if use_w4s_base:
            fslot_row = tier.slot_table[layer_key]
            d4s = ws["desc4s"]
            _desc_build_kernel_base_delta_split[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, fslot_row,
                btier.miss_count, d, d4s,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                tier.pool.data_ptr(), tier.slot_bytes, tier.w13_bytes,
                st["K13"], (st["K13"] // 128) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 128) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, d4s.shape[1] * 8, mblock, BLOCK=256)
        elif use_fp4:
            fslot_row = tier.slot_table[layer_key]
            _desc_build_kernel_base_delta[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, fslot_row,
                btier.miss_count, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                tier.pool.data_ptr(), tier.slot_bytes,
                st["off4_s13"], st["off4_c2"], st["off4_s2"],
                st["K13"], (st["K13"] // 128) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 128) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        else:
            _desc_build_kernel_basecache[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row,
                btier.miss_count, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                st["K13"], (st["K13"] // 128) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 128) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        # Miss pairs get scatter weight 0: the GEMMs early-EXIT on m=0 and
        # never write their c13/c2 rows, but those workspace rows hold STALE
        # values from a previous forward — zeroing the WEIGHT (not the rows)
        # makes the miss contribution an exact 0 for free. Graph-safe (pure
        # tensor ops on captured buffers). FP4-resident pairs are NOT misses
        # — except under split, where serving needs the BASE slot too (an
        # FP4-mapped/base-missing pair contributed zero and must replay).
        e_pair = expert_blocks.to(torch.long).clamp_(0, st["E"] - 1)
        resident = (slot_row[e_pair] >= 0)
        if use_fp4 and not use_w4s_base:
            resident |= (fslot_row[e_pair] >= 0)
        miss_rows = resident.repeat_interleave(mblock)[:slots]
        if not use_fp4:
            tier = None      # downstream w4 launches key off `tier`
    else:
        tier = moe_w2_delta._TIER       # peek only; created by the plane builder
        if tier is not None and not prefill:
            if torch.cuda.is_current_stream_capturing():
                tier.notify_capture()
            slot_row = tier.slot_table[layer_key]
            pool_ptr = tier.pool.data_ptr()
            moe_w2_delta.mark_seen(tier.seen[layer_key],
                                   topk_ids.view(-1).long())
        else:
            if tier is not None:
                moe_w2_delta.mark_seen(tier.seen[layer_key],
                                       topk_ids.view(-1).long())
            slot_row = ws["no_slots"]
            pool_ptr = ws["a1"].data_ptr()      # never dereferenced (m4=0)
        _desc_build_kernel[(triton.cdiv(pairs, 256),)](
            expert_blocks, num_post, slot_row, d,
            a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
            a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
            st["planes13"].data_ptr(), st["sc13"].data_ptr(),
            st["planes2"].data_ptr(), st["sc2"].data_ptr(), pool_ptr,
            st["planes13"].shape[1], st["sc13"].shape[1],
            st["planes2"].shape[1], st["sc2"].shape[1],
            (tier.slot_bytes if tier is not None else moe_w2_delta.SLOT_BYTES),
            (tier.w13_bytes if tier is not None else moe_w2_delta.W13_BYTES),
            # row strides (bytes). H-side: a1 fp8 [H], as1 f32 [H/128], c2 bf16
            # [H]. per-rank intermediate side: c13 bf16 [2I], a2 fp8 [I], as2
            # f32 [I/128]. K13 = H, K2 = I -> identical to the old literals on
            # DS4 TP1 (H=4096, I=2048); GLM-5.x gets H=6144, TP shards shrink I.
            st["K13"], (st["K13"] // 128) * 4, 4 * st["K2"], st["K2"],
            (st["K2"] // 128) * 4, 2 * st["K13"],
            st["E"], pairs, cap * 6, mblock, BLOCK=256)
        if tier is not None and not prefill and moe_w2_delta.split_enabled():
            # split-FP4: the extra 8-field tables for moe_w4q_mm (base/bs =
            # the resident plane rows, ref = the slot's quintal sections)
            d4s = ws["desc4s"]
            _desc_build_kernel_w4s[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, d4s,
                a1_base.data_ptr(), ws["as1"].data_ptr(),
                ws["c13"].data_ptr(), a2_base.data_ptr(),
                ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                st["planes13"].data_ptr(), st["sc13"].data_ptr(),
                st["planes2"].data_ptr(), st["sc2"].data_ptr(), pool_ptr,
                st["planes13"].shape[1], st["sc13"].shape[1],
                st["planes2"].shape[1], st["sc2"].shape[1],
                tier.slot_bytes, tier.w13_bytes,
                st["K13"], (st["K13"] // 128) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 128) * 4, 2 * st["K13"],
                st["E"], pairs, d4s.shape[1] * 8, mblock, BLOCK=256)

    # ---- w13 GEMMs (both tiers) -> fused silu*up -> quant -> w2 GEMMs
    # AFRAG prefill: single-pass triton repack row-major a1/a2 -> fragment-major
    # a1f/a2f (desc built against a1f/a2f above) so the GEMM loads each m16k32
    # A-fragment in one LDG.128. Numerics bit-identical to mc4.
    w2tier = ("w2mc4afrag" if use_afrag else "w2mc4") if prefill else "w2"
    # AFRAG repacks COMPLETE 16-row tiles. `slots` is moe_align's OVER-ALLOCATED
    # row count (sorted_ids.numel() = topk*T + E*15), NOT a multiple of 16; the
    # desc/kernel only ever touch the first `pairs*16` rows (num_post <= pairs*16),
    # so repack exactly that tile-aligned region. Rows [pairs*16:slots] are unused
    # filler (never read). Capacity is fine: pairs*16 <= slots <= a1.shape[0]-4.
    if use_afrag:
        _afrag_repack(ws["a1"], ws["a1f"], pairs, st["K13"])
    _launch(w2tier, st["K13"], d[0], st["N13"], pairs, stream)
    # split-FP4 dispatch: both residency modes fill ws["desc4s"] (classic:
    # _desc_build_kernel_w4s against resident planes; base cache:
    # _desc_build_kernel_base_delta_split against the coupled base slots)
    use_w4s = (tier is not None and not prefill
               and moe_w2_delta.split_enabled())
    if tier is not None and not prefill:
        if use_w4s:
            _launch("w4q", st["K13"], ws["desc4s"][0], st["N13"], pairs,
                    stream)
        else:
            _launch("w4", st["K13"], d[2], st["N13"], pairs, stream)
    act = ws["act"][:slots]
    torch.ops._C.silu_and_mul(act, ws["c13"][:slots])
    _, qs2 = per_token_group_quant_fp8(act, 128, out_q=ws["a2"][:slots])
    ws["as2"][:slots] = qs2
    if use_afrag:
        _afrag_repack(ws["a2"], ws["a2f"], pairs, st["K2"])
    _launch(w2tier, st["K2"], d[1], st["N2"], pairs, stream)
    if tier is not None and not prefill:
        if use_w4s:
            _launch("w4q", st["K2"], ws["desc4s"][1], st["N2"], pairs, stream)
        else:
            _launch("w4", st["K2"], d[3], st["N2"], pairs, stream)

    # ---- weighted unpermute (pad slots masked out), DETERMINISTIC.
    # The old `out.index_add_(0, rows, c2*w)` scattered with atomics, so the
    # f32 accumulation ORDER varied run-to-run: identical inputs wobbled by
    # up to ~1.6e-2 abs on prefill, and single-token probes produced a small
    # set of bit-distinct logit variants — the root cause of the "greedy
    # decode is not reproducible" investigation (PP_DETERMINISM.md; it was
    # never PP-specific). Deterministic scheme: every VALID slot owns a
    # unique (token, j) coordinate (valid sorted_ids are a permutation of
    # token*top_k + j), so index_copy_ into [T*top_k (+1 dump row), H] has no
    # write collisions except filler slots, which all target the discarded
    # dump row. The final sum(dim=1) reduces top_k in a fixed order.
    # Static shapes + no host branches -> cudagraph-capture-safe.
    w = topk_weights.reshape(-1)[sorted_ids.clamp(max=T * top_k - 1)]
    w = torch.where(valid, w, torch.zeros_like(w)).to(torch.float32)
    if miss_rows is not None:
        # base cache: rows of non-resident pairs hold stale workspace values
        # (their GEMMs early-EXITed) — zero their scatter weight so a miss
        # contributes exactly nothing (the replay recomputes them properly).
        w = w * miss_rows.to(torch.float32)
    dump = T * top_k                       # collision row for filler slots
    dst = torch.where(valid, sorted_ids,
                      torch.full_like(sorted_ids, dump)).long()
    gath = torch.zeros(dump + 1, H, dtype=torch.float32, device=dev)
    gath.index_copy_(0, dst, ws["c2"][:slots].float() * w.unsqueeze(1))
    return gath[:dump].view(T, top_k, H).sum(dim=1).to(x.dtype)


def _moe_w2_forward_fake(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    "moe_w2_forward",
    _moe_w2_forward,
    fake_impl=_moe_w2_forward_fake,
)


def moe_w2_forward(x, topk_weights, topk_ids, layer_key):
    return torch.ops.vllm.moe_w2_forward(x, topk_weights, topk_ids, layer_key)


@functools.cache
def ready() -> bool:
    return enabled() and _ensure_ready()
