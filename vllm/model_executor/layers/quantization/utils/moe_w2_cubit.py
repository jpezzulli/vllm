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
            QMMA.SF block-32 sfb, f32 act-scale fold per k32) for BOTH
            w13 and w2.
  glue    : moe_align_block_size(block=4) pairs, a32 activation quant
            (fp8 e4m3 + exact f32 PER-32-GROUP scales; the a128 lineage
            quantized per-128 — the 4x-coarser groups plus the group-128
            mid-pipeline requant were the last measured source of the
            +8-11% completion-token inflation vs native; the e8m0-scale
            MXFP8-parity variant lost accuracy: GSM8K 95.5% vs 97.0%),
            silu*up in torch, weighted scatter-add unpermute. All
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
# Numerics are bit-identical to mc4. Needs moe_w2_mm_mc4afrag_k{K}_a32.cubin present
# (loader degrades to mc4 when missing). Opt out: VLLM_MOE_W2_AFRAG=0.
_AFRAG = os.getenv("VLLM_MOE_W2_AFRAG", "1") == "1"
_afrag_ok = False
# Fused gather-reduce removes the enormous deterministic scatter buffer.
# Keep a runtime A/B escape hatch until every native-grade cell is remeasured.
_FUSED_UNPERMUTE = os.getenv(
    "VLLM_MOE_W2_FUSED_UNPERMUTE", "0") == "1"
# PREFILL QUALITY LEVER (default ON): prefill-sized calls consume the FP4
# delta tier exactly like decode — pairs whose expert is FP4-resident divert
# to moe_w4(q)_mm, the rest stay on the 2-bit planes. Before this, prefill
# ALWAYS computed on bare 2-bit planes, which was measured to be the whole
# source of the +8-11% completion-token inflation vs native (GSM8K-200 DS4
# TP2 tau1.0: 125 tok -> 116-117 = native parity when prefill reads the FP4
# tier; the KV built from 2-bit prefill flattens decode logits cumulatively
# with context length — internal/PREFILL_KV_INFLATION_FINDINGS.md). The
# w4/w4q kernels take M<=4, so each 16-token prefill pair dispatches as 4
# sub-entries; the 2-bit majority keeps the MC4/AFRAG fast path. Prefill
# quality tracks pool coverage (partial pool = partial recovery — graceful).
# Cost: FP4-resident pairs read the slot plane once per sub-entry (4x plane
# traffic on the diverted minority). Opt out: VLLM_MOE_W2_PREFILL_FP4=0.
# BASE-cache prefill is untouched (its need-pool is gate-scoped and tiny).
#
# "ensure" mode (VLLM_MOE_W2_PREFILL_FP4=ensure) upgrades the opportunistic
# lever to the DECODE-CLASS GUARANTEE: before each layer's desc build the
# chunk's routed set is made resident via tier.ensure_resident (the base
# cache's prefill idiom — synchronous fetch + per-layer pin scope), so
# EVERY pair diverts to FP4 regardless of steady-state coverage. The pool
# only needs to hold one layer's chunk working set (<= E slots); parity no
# longer costs a step-union-sized pool. Prices: H2D on cold working sets
# (a broad prompt can stream the whole quintal store once, ~1 s/40 GiB on
# PCIe5), the decode hot set gets displaced by long prompts (tau-driven
# fires re-promote it after the prompt — transient), and the guarantee
# only holds on EAGER prefill: chunks replayed from captured piecewise
# graphs skip host code and stay opportunistic (same boundary the base
# cache accepts).
_PF4_ENV = os.getenv("VLLM_MOE_W2_PREFILL_FP4", "1")
_PREFILL_FP4 = _PF4_ENV not in ("0", "")
_PREFILL_FP4_ENSURE = _PF4_ENV == "ensure"
_pf4_ensure_logged = False
# Decode/prefill routing threshold: calls with T > this take the prefill
# path (MC4/AFRAG kernels + prefill-FP4). The default 96 is the LARGEST
# CUDAGRAPH CAPTURE SIZE of the standing multi-seq configs — a captured
# decode graph must never cross into the prefill path. Low-concurrency
# quality configs can lower it (e.g. 16 at max-num-seqs 1, where the
# biggest decode step is seqs x (1+spec) = 3 tokens) so that SHORT PROMPTS
# (a 90-token chunk has T <= 96) reach the prefill path and the ensure
# guarantee instead of the opportunistic decode path. Keep it ABOVE the
# config's largest captured decode size or captured decode graphs would
# bake prefill-path behaviour.
_PREFILL_T = int(os.getenv("VLLM_MOE_W2_PREFILL_T", "96"))


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

# ---- activation-scale group size per GEMM (the a32 kernel format) --------
# The _a32 kernels read f32 A scales at PER-32 stride; quantizing with a
# coarser group and repeating each scale over the 32-groups it covers is
# mathematically identical to quantizing at that coarser group — so one
# cubin set serves any {32, 64, 128} combination. G1 = the x -> w13 GEMM,
# G2 = the silu·up requant -> w2 GEMM; UE8M0 rounds scales up to powers of
# two (per_token_group_quant_fp8's platform default on this stack, i.e.
# what the retired a128 lineage actually served).
#
# DEFAULTS = 128/128/UE8M0: the measured-best combination. The activation-
# precision handoff's plan A (per-32 groups) was built and E2E-FALSIFIED
# here — GSM8K-200, 2x6000 quintal tau1.0, all with the identical stack:
#   G1=128 G2=128 ue8m0 (a128-equivalent):  97.0%  122 tok  (flips 1<->1)
#   G1=32  G2=128 f32:                      96.5%  124 tok
#   G1=32  G2=32  f32:                      95.5%  127 tok
#   G1=32  G2=32  e8m0 (native MXFP8 fmt):  95.5%  122 tok
#   native / a128 anchors:                  97.0%  116 / 125 tok
# Finer A groups do NOT shorten completions (the +8-11% inflation vs native
# does not come from activation-scale granularity) and consistently cost
# accuracy against the 2-bit/quintal weight planes. The envs stay for
# format experiments; the per-32-capable kernels are the delivery.
_G1 = int(os.getenv("VLLM_MOE_W2_A32_G1", "128"))
_G2 = int(os.getenv("VLLM_MOE_W2_A32_G2", "128"))
_A32_UE8M0 = os.getenv("VLLM_MOE_W2_A32_UE8M0", "1") == "1"
assert _G1 in (32, 64, 128) and _G2 in (32, 64, 128), (_G1, _G2)


def _quant_a32(x, out_q, out_s, group: int):
    """Quantize rows of `x` into the per-32-stride a32 scale plane using
    `group`-sized amax groups. For group > 32 the scale broadcast into the
    strided plane is a single view-copy (no repeat_interleave temporary)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    _, s = per_token_group_quant_fp8(x, group, out_q=out_q,
                                     use_ue8m0=_A32_UE8M0)
    r = group // 32
    if r == 1:
        out_s.copy_(s)
    else:
        out_s.view(s.shape[0], -1, r).copy_(s.unsqueeze(-1))


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


_cutoff_cache: int | None = None


def _layer_cutoff() -> int:
    """Main-stack layer count: layers >= this are the MTP drafter. Taken from
    the model config when available (43 for DS4-Flash, 78 for GLM-5.2, 61 for
    Kimi-K2.7); VLLM_MOE_W2_NUM_LAYERS overrides.

    get_text_config() unwraps composite VLM configs (KimiK25Config keeps
    num_hidden_layers on .text_config; a bare hf_config lookup would raise
    and silently fall back to 43, sending layers 43+ down the stock path);
    for text-only configs it returns self.

    Manual memoization on SUCCESS ONLY (was @functools.cache): a transient
    config-not-current exception must not freeze the guessed 43 forever —
    on a 78-layer GLM that silently sent layers 43+ down the stock path
    (mixed precision, no log — review finding 2.4). The fallback is now
    loud and uncached."""
    global _cutoff_cache
    if _cutoff_cache is not None:
        return _cutoff_cache
    v = os.getenv("VLLM_MOE_W2_NUM_LAYERS")
    if v is not None:
        _cutoff_cache = int(v)
        return _cutoff_cache
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config().model_config.hf_config
        cfg = cfg.get_text_config()
        n = cfg.num_hidden_layers
        if n:
            _cutoff_cache = int(n)
            return _cutoff_cache
    except Exception as e:  # noqa: BLE001
        logger.error(
            "moe_w2 _layer_cutoff: no current vllm config (%s) — TEMPORARY "
            "uncached fallback 43 (DS4 layout); set VLLM_MOE_W2_NUM_LAYERS "
            "explicitly if this repeats past load.", e)
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
        # Hand-written SASS = sm_120 ONLY, and the cubins carry no PTX to
        # JIT elsewhere. Refuse EARLY with a clear message instead of the
        # late shape-assert at weight load (review finding 2.6): SM121
        # (GB10/DGX Spark), SM100 and SM90 cannot run these kernels.
        cap = torch.cuda.get_device_capability()
        if cap != (12, 0):
            raise RuntimeError(
                f"moe_w2 cubins are sm_120-only (hand-written SASS, no "
                f"PTX); this device reports sm_{cap[0]}{cap[1]}. Unset "
                "VLLM_MOE_W2 on this hardware.")
        cu = _driver()
        # ONLY `_a32` cubins load (activation format rev: f32 A scales at
        # PER-32-GROUP granularity, folded per k32). The a128 lineage reads
        # the desc `as` field with a (K/128)*4 row stride — feeding it the
        # a32 (K/32)*4 plane would be silent garbage, so the filename suffix
        # IS the format contract: a cubin dir without the complete _a32 set
        # fails loudly in _require_kernels (no old/new mixing possible).
        # The a128-era mc2 prefill experiment is retired (superseded by
        # mc4/AFRAG; never launched by the serving path).
        # K set: the shipped families cover DS4/GLM/Kimi at TP1-8;
        # VLLM_MOE_W2_KS extends it WITHOUT a code edit when a new model
        # brings a new contraction (comma-separated; cubins must exist).
        ks = tuple(int(x) for x in os.getenv(
            "VLLM_MOE_W2_KS", "7168,6144,4096,2048,1024,512").split(","))
        for tier, kern in (("w2", b"moe_w2_mm"), ("w4", b"moe_w4_mm"),
                           ("w4q", b"moe_w4q_mm"), ("w2mc4", b"moe_w2_mm")):
            # GEMM contraction K: gate-up needs K=hidden (4096 DS4-Flash,
            # 6144 GLM-5.x, 7168 Kimi-K2.x); down needs K=I/TP (2048 @ TP1,
            # 1024 @ TP2, 512 @ TP4). Cubins are loaded opportunistically --
            # the plane builders assert the shapes the model actually needs
            # are present (_require_kernels fails loudly at weight load).
            for k in ks:
                if tier == "w2mc4":
                    fname = f"moe_w2_mm_mc4_k{k}_a32.cubin"
                else:
                    fname = f"moe_{tier}_mm_k{k}_a32.cubin"
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
                for k in ks:
                    path = os.path.join(
                        _DIR, f"moe_w2_mm_mc4afrag_k{k}_a32.cubin")
                    if not os.path.exists(path):
                        continue
                    mod = ctypes.c_void_p()
                    _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                        f"cuModuleLoad {path}")
                    fn = ctypes.c_void_p()
                    _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"moe_w2_mm"),
                        "cuModuleGetFunction afrag")
                    _fns[("w2mc4afrag", k)] = fn
                loaded = sorted(
                    k for tier, k in _fns if tier == "w2mc4afrag")
                _afrag_ok = bool(loaded)
                if loaded:
                    logger.info(
                        "moe_w2_cubit: AFRAG prefill cubins loaded for K=%s",
                        loaded)
                else:
                    logger.warning(
                        "moe_w2_cubit: no AFRAG cubins found; using mc4")
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
    if missing:
        raise RuntimeError(
            f"moe_w2_cubit: missing cubins for K13={K13}/K2={K2}: "
            f"{missing} (dir {_DIR}; set VLLM_MOE_W2_CUBIT_DIR)")


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


def _layer_contract(layer) -> dict:
    """Validate semantics the hand-written W2 path actually implements."""
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        if cfg.model_config.dtype != torch.bfloat16:
            raise ValueError(
                "VLLM_MOE_W2 kernels use BF16 intermediates and currently "
                f"require model dtype bfloat16, got {cfg.model_config.dtype}")
        if cfg.use_v2_model_runner:
            raise ValueError(
                "VLLM_MOE_W2 is not integrated with Model Runner V2 "
                "(mandatory base replay and gate hooks are absent)")
        pc = cfg.parallel_config
        if (getattr(pc, "use_ubatching", False)
                or getattr(pc, "ubatch_size", 0)):
            raise ValueError(
                "VLLM_MOE_W2 does not support ubatching/DBO: its CUDA "
                "workspaces are shared across forwards")
        if (getattr(pc, "enable_expert_parallel", False)
                or getattr(pc, "enable_eplb", False)):
            raise ValueError(
                "VLLM_MOE_W2 does not support expert parallelism or EPLB")
    except ValueError:
        raise
    except Exception:
        # Layer-level checks below remain authoritative in offline tools that
        # intentionally construct layers without a current VllmConfig.
        pass
    activation = getattr(layer, "activation", "silu")
    activation = getattr(activation, "value", activation)
    if activation != "silu":
        raise ValueError(
            "VLLM_MOE_W2 supports only packed SILU-gated routed experts; "
            f"layer {getattr(layer, 'layer_name', '')!r} uses "
            f"{activation!r}")
    alpha = getattr(layer, "swiglu_alpha", None)
    beta = getattr(layer, "swiglu_beta", None)
    if alpha not in (None, 1.0) or beta not in (None, 0.0):
        raise ValueError(
            "VLLM_MOE_W2 does not implement non-default SwiGLU alpha/beta "
            f"(got alpha={alpha}, beta={beta})")
    moe_config = getattr(layer, "moe_config", None)
    if bool(getattr(moe_config, "has_bias", False)):
        raise ValueError(
            "VLLM_MOE_W2 does not implement routed-expert w13/w2 bias")
    if getattr(layer, "expert_map", None) is not None:
        raise ValueError(
            "VLLM_MOE_W2 does not support expert parallel/EPLB mappings")
    if bool(getattr(layer, "apply_router_weight_on_input", False)):
        raise ValueError(
            "VLLM_MOE_W2 does not support router weight on input")
    limit = getattr(layer, "swiglu_limit", None)
    if limit is not None:
        limit = float(limit)
        if not limit > 0:
            raise ValueError(
                f"VLLM_MOE_W2 swiglu_limit must be positive, got {limit}")
    return {
        "activation": activation,
        "swiglu_limit": limit,
        "swiglu_alpha": 1.0,
        "swiglu_beta": 0.0,
    }


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
    _layer_contract(layer)
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
    _layer_contract(layer)
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
    contract = _layer_contract(layer)
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
        **contract,
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
    _layer_contract(layer)
    if not _ensure_ready():
        raise RuntimeError("moe_w2 cubins missing or failed to load")
    dev = torch.device("cuda")
    if getattr(layer, "_moe_w2_pack_skip", False):
        E, N13, K13, N2, K2 = layer._moe_w2_shapes
        from vllm.model_executor.layers.quantization.utils import moe_w2_delta
        _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())
        if moe_w2_delta.base_enabled():
            if not _try_skip_requant(
                    layer, layer_key, E, N13, K13, N2, K2,
                    ("w13_weight", "w13_weight_scale", "w2_weight",
                     "w2_weight_scale")):
                raise RuntimeError(
                    f"moe_w2 mxfp4 layer {layer_key} pack generation "
                    "changed during loader skip")
        elif not _consume_planes_cache(
                layer, layer_key, dev, E, N13, K13, N2, K2):
            raise RuntimeError(
                f"moe_w2 mxfp4 layer {layer_key} planes cache changed "
                "during loader skip")
        return
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
    if _consume_planes_cache(
            layer, layer_key, dev, E, N13, K13, N2, K2):
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

    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as planes_cache)
    lidx = planes_cache.layer_idx_from_name(
        getattr(layer, "layer_name", ""))
    if planes_cache.enabled() and lidx is not None:
        planes_cache.store(
            lidx,
            dict(planes13=planes13, sc13=sc13, planes2=planes2, sc2=sc2,
                 fp13=fp13, fp2=fp2))

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

    _layer_contract(layer)
    if not _ensure_ready():
        raise RuntimeError("moe_w2 cubins missing or failed to load")
    dev = torch.device("cuda")
    w13 = layer.w13_weight.data                       # [E, 2I, H] e4m3 (cpu)
    s13 = getattr(layer, f"w13_{scale_suffix}").data  # [E, 2I/128, H/128] f32
    w2 = layer.w2_weight.data                         # [E, H, I] e4m3
    s2 = getattr(layer, f"w2_{scale_suffix}").data    # [E, H/128, I/128] f32
    block_shape = getattr(layer, "weight_block_size", None) or (128, 128)
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
                wg[i], sg[i], block_shape=block_shape,
                want_nibbles=fp13 is not None)
            planes13[e0 + i] = pack_fragment_major(codes)
            sc13[e0 + i] = pack_scales(sbytes)
            if fp13 is not None:
                fp13[e0 + i] = _pack_fp4_plane(nib)
        wg = w2[e0:e1].to(dev, non_blocking=True)
        sg = s2[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            codes, sbytes, nib = fp8_block_to_codes_scales(
                wg[i], sg[i], block_shape=block_shape,
                want_nibbles=fp2 is not None)
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

    _layer_contract(layer)
    if not _ensure_ready():
        raise RuntimeError("moe_w2 cubins missing or failed to load")
    dev = torch.device("cuda")
    if getattr(layer, "_moe_w2_pack_skip", False):
        # loader-level skip: the params are 0-byte stubs (plan_pack_skip),
        # shapes travel via the create-time stash. The probed store MUST
        # still serve the layer — there is no checkpoint copy to fall
        # back to.
        E, N13, K13, N2, K2 = layer._moe_w2_shapes
        _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())
        if moe_w2_delta.base_enabled():
            if not _try_skip_requant(
                    layer, layer_key, E, N13, K13, N2, K2,
                    ("w13_weight", "w13_weight_scale", "w2_weight",
                     "w2_weight_scale")):
                raise RuntimeError(
                    f"moe_w2: layer {layer_key} was loader-skipped on a "
                    "validated pack hit but the pack no longer serves it "
                    "(generation changed mid-load); restart after checking "
                    "VLLM_MOE_W2_STORE_DIR")
            return
        # GPU-resident: materialize the planes from the planes cache
        # (probed at create time; a miss here means the cache dir changed
        # under a live load).
        if not _consume_planes_cache(
                layer, layer_key, dev, E, N13, K13, N2, K2):
            raise RuntimeError(
                f"moe_w2: layer {layer_key} was loader-skipped on a "
                "planes-cache hit but the cache generation changed "
                "mid-load; restart after checking "
                "VLLM_MOE_W2_PLANES_CACHE")
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
    contract = _layer_contract(layer)
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
        # Stage sections directly. Concatenating these multi-GiB tensors on
        # GPU created a full duplicate layer transient during first boot.
        btier.add_layer_host_sections(
            layer_key, (planes13, sc13), (planes2, sc2))
        _LAYERS[layer_key] = dict(
            N13=N13, K13=K13, N2=N2, K2=K2, E=E, base=True, tl_idx=_tl,
            off_s13=c13len, off_c2=c13len + s13len,
            off_s2=c13len + s13len + c2len,
            # FP4 need-pool slot sections ([fp4_13|sc13|fp4_2|sc2]; fp4 codes
            # are 2x the 2-bit codes, scale sections identical) — read by the
            # base+delta desc kernel when the FP4 tier coexists.
            off4_s13=2 * c13len, off4_c2=2 * c13len + s13len,
            off4_s2=2 * c13len + s13len + 2 * c2len,
            **contract,
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

    _check_resident_fit(layer_key, dev,
                        planes13.nbytes + sc13.nbytes
                        + planes2.nbytes + sc2.nbytes)
    _LAYERS[layer_key] = dict(
        planes13=planes13, sc13=sc13, planes2=planes2, sc2=sc2,
        N13=N13, K13=K13, N2=N2, K2=K2, E=E, tl_idx=_tl,
        **contract,
    )
    # Release checkpoint copies; keep CUDA stubs so device probes stay happy.
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in param_names:
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info("moe_w2: layer %d planes built (%.2f GiB)", layer_key,
                (planes13.nbytes + sc13.nbytes + planes2.nbytes + sc2.nbytes)
                / 2**30)


_resident_fit_checked = False


def _check_resident_fit(layer_key: int, dev, bytes_per_layer: int) -> None:
    """RESIDENCY HARDSTOP (user directive 2026-07-13): if resident planes
    cannot fit in VRAM without degrading serving (no room for KV/graphs),
    refuse EARLY with actionable guidance instead of a raw CUDA OOM deep
    into the requant. Mirrors the base-cache working-set hardstop one rung
    down the residency ladder.

    Runs once, at the FIRST resident layer (its exact per-layer bytes x
    the model's remaining MoE layer count = total plane demand; uniform
    layers assumed - true for DS4/GLM/Kimi). Reserves are estimates and
    deliberately conservative in the SAFE direction only (a false PASS
    just falls through to the torch OOM backstop below; a false FAIL is
    avoided by keeping reserves minimal: 2 GiB graphs/workspaces + ~1.7
    GiB MTP drafter scratch when speculative decoding is on + a KV floor
    of one max_model_len sequence at the measured fp8 MLA rate)."""
    global _resident_fit_checked
    if _resident_fit_checked:
        return
    _resident_fit_checked = True
    try:
        from vllm.config import get_current_vllm_config
        vcfg = get_current_vllm_config()
        hf = vcfg.model_config.hf_text_config
        n_layers = int(getattr(hf, "num_hidden_layers", 0) or 0)
        n_dense = int(getattr(hf, "first_k_dense_replace", 0) or 0)
        n_moe = max(n_layers - n_dense, 1)
        max_len = int(vcfg.model_config.max_model_len)
        spec_on = vcfg.speculative_config is not None
        free_b, _total_b = torch.cuda.mem_get_info(dev)
        planes_total = bytes_per_layer * (n_moe - len(_LAYERS))
        kv_floor = max_len * 16 * 1024            # ~15.6 KB/tok measured
        reserve = 2 * 2**30 + (int(1.7 * 2**30) if spec_on else 0) + kv_floor
        budget = free_b - reserve
        if planes_total > budget and os.getenv(
                "VLLM_MOE_W2_FORCE_RESIDENT", "0") == "1":
            # Explicit consent valve (the FORCE_POOL pattern): the reserve
            # model is deliberately conservative and refuses knife-edge 1x
            # configs that DO serve (the validated pro6000x1 recipe sits
            # ~3-4 GiB inside this guard's reserves). Forced boots fall
            # through to the torch OOM backstop if the guard was right.
            logger.warning(
                "moe_w2 RESIDENT planes exceed the estimated budget by "
                "%.1f GiB but VLLM_MOE_W2_FORCE_RESIDENT=1 - continuing "
                "on user consent (a real shortfall will OOM at load or "
                "capture).", (planes_total - budget) / 2**30)
            return
        if planes_total > budget:
            deficit = (planes_total - budget) / 2**30
            # suggested pool: what actually fits, rounded down to .5 GiB
            fit_gb = max((budget + bytes_per_layer * len(_LAYERS))
                         / 2**30, 0.0)
            sug = max(int(fit_gb * 2) / 2, 1.0)
            raise ValueError(
                f"moe_w2 RESIDENT planes do not fit: {n_moe} MoE layers x "
                f"{bytes_per_layer / 2**30:.2f} GiB = "
                f"{n_moe * bytes_per_layer / 2**30:.1f} GiB of 2-bit "
                f"planes vs ~{budget / 2**30:.1f} GiB of VRAM budget "
                f"(free {free_b / 2**30:.1f} minus KV floor for "
                f"{max_len} tokens, graphs/workspaces"
                f"{' and MTP scratch' if spec_on else ''}) - short by "
                f"~{deficit:.1f} GiB. Serving would OOM or degrade. Use "
                f"the BASE-CACHE rung of the residency ladder instead: "
                f"VLLM_MOE_W2_BASE_CACHE_GB={sug:.1f} (host-resident "
                f"planes, GPU expert cache; add VLLM_MOE_W2_STORE_DIR + "
                f"VLLM_MOE_W2_BASE_RAM_GB for the NVMe tier if host RAM "
                f"is short). The boot guard will then verify the pool "
                f"against its working-set floor.")
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 - the check must not break boot
        logger.warning("moe_w2 resident-fit check skipped: %s", e)


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------

def _workspaces(slots: int, tokens: int, dev, inter: int = 2048,
                hidden: int = 4096, n_experts: int = 256) -> dict:
    # `inter` = per-rank expert intermediate size I (2048 on 1 GPU; 1024 @ TP2,
    # 512 @ TP4 as the experts shard). The hidden H (4096 DS4, 6144 GLM-5.x) is
    # NOT sharded, so the A-side (a1), x-quant (xq) and w2 output (c2) buffers
    # stay H-wide; only the gate/up output (c13 = 2I), the intermediate
    # activation (act/a2 = I) and its per-32 scales (as2 = I/32) follow the
    # shard.
    #
    # Activation scales are per-32-GROUP f32 (a32 cubin rev): [rows, K/32]
    # f32 — 4x the scale elements of the retired per-128 format (row stride
    # (K/32)*4 bytes; the desc-build strides moved with it).
    if (_WS.get("slots", 0) < slots or _WS.get("tokens", 0) < tokens
            or _WS.get("inter") != inter or _WS.get("hidden") != hidden
            or _WS.get("n_experts", 0) < n_experts):
        # Captured graphs bake these buffer ADDRESSES; a realloc after any
        # capture would leave every captured replay reading freed memory.
        # The system invariant is "profile_run maxes the sizes before
        # capture" — enforce it instead of assuming it (review 2.5): any
        # growth after the first capture (or during one) is a hard error.
        if _WS.get("frozen"):
            raise RuntimeError(
                f"moe_w2 workspace growth after graph capture: have "
                f"slots={_WS.get('slots')}, tokens={_WS.get('tokens')}, "
                f"need slots={slots}, tokens={tokens}. Captured graphs "
                "hold the old buffers - the profile run must cover the "
                "largest schedulable batch (check max_num_batched_tokens/"
                "max_num_seqs vs the profile shapes).")
        if torch.cuda.is_available() and torch.cuda.is_initialized() \
                and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "moe_w2 workspace realloc DURING cudagraph capture "
                f"(slots {_WS.get('slots', 0)}->{slots}, tokens "
                f"{_WS.get('tokens', 0)}->{tokens}) - the profile run "
                "must pre-size the workspaces.")
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
            xs=torch.zeros(tokens + 1, hidden // 32, dtype=torch.float32,
                           device=dev),
            a1=torch.zeros(slots + 4, hidden, dtype=torch.float8_e4m3fn,
                           device=dev),
            as1=torch.zeros(slots + 4, hidden // 32, dtype=torch.float32,
                            device=dev),
            # zeros, not empty: pad-pair rows are never written by the kernel
            # (early EXIT) yet flow through silu/scatter math with weight 0;
            # uninitialized inf/nan would poison 0*x.
            c13=torch.zeros(slots + 4, 2 * inter, dtype=torch.bfloat16,
                            device=dev),
            act=torch.zeros(slots + 4, inter, dtype=torch.bfloat16, device=dev),
            a2=torch.zeros(slots + 4, inter, dtype=torch.float8_e4m3fn,
                           device=dev),
            as2=torch.zeros(slots + 4, inter // 32,
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
            # inverse permutation for the fused deterministic unpermute.
            # `slots` >= tokens*top_k, so one buffer covers every valid route.
            inv_sorted=torch.empty(slots + 4, dtype=torch.int32, device=dev),
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
def _invert_sorted_ids_kernel(sorted_ids_ptr, inverse_ptr, n_slots,
                              n_routes, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(sorted_ids_ptr + offsets, mask=offsets < n_slots, other=-1)
    valid = (offsets < n_slots) & (ids >= 0) & (ids < n_routes)
    # Valid sorted ids are a permutation of token*top_k+j, so stores never
    # collide. Padding ids are ignored rather than redirected to a dump row.
    tl.store(inverse_ptr + ids, offsets, mask=valid)


@triton.jit
def _fp32_mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32_add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _deterministic_unpermute_kernel(
    c2_ptr,
    weights_ptr,
    inverse_ptr,
    row_mask_ptr,
    out_ptr,
    hidden,
    c2_stride,
    weight_stride_t,
    weight_stride_k,
    out_stride,
    TOP_K: tl.constexpr,
    HAS_ROW_MASK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h < hidden
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    # Fixed j-order preserves deterministic accumulation without the
    # [T*top_k,H] FP32 scatter buffer used by index_copy_ + sum.
    for j in tl.static_range(0, TOP_K):
        route = token * TOP_K + j
        slot = tl.load(inverse_ptr + route)
        value = tl.load(
            c2_ptr + slot * c2_stride + h, mask=h_mask, other=0.0
        ).to(tl.float32)
        weight = tl.load(
            weights_ptr + token * weight_stride_t + j * weight_stride_k
        ).to(tl.float32)
        if HAS_ROW_MASK:
            weight = _fp32_mul_rn(
                weight, tl.load(row_mask_ptr + slot).to(tl.float32))
        product = _fp32_mul_rn(value, weight)
        acc = _fp32_add_rn(acc, product)
    tl.store(out_ptr + token * out_stride + h, acc, mask=h_mask)


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


def a32_dequant_ref(x: torch.Tensor, gemm: int = 1) -> torch.Tensor:
    """Torch-only reference roundtrip of the activation quant the forward
    serves for the given GEMM (group _G1/_G2, optional UE8M0 rounding —
    mirrors per_token_group_quant_fp8): rows [M, K] -> f32 dequant. Used
    by the tools/ test references so they cannot drift from the serving
    format regardless of the VLLM_MOE_W2_A32_* envs in effect."""
    group = _G1 if gemm == 1 else _G2
    m, k = x.shape
    xb = x.float().view(m, k // group, group)
    scale = (xb.abs().amax(-1).clamp_min(1e-10) / 448.0)
    if _A32_UE8M0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (xb / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.float().view(m, k) * scale.repeat_interleave(group, 1)


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
def _desc_build_kernel_prefill4(
    eids_ptr, npost_ptr, slot_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    a1b_rm, a2b_rm,
    p13b, s13b, p2b, s2b, poolb,
    p13s, s13s, p2s, s2s,
    slot_bytes, w13_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """Prefill variant of _desc_build_kernel (VLLM_MOE_W2_PREFILL_FP4):
    FP4-resident pairs divert to the w4 tier like decode, but the w4/w4q
    kernels take M<=4, so each diverted mblock(16)-token pair is emitted as
    FOUR w4 sub-entries (rows +0/+4/+8/+12, m=4 each) at desc index
    p*4+sub — capacity fits because prefill pairs = slots/16 and the desc
    tables are sized slots/4. The w2 tables keep pair granularity (m=0 for
    diverted pairs -> MC4/AFRAG early-EXIT). w2 A-pointers follow the
    (possibly fragment-major) a1b/a2b bases; the w4 sub-entries always read
    the ROW-MAJOR a1b_rm/a2b_rm (repack leaves them intact)."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = slot >= 0
    m2 = tl.where(live & ~is4, mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    slot_c = tl.maximum(slot, 0)
    bs13 = s13b + e * s13s
    bs2 = s2b + e * s2s
    # w2 tables (pair granularity, diverted pairs zeroed)
    for gi in tl.static_range(2):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = (p13b + e * p13s, bs13, a1b + base * a1_rb,
                               as1b + base * as1_rb, c13b + base * c13_rb)
        else:
            b, s, a, as_, c = (p2b + e * p2s, bs2, a2b + base * a2_rb,
                               as2b + base * as2_rb, c2b + base * c2_rb)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m2, mask=mask)
    # w4 tables (sub-entry granularity: 4 x m<=4 rows per diverted pair)
    for sub in tl.static_range(4):
        rbase = base + sub * 4
        m4 = tl.where(live & is4, 4, 0).to(tl.int64)
        for gi in tl.static_range(2):
            d = d_ptr + (2 + gi) * cap6 + (p * 4 + sub) * 6
            if gi == 0:
                b, s, a, as_, c = (poolb + slot_c * slot_bytes, bs13,
                                   a1b_rm + rbase * a1_rb,
                                   as1b + rbase * as1_rb,
                                   c13b + rbase * c13_rb)
            else:
                b, s, a, as_, c = (poolb + slot_c * slot_bytes + w13_bytes,
                                   bs2, a2b_rm + rbase * a2_rb,
                                   as2b + rbase * as2_rb,
                                   c2b + rbase * c2_rb)
            tl.store(d + 0, a, mask=mask)
            tl.store(d + 1, as_, mask=mask)
            tl.store(d + 2, b, mask=mask)
            tl.store(d + 3, s, mask=mask)
            tl.store(d + 4, c, mask=mask)
            tl.store(d + 5, m4, mask=mask)


@triton.jit
def _desc_build_kernel_w4s_prefill4(
    eids_ptr, npost_ptr, slot_ptr, d_ptr,
    a1b_rm, as1b, c13b, a2b_rm, as2b, c2b,
    p13b, s13b, p2b, s2b, poolb,
    p13s, s13s, p2s, s2s,
    slot_bytes, w13r_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap8, mblock,
    BLOCK: tl.constexpr,
):
    """Prefill variant of _desc_build_kernel_w4s: split-FP4 sub-entries
    (moe_w4q_mm, M<=4) at desc index p*4+sub for FP4-resident pairs. base/
    bs stay per-expert resident-plane pointers; a/as/c take the sub-block
    row offset (ROW-MAJOR activation bases — w4q never reads AFRAG)."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = slot >= 0
    base = p.to(tl.int64) * mblock
    ref = poolb + tl.maximum(slot, 0) * slot_bytes
    for sub in tl.static_range(4):
        rbase = base + sub * 4
        m4 = tl.where(live & is4, 4, 0).to(tl.int64)
        for gi in tl.static_range(2):
            d = d_ptr + gi * cap8 + (p * 4 + sub) * 8
            if gi == 0:
                bb, rr, ss, a, as_, c = (
                    p13b + e * p13s, ref, s13b + e * s13s,
                    a1b_rm + rbase * a1_rb, as1b + rbase * as1_rb,
                    c13b + rbase * c13_rb)
            else:
                bb, rr, ss, a, as_, c = (
                    p2b + e * p2s, ref + w13r_bytes, s2b + e * s2s,
                    a2b_rm + rbase * a2_rb, as2b + rbase * as2_rb,
                    c2b + rbase * c2_rb)
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


@triton.jit
def _desc_build_kernel_base_delta_prefill4(
    eids_ptr, npost_ptr, bslot_ptr, fslot_ptr, miss_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    a1b_rm, a2b_rm,
    bpoolb, bslot_bytes, off_s13, off_c2, off_s2,
    fpoolb, fslot_bytes, off4_s13, off4_c2, off4_s2,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """PREFILL variant of _desc_build_kernel_base_delta (prefill-FP4 over
    the base cache): FP4-resident pairs divert to the w4 tier exactly like
    decode, but as FOUR M<=4 sub-entries per 16-token pair (d[2]/d[3] at
    index p*4+sub — the same capacity argument as the resident prefill4
    builder). w2 pair entries read the BASE pool; w4 sub-entries read the
    FP4 pool sections and the ROW-MAJOR activation bases (AFRAG repack
    leaves them valid). A live pair resident in NEITHER pool contributes
    zero and bumps miss_ptr (with ensure_resident on both tiers this is
    the pool-too-small warning path, not a steady state)."""
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
    n_miss = tl.sum(tl.where(mask & live & ~bhit & ~is4, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    bs = bpoolb + tl.maximum(bslot, 0) * bslot_bytes
    fs = fpoolb + tl.maximum(fslot, 0) * fslot_bytes
    # w2 tables (pair granularity, base pool)
    for gi in tl.static_range(2):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = (bs, bs + off_s13, a1b + base * a1_rb,
                               as1b + base * as1_rb, c13b + base * c13_rb)
        else:
            b, s, a, as_, c = (bs + off_c2, bs + off_s2, a2b + base * a2_rb,
                               as2b + base * as2_rb, c2b + base * c2_rb)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m2, mask=mask)
    # w4 tables (sub-entries, FP4 pool sections)
    for sub in tl.static_range(4):
        rbase = base + sub * 4
        m4 = tl.where(live & is4, 4, 0).to(tl.int64)
        for gi in tl.static_range(2):
            d = d_ptr + (2 + gi) * cap6 + (p * 4 + sub) * 6
            if gi == 0:
                b, s, a, as_, c = (fs, fs + off4_s13,
                                   a1b_rm + rbase * a1_rb,
                                   as1b + rbase * as1_rb,
                                   c13b + rbase * c13_rb)
            else:
                b, s, a, as_, c = (fs + off4_c2, fs + off4_s2,
                                   a2b_rm + rbase * a2_rb,
                                   as2b + rbase * as2_rb,
                                   c2b + rbase * c2_rb)
            tl.store(d + 0, a, mask=mask)
            tl.store(d + 1, as_, mask=mask)
            tl.store(d + 2, b, mask=mask)
            tl.store(d + 3, s, mask=mask)
            tl.store(d + 4, c, mask=mask)
            tl.store(d + 5, m4, mask=mask)


@triton.jit
def _desc_build_kernel_base_delta_split_prefill4(
    eids_ptr, npost_ptr, bslot_ptr, fslot_ptr, miss_ptr, d_ptr, d4s_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    a1b_rm, a2b_rm,
    bpoolb, bslot_bytes, off_s13, off_c2, off_s2,
    fpoolb, fslot_bytes, w13r_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, cap8, mblock,
    BLOCK: tl.constexpr,
):
    """PREFILL variant of _desc_build_kernel_base_delta_split: quintal
    refinement read AGAINST the base slot, so a pair diverts to w4q only
    when resident in BOTH tables (same coupling as decode); diverted pairs
    are emitted as FOUR M<=4 w4q sub-entries (d4s at p*4+sub) reading the
    ROW-MAJOR activation bases. FP4-mapped/base-missing pairs count as
    misses exactly like decode (the base tier's eviction hard-excludes
    FP4-mapped experts, and prefill ensure_resident fetches the base first,
    so this is transient)."""
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
    n_miss = tl.sum(tl.where(mask & live & ~bhit, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    bs = bpoolb + tl.maximum(bslot, 0) * bslot_bytes
    fs = fpoolb + tl.maximum(fslot, 0) * fslot_bytes
    for gi in tl.static_range(2):      # w2 tables (base pool sections)
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = (bs, bs + off_s13, a1b + base * a1_rb,
                               as1b + base * as1_rb, c13b + base * c13_rb)
        else:
            b, s, a, as_, c = (bs + off_c2, bs + off_s2, a2b + base * a2_rb,
                               as2b + base * as2_rb, c2b + base * c2_rb)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m2, mask=mask)
    for sub in tl.static_range(4):     # w4s sub-entries (base + refinement)
        rbase = base + sub * 4
        m4 = tl.where(live & is4, 4, 0).to(tl.int64)
        for gi in tl.static_range(2):
            d = d4s_ptr + gi * cap8 + (p * 4 + sub) * 8
            if gi == 0:
                bb, rr, ss, a, as_, c = (
                    bs, fs, bs + off_s13,
                    a1b_rm + rbase * a1_rb, as1b + rbase * as1_rb,
                    c13b + rbase * c13_rb)
            else:
                bb, rr, ss, a, as_, c = (
                    bs + off_c2, fs + w13r_bytes, bs + off_s2,
                    a2b_rm + rbase * a2_rb, as2b + rbase * as2_rb,
                    c2b + rbase * c2_rb)
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
    global _pf4_ensure_logged

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
    # QMMA-M, plane reads amortized 4x, ~1.5x over MC2) plus the prefill-FP4
    # tier dispatch. Threshold default 96 = the largest cudagraph capture
    # size of the standing configs: anything above is necessarily a prefill
    # chunk; short tail chunks keep the decode path. VLLM_MOE_W2_PREFILL_T
    # lowers it on low-concurrency configs so short prompts reach the
    # prefill path (and the ensure guarantee) — see _PREFILL_T.
    prefill = T > _PREFILL_T
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

    # ---- activation quant (a32: exact f32 scales, group _G1, default 32)
    # into the padded buffer; the buffer's last row is the permanent zero
    # pad row for filler slots.
    xq = ws["xq"]
    pad_row = xq.shape[0] - 1
    _quant_a32(x, xq[:T], ws["xs"][:T], _G1)
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
    # below); point the w2 desc 'a' fields there. The w4 tier never reads
    # AFRAG: decode is row-major anyway and the prefill-FP4 sub-entries
    # take the row-major bases explicitly (the repack copies OUT of a1/a2,
    # leaving them valid).
    use_afrag = (
        prefill
        and _afrag_ok
        and ("w2mc4afrag", st["K13"]) in _fns
        and ("w2mc4afrag", st["K2"]) in _fns
    )
    a1_base = ws["a1f"] if use_afrag else ws["a1"]
    a2_base = ws["a2f"] if use_afrag else ws["a2"]
    d = ws["desc"]
    cap = d.shape[1]
    miss_rows = None
    use_pf4 = False        # prefill-FP4 (resident AND base modes; _PREFILL_FP4)
    if torch.cuda.is_current_stream_capturing():
        # first capture freezes the workspace sizes (see _workspaces)
        _WS["frozen"] = True
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
        use_pf4 = (_PREFILL_FP4 and prefill and tier is not None
                   and tier.n_slots > 0)
        if torch.cuda.is_current_stream_capturing():
            btier.notify_capture()
            if tier is not None:
                tier.notify_capture()
        elif prefill:
            btier.ensure_resident(layer_key, topk_ids.view(-1))
            if use_pf4 and _PREFILL_FP4_ENSURE:
                # decode-class guarantee for the FP4 side too: fetch the
                # chunk's routed set into the need-pool AFTER the base rows
                # (split w4q reads the refinement AGAINST the base slot, and
                # the base eviction hard-excludes FP4-mapped experts, so the
                # base ensure must land first). Layer pins rotate per tier.
                if not _pf4_ensure_logged:
                    _pf4_ensure_logged = True
                    logger.info("moe_w2 prefill-FP4 ensure mode (base "
                                "cache): eager chunk working sets fetched "
                                "to both tiers (first call: layer %d, T=%d)",
                                layer_key, T)
                tier.ensure_resident(layer_key, topk_ids.view(-1))
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
        if use_pf4 and moe_w2_delta.split_enabled():
            # prefill-FP4 over the base cache, SPLIT: w2 pair entries from
            # the base pool + w4q SUB-entries (M<=4) coupling base codes
            # with the quintal refinement. See the prefill4 builders.
            fslot_row = tier.slot_table[layer_key]
            d4s = ws["desc4s"]
            _desc_build_kernel_base_delta_split_prefill4[
                    (triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, fslot_row,
                btier.miss_count, d, d4s,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                ws["a1"].data_ptr(), ws["a2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                tier.pool.data_ptr(), tier.slot_bytes, tier.w13_bytes,
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, d4s.shape[1] * 8, mblock, BLOCK=256)
        elif use_pf4:
            # prefill-FP4 over the base cache, NON-split: w4 sub-entries
            # read full-FP4 sections from the need-pool slots.
            fslot_row = tier.slot_table[layer_key]
            _desc_build_kernel_base_delta_prefill4[
                    (triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, fslot_row,
                btier.miss_count, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                ws["a1"].data_ptr(), ws["a2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                tier.pool.data_ptr(), tier.slot_bytes,
                st["off4_s13"], st["off4_c2"], st["off4_s2"],
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        elif use_w4s_base:
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
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
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
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        else:
            _desc_build_kernel_basecache[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row,
                btier.miss_count, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
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
        if (use_fp4 or use_pf4) and not moe_w2_delta.split_enabled():
            # non-split: a full-FP4 slot serves the pair without its base
            # row (decode d[2]/d[3] or the prefill4 sub-entries).
            resident |= (fslot_row[e_pair] >= 0)
        miss_rows = resident.repeat_interleave(mblock)[:slots]
        if not use_fp4 and not use_pf4:
            tier = None      # downstream w4 launches key off `tier`
    else:
        tier = moe_w2_delta._TIER       # peek only; created by the plane builder
        # prefill consumes the FP4 tier too (quality lever, see _PREFILL_FP4)
        use_pf4 = (_PREFILL_FP4 and prefill and tier is not None
                   and tier.n_slots > 0)
        if tier is not None and not prefill:
            if torch.cuda.is_current_stream_capturing():
                tier.notify_capture()
            slot_row = tier.slot_table[layer_key]
            pool_ptr = tier.pool.data_ptr()
            moe_w2_delta.mark_seen(tier.seen[layer_key],
                                   topk_ids.view(-1).long())
        else:
            if tier is not None:
                # seen marks land BEFORE the desc build: the manager's
                # victim selection excludes this window's experts, so a
                # background pass cannot rewrite a slot between the desc
                # build below and the GEMMs reading it (same protection
                # class as decode's step-scoped windows).
                moe_w2_delta.mark_seen(tier.seen[layer_key],
                                       topk_ids.view(-1).long())
            if use_pf4:
                if torch.cuda.is_current_stream_capturing():
                    tier.notify_capture()
                elif _PREFILL_FP4_ENSURE:
                    # decode-class guarantee: fetch the chunk's routed set
                    # into the pool before the desc build reads slot_table
                    # (base-cache prefill idiom; layer pins rotate).
                    if not _pf4_ensure_logged:
                        _pf4_ensure_logged = True
                        logger.info("moe_w2 prefill-FP4 ensure mode: eager "
                                    "chunk working sets fetched to the FP4 "
                                    "tier (first call: layer %d, T=%d)",
                                    layer_key, T)
                    tier.ensure_resident(layer_key, topk_ids.view(-1))
                slot_row = tier.slot_table[layer_key]
                pool_ptr = tier.pool.data_ptr()
            else:
                slot_row = ws["no_slots"]
                pool_ptr = ws["a1"].data_ptr()  # never dereferenced (m4=0)
        if use_pf4:
            # w2 tables at pair granularity (diverted pairs m=0) + w4 tables
            # at SUB-ENTRY granularity (p*4+sub, m<=4 — the w4/w4q kernel M
            # limit). w4 entries always read ROW-MAJOR a1/a2 (AFRAG repack
            # leaves the row-major source buffers intact).
            _desc_build_kernel_prefill4[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(),
                ws["c13"].data_ptr(), a2_base.data_ptr(),
                ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                ws["a1"].data_ptr(), ws["a2"].data_ptr(),
                st["planes13"].data_ptr(), st["sc13"].data_ptr(),
                st["planes2"].data_ptr(), st["sc2"].data_ptr(), pool_ptr,
                st["planes13"].shape[1], st["sc13"].shape[1],
                st["planes2"].shape[1], st["sc2"].shape[1],
                tier.slot_bytes, tier.w13_bytes,
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        else:
            _desc_build_kernel[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(),
                ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                st["planes13"].data_ptr(), st["sc13"].data_ptr(),
                st["planes2"].data_ptr(), st["sc2"].data_ptr(), pool_ptr,
                st["planes13"].shape[1], st["sc13"].shape[1],
                st["planes2"].shape[1], st["sc2"].shape[1],
                (tier.slot_bytes if tier is not None
                 else moe_w2_delta.SLOT_BYTES),
                (tier.w13_bytes if tier is not None
                 else moe_w2_delta.W13_BYTES),
                # row strides (bytes). H-side: a1 fp8 [H], as1 f32 [H/32]
                # (a32 per-32 groups; the a128 lineage was f32 [H/128]), c2
                # bf16 [H]. per-rank intermediate side: c13 bf16 [2I], a2
                # fp8 [I], as2 f32 [I/32]. K13 = H, K2 = I; GLM-5.x gets
                # H=6144, TP shards shrink I.
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        if (tier is not None and (not prefill or use_pf4)
                and moe_w2_delta.split_enabled()):
            # split-FP4: the extra 8-field tables for moe_w4q_mm (base/bs =
            # the resident plane rows, ref = the slot's quintal sections).
            # Prefill emits sub-entries (M<=4) via the prefill4 variant.
            d4s = ws["desc4s"]
            builder = (_desc_build_kernel_w4s_prefill4 if use_pf4
                       else _desc_build_kernel_w4s)
            a1_w4s = ws["a1"] if use_pf4 else a1_base
            a2_w4s = ws["a2"] if use_pf4 else a2_base
            builder[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, d4s,
                a1_w4s.data_ptr(), ws["as1"].data_ptr(),
                ws["c13"].data_ptr(), a2_w4s.data_ptr(),
                ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                st["planes13"].data_ptr(), st["sc13"].data_ptr(),
                st["planes2"].data_ptr(), st["sc2"].data_ptr(), pool_ptr,
                st["planes13"].shape[1], st["sc13"].shape[1],
                st["planes2"].shape[1], st["sc2"].shape[1],
                tier.slot_bytes, tier.w13_bytes,
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
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
    # _desc_build_kernel_base_delta_split against the coupled base slots).
    # Prefill-FP4 launches the w4 tier over SUB-ENTRIES (4 x M<=4 rows per
    # 16-token pair -> grid pairs*4), see _desc_build_kernel_prefill4.
    use_w4s = (tier is not None and not prefill
               and moe_w2_delta.split_enabled())
    if tier is not None and not prefill:
        if use_w4s:
            _launch("w4q", st["K13"], ws["desc4s"][0], st["N13"], pairs,
                    stream)
        else:
            _launch("w4", st["K13"], d[2], st["N13"], pairs, stream)
    elif use_pf4:
        if moe_w2_delta.split_enabled():
            _launch("w4q", st["K13"], ws["desc4s"][0], st["N13"], pairs * 4,
                    stream)
        else:
            _launch("w4", st["K13"], d[2], st["N13"], pairs * 4, stream)
    act = ws["act"][:slots]
    if st.get("swiglu_limit") is None:
        torch.ops._C.silu_and_mul(act, ws["c13"][:slots])
    else:
        torch.ops._C.silu_and_mul_with_clamp(
            act,
            ws["c13"][:slots],
            st["swiglu_limit"],
            st.get("swiglu_alpha", 1.0),
            st.get("swiglu_beta", 0.0),
        )
    # mid-pipeline requant (group _G2, default 32 — the a128-era group-128
    # requant here was one of the two activation-precision gaps vs native)
    _quant_a32(act, ws["a2"][:slots], ws["as2"][:slots], _G2)
    if use_afrag:
        _afrag_repack(ws["a2"], ws["a2f"], pairs, st["K2"])
    _launch(w2tier, st["K2"], d[1], st["N2"], pairs, stream)
    if tier is not None and not prefill:
        if use_w4s:
            _launch("w4q", st["K2"], ws["desc4s"][1], st["N2"], pairs, stream)
        else:
            _launch("w4", st["K2"], d[3], st["N2"], pairs, stream)
    elif use_pf4:
        if moe_w2_delta.split_enabled():
            _launch("w4q", st["K2"], ws["desc4s"][1], st["N2"], pairs * 4,
                    stream)
        else:
            _launch("w4", st["K2"], d[3], st["N2"], pairs * 4, stream)

    # ---- weighted unpermute (pad slots masked out), DETERMINISTIC.
    # The old `out.index_add_(0, rows, c2*w)` scattered with atomics, so the
    # f32 accumulation ORDER varied run-to-run: identical inputs wobbled by
    # up to ~1.6e-2 abs on prefill, and single-token probes produced a small
    # set of bit-distinct logit variants — the root cause of the "greedy
    # decode is not reproducible" investigation (PP_DETERMINISM.md; it was
    # never PP-specific). Invert the permutation, then one Triton program
    # gathers and reduces top_k in fixed order for each token/H tile. This
    # removes the previous [T*top_k,H] FP32 gath tensor (1.83 GiB at
    # T=9984, top_k=8, H=6144) and its equally large weighted temporary.
    if not _FUSED_UNPERMUTE:
        w = topk_weights.reshape(-1)[sorted_ids.clamp(max=T * top_k - 1)]
        w = torch.where(valid, w, torch.zeros_like(w)).to(torch.float32)
        if miss_rows is not None:
            w = w * miss_rows.to(torch.float32)
        dump = T * top_k
        dst = torch.where(
            valid, sorted_ids, torch.full_like(sorted_ids, dump)).long()
        gath = torch.zeros(
            dump + 1, H, dtype=torch.float32, device=dev)
        gath.index_copy_(
            0, dst, ws["c2"][:slots].float() * w.unsqueeze(1))
        return gath[:dump].view(T, top_k, H).sum(dim=1).to(x.dtype)

    n_routes = T * top_k
    inverse = ws["inv_sorted"][:n_routes]
    _invert_sorted_ids_kernel[(triton.cdiv(slots, 256),)](
        sorted_ids,
        inverse,
        slots,
        n_routes,
        BLOCK=256,
    )
    out = torch.empty((T, H), dtype=x.dtype, device=dev)
    row_mask_ptr = miss_rows if miss_rows is not None else inverse
    _deterministic_unpermute_kernel[
        (T, triton.cdiv(H, 256))
    ](
        ws["c2"],
        topk_weights,
        inverse,
        row_mask_ptr,
        out,
        H,
        ws["c2"].stride(0),
        topk_weights.stride(0),
        topk_weights.stride(1),
        out.stride(0),
        TOP_K=top_k,
        HAS_ROW_MASK=miss_rows is not None,
        BLOCK_H=256,
        num_warps=4,
    )
    return out


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


def shutdown() -> None:
    """Release model-owned registries/workspaces while retaining cubin modules."""
    global _n_created, _skip_logged, _stream_logged
    global _cutoff_cache, _resident_fit_checked
    _LAYERS.clear()
    _WS.clear()
    _n_created = 0
    _skip_logged = False
    _stream_logged = False
    _cutoff_cache = None
    _resident_fit_checked = False
    ready.cache_clear()
