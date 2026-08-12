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
import time

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import moe_w2_mapped_host
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
# Diagnostic A/B only: DS4 declares swiglu_limit=10, which remains mandatory
# by default. Setting this to 0 reproduces the historical W2 activation path
# without changing the native/shared-expert path.
_SWIGLU_CLAMP = os.getenv("VLLM_MOE_W2_SWIGLU_CLAMP", "1") == "1"
# DeepGEMM's native DS4 path evaluates the clamped gated activation in FP32,
# then rounds the product through BF16 before activation quantization. Keep
# the old BF16-intermediate custom op as an A/B escape hatch.
_SWIGLU_CLAMP_FP32 = os.getenv(
    "VLLM_MOE_W2_SWIGLU_CLAMP_FP32", "1") == "1"
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

# ---- EXL3 trellis base tier (VLLM_MOE_W2_EXL3_BASE=1) --------------------
# v1 (2026-07-29): the RESIDENT base tier is served from an EXL3 pack
# (exl3-ab/build_exl3_pack.py; w13@K2 + w2@K{1,2,3} per expert) through
# exllamav3's grouped trellis GEMM instead of the 2-bit scalar planes —
# moe_w2_exl3.Exl3BaseTier holds per-layer pointer tables over the pack
# tensors and computes the full routed block (parity 0.0007 vs pack
# reconstruction, block error 0.26-0.30 vs FP4 at 2.01 bpw on real
# routing — EXL3_BASE_TIER_FEASIBILITY.md Phase 2). The 2-bit planes are
# NOT built in this mode (the ~65 GiB tier replaces them and both would
# not fit), the checkpoint expert bytes are never staged (loader-skip,
# same mechanism as boot-from-pack) and no FP4 planes are packed.
#
# v1 support envelope, REFUSED fail-closed at load otherwise:
#   resident TP1 only (no base cache, no FP4 need-pool, no gate) and
#   eager only (--enforce-eager; the per-token dispatch loop is host
#   code — cudagraph capture would bake a single replayed token set).
# v2 items (not here): FP4 need-pool interplay via forward_topk's
# fp4_mask (dispatch semantics already parity-proven), gate, TP via
# exl3_mgemm's min/max_index, cudagraph/fused path.
#   VLLM_MOE_W2_EXL3_BASE  1 = serve the base tier from the EXL3 pack
#   VLLM_MOE_W2_EXL3_PACK  pack dir (exl3-l{li:02d}.safetensors files)
#   VLLM_MOE_W2_EXL3_W2K   down-proj K variant 1|2|3 (default 2; also
#                          read by Exl3BaseTier when unset there)
#   VLLM_MOE_W2_EXL3_REPO  exllamav3 checkout for the JIT ext (consumed
#                          by moe_w2_exl3._load_ext)
_EXL3_BASE = os.getenv("VLLM_MOE_W2_EXL3_BASE", "0") == "1"
_EXL3_PACK = os.getenv("VLLM_MOE_W2_EXL3_PACK", "/serve-tau/exl3-packs-ds4")
_EXL3_W2K = int(os.getenv("VLLM_MOE_W2_EXL3_W2K", "2"))
# layer_key -> (Exl3BaseTier holding that one layer, pack layer index)
_EXL3_TIERS: dict[int, tuple] = {}
_exl3_config_checked = False
# v2 FP4 need-pool slot geometry (DS4 TP1 is uniform across layers): byte
# offsets of the [fp4_13|sc13|fp4_2|sc2] sections + matrix shapes, set once by
# _exl3_stage_fp4 and read by _exl3_fp4_apply. None until the pool is staged.
_EXL3_FP4_GEOM: dict | None = None
# v2 FP4 apply weight cache: dequantized bf16 expert weights keyed by
# (layer_key, expert). The FP4 weights of a given expert are INVARIANT (the
# checkpoint's FP4; pool eviction only moves WHERE they live, never the
# values), so this cache is correctness-safe with NO slot invalidation. It
# turns the eager apply from "re-dequant the resident set every token" into a
# one-time-per-expert cost, so decode speed approaches the base's. Bounded
# LRU (insertion-ordered dict): N x (w13 32 + w2 16 MiB bf16) = N x 48 MiB.
_EXL3_FP4_WCACHE: dict = {}
_EXL3_FP4_WCACHE_MAX = int(os.getenv("VLLM_MOE_W2_EXL3_FP4_CACHE", "96"))
# v2-perf: serve the FP4-resident experts through the production moe_w4_mm
# cubin (reads the 4-bit pool slots directly, fp8-a32 activations, ~2 launches
# /layer) instead of the eager torch dequant+GEMM apply. Off by default (the
# torch path is the validated fallback); VALIDATE=1 runs both and logs the
# rel-err on the first layers to certify the cubin path against the torch one.
_EXL3_FP4_CUBIN = os.getenv("VLLM_MOE_W2_EXL3_FP4_CUBIN", "0") == "1"
_EXL3_FP4_VALIDATE = os.getenv("VLLM_MOE_W2_EXL3_FP4_VALIDATE", "0") == "1"
_exl3_fp4_val_n = 0
# v2-perf Stage B: allow cudagraph capture of the EXL3 decode (the only lever
# that breaks the eager per-layer launch wall). Requires the capture-safe
# forward: base served for ALL experts with FP4-resident ones zeroed via
# masked weights (fixed shape, no compaction/sync) + the cubin FP4 apply
# (kernel-based). Implies the cubin apply. Off by default (eager path).
_EXL3_CUDAGRAPH = os.getenv("VLLM_MOE_W2_EXL3_CUDAGRAPH", "0") == "1"
if _EXL3_CUDAGRAPH:
    _EXL3_FP4_CUBIN = True   # the torch apply's per-expert loop is not capturable
# ---- Δ-pool mode (P6, [M] 2026-07-30): the need-pool holds EXL3 residual
# DELTA slots (pack v3: exl3-delta-l*.safetensors, independent per-expert
# residual tensors over the base pack; built by internal/exl3-sr-poc/
# build_delta_pack.py) instead of FP4 planes. The DeltaTier/gate/replay
# machinery is unchanged (slots are opaque bytes); the apply computes the
# full block for pool-resident experts with base+Δ summed PRE-activation
# (Exl3BaseTier.forward_topk_dual — weight-level additivity, capture-safe).
# Slot = 6.03 MiB vs 12.75 FP4 -> ~2.1x more pool coverage per GiB, and the
# checkpoint experts are loader-skipped like v1 (no ~138 GiB FP4 host pin;
# the Δ host store pins ~65 GB instead).
_EXL3_DELTA_PACK = os.getenv("VLLM_MOE_W2_EXL3_DELTA_PACK", "")
# P6-perf: serve base+Δ in ONE unified pass (6 mgemm/token) instead of
# masked-base + dual (9, with duplicated base g/u work). Algebraically
# identical (parity-tested); flag kept for A/B fallback.
_EXL3_DELTA_UNIFIED = os.getenv("VLLM_MOE_W2_EXL3_DELTA_UNIFIED",
                                "1") == "1"
# Track 2.5 (2026-08-05, [B]): serve the M=1 decode step from the flat
# exl3_decode_wave kernel instead of the six mgemm barrier-trains (2.0x
# decode uplift, sonda 3.0 -> af76cdd). Same base/base+Δ block as the
# unified path (parity 9.8e-4 vs forward_topk_unified on real packs); a
# drop-in inside the capture-safe unified branch, gated to M=1 and the
# pack-v3 serving pair (base 2,2,1 -> w2_k=1; Δ 2,2,3 -> dk13=2,dk2=3) the
# kernel hardcodes. Falls back to unified when M>1 or the config differs.
_EXL3_WAVE = os.getenv("VLLM_MOE_W2_EXL3_WAVE", "0") == "1"
_exl3_wave_seen: list = []   # one-shot activation-log sentinel
# M8 charter F2 ([K] 2026-08-09, §2bis): serve M ∈ [2, 8] decode steps on
# the M=8-native wave kernels — trellis decode ONCE per union expert, HMMA
# over 8 token rows — instead of falling off the wave to unified/masked
# (spec-decode exclusion) or the anti-scaling M·k expansion. The union of
# the step's routed experts is partitioned into fixed groups of 6 and the
# m8 launcher runs once per group (moe_w2_exl3.build_m8_groups /
# forward_topk_wave_m8). Same pack-v3 serving-pair gate as _EXL3_WAVE;
# additionally requires an ext build exposing exl3_decode_wave_dual_m8
# with its m8 cubins resolved (tier.ext_wave_m8_available(); missing ->
# fallbacks unchanged). M=1 stays on the M=1 wave route. Default OFF.
_EXL3_WAVE_M8 = os.getenv("VLLM_MOE_W2_EXL3_WAVE_M8", "0") == "1"
_exl3_wave_m8_seen: list = []   # one-shot activation-log sentinel
_exl3_wave_m8g_seen: list = []  # one-shot activation-log sentinel (graph)
_EXL3_DELTA_GEOM: dict | None = None

# 9b step-1 h_l capture ([P] 2026-08-03, default-off): mirrors the gate's
# VLLM_MOE_W2_GATE_HCAP_DIR knob — when set, the EXL3 forward copies each
# layer's MoE input row into this persistent [n_layers, H] buffer
# (in-graph, fixed shapes). Consumers: moe_w2_gate.hcap_store via the
# runner. Measurement-only; the disabled path adds one falsy check.
_HCAP_ON = bool(os.getenv("VLLM_MOE_W2_GATE_HCAP_DIR", ""))
_HCAP_BUF: torch.Tensor | None = None


def _hcap_buf(layer_key: int, x: torch.Tensor) -> torch.Tensor:
    global _HCAP_BUF
    if _HCAP_BUF is None:
        n_layers = max(k for k in _EXL3_TIERS) + 1 if _EXL3_TIERS else 64
        _HCAP_BUF = torch.zeros(n_layers, x.shape[-1], dtype=torch.half,
                                device=x.device)
    return _HCAP_BUF[layer_key]


def hcap_snapshot() -> torch.Tensor | None:
    """Decision-time clone of the h_l buffer (runner-side, eager)."""
    return _HCAP_BUF.clone() if _HCAP_BUF is not None else None
_EXL3_DELTA_VALIDATE = os.getenv("VLLM_MOE_W2_EXL3_DELTA_VALIDATE",
                                 "0") == "1"
_exl3_delta_val_n = 0
_exl3_delta_abi_checked = False

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
    # Draft/speculator modules (DSpark heads, Eagle) are SEPARATE models
    # built under a non-default compilation model tag; a draft whose layer
    # indices restart at 0 would collide with the main pack/planes
    # namespace, so non-backbone tags are excluded by default. Draft
    # numerics only move the acceptance rate — verify forwards through
    # the main stack are the correctness boundary.
    #
    # EXCEPTION (opt-in, VLLM_MOE_W2_DRAFT=1): the DSpark head numbers its
    # layers DISJOINTLY (layers.{num_hidden_layers+i}) so it CAN share the
    # W2 planes path — set VLLM_MOE_W2_NUM_LAYERS to main+draft so the
    # cutoff admits it. Motivation: the head is 3 full MoE decoder layers
    # (~10-19 GiB native) which do not fit a 96 GiB card next to the
    # fully-resident main planes; as 2-bit planes they cost ~5 GiB, and
    # rejection sampling keeps the OUTPUT distribution exactly the
    # target's regardless of draft precision.
    from vllm.compilation import backends as _cb
    _tag = getattr(_cb, "model_tag", "backbone")
    if _tag != "backbone" and not (
            _tag == "dspark_head"
            and os.getenv("VLLM_MOE_W2_DRAFT", "0") == "1"):
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


def _planes_cache_parts(
    planes13,
    sc13,
    planes2,
    sc2,
    fp13,
    fp2,
    *,
    allow_separate_delta: bool = False,
):
    """Parts persisted in the existing planes-cache representation.

    With the corrected direct loader, the standalone delta pack is the
    authoritative FP4 cache. Avoid writing a second 138 GiB FP4 copy into
    the planes cache; all other configurations retain their existing parts.
    """
    parts = dict(
        planes13=planes13,
        sc13=sc13,
        planes2=planes2,
        sc2=sc2,
    )
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    separate_delta = (
        allow_separate_delta
        and os.getenv("VLLM_MOE_W2_FAST_LOAD", "0") == "1"
        and bool(os.getenv("VLLM_MOE_W2_STORE_DIR", "").strip())
        and moe_w2_delta.enabled()
        and not moe_w2_delta.base_enabled()
        and not moe_w2_delta.split_enabled()
    )
    if not separate_delta:
        parts.update(fp13=fp13, fp2=fp2)
    return parts


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
_fast_loader = None
_fast_probe_s = 0.0
_fast_probe_layers = 0


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
            # V2 integration (2026-08-01, the DSpark runner): the V2 GPU
            # model runner carries the manager safe-point + step_end hooks
            # only. That is the complete contract for RESIDENT/delta-only
            # serving (a pool miss is served from the resident planes —
            # correct base quality, no replay needed). The base-cache
            # mandatory replay loops and the confidence-gate re-forwards
            # are V1-runner code and are NOT ported: fail closed on any
            # config that needs them.
            from vllm.model_executor.layers.quantization.utils import (
                moe_w2_delta)
            if moe_w2_delta.base_enabled():
                raise ValueError(
                    "VLLM_MOE_W2 base cache (VLLM_MOE_W2_BASE_CACHE_GB) is "
                    "not integrated with Model Runner V2: the mandatory "
                    "base replay loop is absent — a pool miss would be a "
                    "silent quality corruption. Serve base-cache configs "
                    "on the V1 runner.")
            # The confidence gate is integrated with V2 (2026-08-02): the
            # runner's _moe_w2_gate_reforward ports the V1 inline TP path
            # (decision consensus + fixed-point promote->re-forward before
            # sampling). Only the PP broadcast leg is V1-only, and PP is
            # refused below anyway.
            if cfg.parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "VLLM_MOE_W2 on Model Runner V2 does not support "
                    "pipeline parallelism.")
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
        if not _SWIGLU_CLAMP:
            logger.warning_once(
                "moe_w2_cubit: routed-expert SwiGLU clamp disabled by "
                "VLLM_MOE_W2_SWIGLU_CLAMP=0 (diagnostic A/B only)")
            limit = None
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


def plan_pack_skip(layer, *, allow_direct_delta: bool = False) -> bool:
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
    if _EXL3_BASE:
        # EXL3 resident base: the pack replaces the checkpoint experts
        # entirely, so v1 (no FP4 pool) never needs their bytes staged —
        # probe config + pack coverage here (fail-closed at CREATE) and
        # fall through to the shared loader-skip stubbing below.
        # _exl3_build_layer registers the tier from the pack at
        # process_weights_after_loading time.
        _exl3_check_config()
        lidx = _exl3_pack_index(layer, key)
        if moe_w2_delta.enabled():
            if _EXL3_DELTA_PACK:
                # Δ-pool (P6): the pool stages from the DELTA PACK on disk,
                # never from checkpoint experts — loader-skip like v1 (fall
                # through to the shared stubbing below). Probe the layer's
                # delta file at CREATE so a partial pack refuses early.
                dpath = os.path.join(_EXL3_DELTA_PACK,
                                     f"exl3-delta-l{lidx:02d}.safetensors")
                if not os.path.exists(dpath):
                    raise ValueError(
                        f"moe_w2 EXL3 Δ-pool: delta pack layer file missing: "
                        f"{dpath}")
            else:
                # v2 FP4 need-pool: there is NO pre-built FP4 pack on disk, so
                # the pool's host store is staged from the checkpoint mxfp4
                # experts in _exl3_build_layer. Keep them host-staged (do NOT
                # loader-skip); STREAM_BUILD still triggers the per-layer
                # build as its experts land, so peak staging stays ~one layer.
                return False
    elif moe_w2_delta.base_enabled():
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
        # GPU-resident: cached 2-bit planes are the runtime base. In the
        # corrected fast path the matching delta pack supplies FP4 directly,
        # so the planes cache does not need a duplicate FP4 representation.
        lidx = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
        base_sizes = _pc.expected_sizes(
            E, N13, K13, N2, K2, want_fp4=False)
        t0 = time.perf_counter()
        files = None if lidx is None else _pc.cache_layer_files(
            lidx, base_sizes)
        global _fast_probe_s, _fast_probe_layers
        _fast_probe_s += time.perf_counter() - t0
        _fast_probe_layers += 1
        if files is None:
            return False
        direct = (
            allow_direct_delta
            and os.getenv("VLLM_MOE_W2_FAST_LOAD", "0") == "1"
            and moe_w2_delta.enabled()
            and not moe_w2_delta.split_enabled()
        )
        if direct:
            n_keys = _layer_cutoff() + 1
            delta_slot = N13 * K13 // 2 + N2 * K2 // 2
            if not pack_has_layer(
                    "delta", key, n_keys, E, delta_slot):
                return False
            from vllm.model_executor.layers.quantization.utils.moe_w2_fast_loader import (  # noqa: E501
                DirectPlaneBatchLoader,
            )
            global _fast_loader
            if _fast_loader is None:
                workers = int(os.getenv(
                    "VLLM_MOE_W2_FAST_LOAD_WORKERS", "4"))
                batch_layers = int(os.getenv(
                    "VLLM_MOE_W2_FAST_LOAD_BATCH_LAYERS", "12"))
                _fast_loader = DirectPlaneBatchLoader(
                    workers=workers, batch_layers=batch_layers)
                logger.info(
                    "moe_w2 FAST-LOAD direct armed: existing planes cache "
                    "+ matching delta pack; workers=%d batch_layers=%d; "
                    "no CPU W2 projection or reconstruction",
                    workers, batch_layers)
            _fast_loader.add_layer(key, files, base_sizes)
            layer._moe_w2_direct_cache = True
        elif moe_w2_delta.enabled():
            # Preserve the original serial cache path for configurations that
            # keep FP4 parts alongside the base planes.
            full_sizes = _pc.expected_sizes(
                E, N13, K13, N2, K2, want_fp4=True)
            if _pc.cache_layer_files(lidx, full_sizes) is None:
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
    if key + 1 == _layer_cutoff():
        logger.info(
            "moe_w2 FAST-LOAD cache eligibility: %d layer probes in "
            "%.3f s",
            _fast_probe_layers, _fast_probe_s)
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


def _exl3_check_config() -> None:
    """EXL3 v1 support envelope — refuse incompatible configs fail-closed
    at LOAD time with an actionable message instead of serving a mixed
    tier stack the dispatch has no branch for (the FP4/base-cache/gate
    machinery keys off the 2-bit slot tables this mode never builds)."""
    global _exl3_config_checked
    if _exl3_config_checked:
        return
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_delta, moe_w2_gate)
    problems = []
    # v2 (2026-07-29): the FP4 need-pool (DELTA_GB>0) and the confidence gate
    # (GATE=1) are now SUPPORTED over the EXL3 base — _exl3_forward composes
    # them additively (EXL3 serves the base experts, FP4-resident experts are
    # masked out and added from the pool; the gate/runner replay loop is
    # format-agnostic). Still refused fail-closed:
    #  - the 2-bit base CACHE: the resident EXL3 trellis tier IS the base, and
    #    both would not fit; the FP4 pool here promotes from its own pinned
    #    host store staged from the checkpoint (no 2-bit base tier).
    #  - split-quintal FP4 (DELTA_SPLIT): the eager torch apply reconstructs
    #    FULL e2m1 nibble planes; the radix-5 quintal refinement is only
    #    decodable against a resident 2-bit base slot the w4q kernel reads.
    if moe_w2_delta.base_enabled():
        problems.append("2-bit base cache active (unset "
                        "VLLM_MOE_W2_BASE_CACHE_GB; the EXL3 tier is the "
                        "resident base)")
    if moe_w2_delta.split_enabled():
        problems.append("split-FP4 active (unset VLLM_MOE_W2_DELTA_SPLIT; "
                        "EXL3 v2 serves the full-FP4 need-pool only)")
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size)
        tp = get_tensor_model_parallel_world_size()
        if tp > 1:
            problems.append(f"TP world size {tp} > 1")
    except Exception:  # noqa: BLE001
        # offline tools without a distributed context; the per-rank shape
        # assert in _exl3_build_layer still catches TP-sharded experts.
        pass
    if not _EXL3_CUDAGRAPH:
        try:
            from vllm.config import get_current_vllm_config
            if not get_current_vllm_config().model_config.enforce_eager:
                problems.append("cudagraphs enabled (serve with "
                                "--enforce-eager, or set "
                                "VLLM_MOE_W2_EXL3_CUDAGRAPH=1 for the "
                                "capturable path)")
        except Exception:  # noqa: BLE001
            # no current config (offline tools); the capture assert in
            # _exl3_forward remains the hard backstop.
            pass
    if problems:
        raise ValueError(
            "VLLM_MOE_W2_EXL3_BASE supports resident TP1, eager, with an "
            "optional full-FP4 need-pool + gate (no 2-bit base cache, no "
            "split-FP4, no TP, --enforce-eager); refused: "
            + "; ".join(problems))
    if not os.path.isdir(_EXL3_PACK):
        raise ValueError(
            f"VLLM_MOE_W2_EXL3_BASE=1 but pack dir {_EXL3_PACK!r} does "
            "not exist (set VLLM_MOE_W2_EXL3_PACK)")
    _exl3_config_checked = True


def _exl3_pack_index(layer, layer_key: int) -> int:
    """Pack file index for this layer, VERIFIED — pack files are keyed by
    the ABSOLUTE transformer layer index (build_exl3_pack.py writes
    exl3-l{li:02d} with manifest["layer"] == li). The transformer index is
    parsed from the layer NAME (never guessed from creation order); on
    DS4-Flash all 43 layers are MoE (first_k_dense_replace unset), so it
    coincides with layer_key (the is_w2_layer creation counter) and v1
    asserts that coincidence: a divergence means a dense-offset model this
    mode was never validated on — refuse rather than silently serve."""
    name = getattr(layer, "layer_name", "")
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as _pc)
    lidx = _pc.layer_idx_from_name(name)
    if lidx is None:
        raise ValueError(
            f"moe_w2 EXL3: cannot parse a transformer layer index from "
            f"layer name {name!r}")
    if not (0 <= lidx < _layer_cutoff()):
        raise ValueError(
            f"moe_w2 EXL3: layer index {lidx} (from {name!r}) outside the "
            f"main stack [0, {_layer_cutoff()})")
    if lidx != layer_key:
        raise ValueError(
            f"moe_w2 EXL3 v1: layer_key {layer_key} != transformer layer "
            f"index {lidx} (from {name!r}) — dense-offset MoE stacks are "
            "not validated with the EXL3 pack mapping; refusing")
    path = os.path.join(_EXL3_PACK, f"exl3-l{lidx:02d}.safetensors")
    if not os.path.exists(path):
        raise ValueError(f"moe_w2 EXL3: pack layer file missing: {path}")
    return lidx


def _exl3_stage_fp4(layer, layer_key: int, dev, E: int,
                    N13: int, K13: int, N2: int, K2: int) -> None:
    """v2: stage this layer's FP4 need-pool host sections from the checkpoint
    mxfp4 experts and register the FP4 delta tier. The EXL3 trellis tier is
    the RESIDENT base (no 2-bit base cache), so — unlike the FP4-over-base
    case — the FP4 slots must carry their OWN block-32 scales:
    [fp4_13|sc13|fp4_2|sc2]. Requires the checkpoint experts host-staged
    (plan_pack_skip returns False when the pool is on)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        mxfp4_to_nibbles, pack_fp4_fragment_major, pack_scales)
    w13 = layer.w13_weight.data          # [E, 2I, H/2] u8 (cpu)
    s13 = layer.w13_weight_scale.data    # [E, 2I, H/32] u8
    w2 = layer.w2_weight.data            # [E, H, I/2] u8
    s2 = layer.w2_weight_scale.data      # [E, H, I/32] u8
    c13, sc13b = N13 * K13 // 2, N13 * K13 // 32
    c2, sc2b = N2 * K2 // 2, N2 * K2 // 32
    ftier = moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                  w13_bytes=c13 + sc13b, w2_bytes=c2 + sc2b)
    if ftier is None:  # enabled() is True here, so this should not happen
        raise RuntimeError("moe_w2 EXL3 v2: FP4 tier requested but get_tier "
                           "returned None (VLLM_MOE_W2_DELTA_GB=0?)")
    fp13 = torch.empty(E, c13, dtype=torch.uint8, device=dev)
    fsc13 = torch.empty(E, sc13b, dtype=torch.uint8, device=dev)
    fp2 = torch.empty(E, c2, dtype=torch.uint8, device=dev)
    fsc2 = torch.empty(E, sc2b, dtype=torch.uint8, device=dev)
    chunk = 32
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = w13[e0:e1].to(dev, non_blocking=True)
        sg = s13[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            fp13[e0 + i] = pack_fp4_fragment_major(mxfp4_to_nibbles(wg[i]))
            fsc13[e0 + i] = pack_scales(sg[i])
        wg = w2[e0:e1].to(dev, non_blocking=True)
        sg = s2[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            fp2[e0 + i] = pack_fp4_fragment_major(mxfp4_to_nibbles(wg[i]))
            fsc2[e0 + i] = pack_scales(sg[i])
    ftier.add_layer_host_sections(layer_key, (fp13, fsc13), (fp2, fsc2))
    del fp13, fsc13, fp2, fsc2
    global _EXL3_FP4_GEOM
    _EXL3_FP4_GEOM = dict(
        N13=N13, K13=K13, N2=N2, K2=K2,
        off_s13=c13, off_c2=c13 + sc13b, off_s2=c13 + sc13b + c2,
        slot_bytes=c13 + sc13b + c2 + sc2b)
    if layer_key == 0:
        logger.info(
            "moe_w2 EXL3 v2: FP4 need-pool staging from checkpoint "
            "(own-scale slots %.2f MiB [fp4_13|sc13|fp4_2|sc2]; policy=need)",
            (c13 + sc13b + c2 + sc2b) / 2**20)


def _exl3_stage_delta(layer_key: int, lidx: int, dev, E: int) -> None:
    """P6 ([M] 2026-07-30): stage this layer's Δ-pool host sections from the
    delta pack. Slot layout = moe_w2_exl3.delta_slot_geom (w13 section |
    w2 section, opaque u8 for the tier). Verifies the pack's ABI binding
    (delta manifest -> base pack SHA256) by hashing ONE base layer file per
    boot (full-pack hashing costs minutes; one file is a spot check of the
    same build)."""
    import hashlib
    import json as _json
    from safetensors import safe_open
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_exl3 import (
        delta_slot_geom)
    global _EXL3_DELTA_GEOM, _exl3_delta_abi_checked
    path = os.path.join(_EXL3_DELTA_PACK,
                        f"exl3-delta-l{lidx:02d}.safetensors")
    with open(os.path.join(
            _EXL3_DELTA_PACK,
            f"exl3-delta-l{lidx:02d}.manifest.json")) as f:
        man = _json.load(f)
    dk = int(man["kd"])
    # per-projection delta K ([P] 2026-08-02, the (2,2,1)+Δ(2,2,3) track):
    # legacy manifests carry only "kd" (uniform); new ones add kd13/kd2 +
    # base_w2k. Legacy defaults reproduce the old behavior exactly.
    kd13 = int(man.get("kd13", dk))
    kd2 = int(man.get("kd2", dk))
    base_w2k = int(man.get("base_w2k", 2))
    base_cb = man.get("base_codebook", man.get("codebook", "3inst"))
    delta_cb = man.get("delta_codebook", "mul1")
    if base_cb != "3inst" or delta_cb != "mul1":
        raise ValueError(
            f"moe_w2 EXL3 Δ-pool: pack codebooks base={base_cb!r} / "
            f"delta={delta_cb!r} unsupported (dispatch expects 3INST base "
            "+ MUL1 delta — the pack-v3 build convention)")
    if base_w2k != _EXL3_W2K:
        raise ValueError(
            "moe_w2 EXL3 Δ-pool: delta pack binds to the "
            f"w2@k{base_w2k} base (manifest base_variant="
            f"{man.get('base_variant')!r}), but serving W2K={_EXL3_W2K} — "
            "rebuild the delta pack for this mix or fix the W2K env")
    if not _exl3_delta_abi_checked:
        base_file = os.path.join(_EXL3_PACK, man["base_pack"])
        h = hashlib.sha256()
        with open(base_file, "rb") as bf:
            while True:
                b = bf.read(1 << 24)
                if not b:
                    break
                h.update(b)
        if h.hexdigest() != man["base_pack_sha256"]:
            raise ValueError(
                "moe_w2 EXL3 Δ-pool ABI violation: delta layer "
                f"{lidx} binds to base {man['base_pack']} sha256 "
                f"{man['base_pack_sha256'][:16]}… but the served base pack "
                f"file hashes to {h.hexdigest()[:16]}…")
        _exl3_delta_abi_checked = True
    geom = delta_slot_geom(kd13, kd2)
    geom["mults"] = (0, 0x83DCD12D)   # MUL1 delta stream (pack v3)
    ftier = moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                  w13_bytes=geom["sec13"],
                                  w2_bytes=geom["sec2"])
    if ftier is None:
        raise RuntimeError("moe_w2 EXL3 Δ-pool: tier requested but get_tier "
                           "returned None (VLLM_MOE_W2_DELTA_GB=0?)")
    rows13 = torch.empty(E, geom["sec13"], dtype=torch.uint8, device=dev)
    rows2 = torch.empty(E, geom["sec2"], dtype=torch.uint8, device=dev)
    with safe_open(path, "pt") as f:
        for e in range(E):
            parts = []
            for proj in ("w1", "w3"):
                for part in ("trellis", "suh", "svh"):
                    t = f.get_tensor(f"e{e}.{proj}.dk{kd13}.{part}")
                    parts.append(t.contiguous().view(torch.uint8).view(-1))
            row = torch.cat(parts)
            if row.numel() != geom["sec13"]:
                raise ValueError(
                    f"moe_w2 EXL3 Δ-pool: e{e} w13 section is "
                    f"{row.numel()} B, expected {geom['sec13']} (layer "
                    f"{lidx}; wrong pack geometry?)")
            rows13[e] = row.to(dev, non_blocking=True)
            parts = [f.get_tensor(f"e{e}.w2.dk{kd2}.{part}")
                     .contiguous().view(torch.uint8).view(-1)
                     for part in ("trellis", "suh", "svh")]
            row = torch.cat(parts)
            if row.numel() != geom["sec2"]:
                raise ValueError(
                    f"moe_w2 EXL3 Δ-pool: e{e} w2 section is {row.numel()} "
                    f"B, expected {geom['sec2']} (layer {lidx})")
            rows2[e] = row.to(dev, non_blocking=True)
    ftier.add_layer_host_sections(layer_key, (rows13,), (rows2,))
    del rows13, rows2
    _EXL3_DELTA_GEOM = geom
    if layer_key == 0:
        logger.info(
            "moe_w2 EXL3 Δ-pool: staging from delta pack %s (dk13=%d "
            "dk2=%d, base w2@k%d, slot %.2f MiB [w13:tr|suh|svh x2 | "
            "w2:tr|suh|svh]; base ABI sha %s… verified; policy=need)",
            _EXL3_DELTA_PACK, kd13, kd2, base_w2k,
            geom["slot_bytes"] / 2**20, man["base_pack_sha256"][:12])


def _exl3_build_layer(layer, layer_key: int) -> None:
    """EXL3-mode replacement for the plane build: load this layer's pack
    tensors GPU-resident and register the tier keyed by layer_key. No
    2-bit planes, no planes cache, no _LAYERS row — the EXL3 forward branch
    never reads them (and the ~65 GiB tier would not fit next to resident
    planes). v1 loader-skips the checkpoint experts entirely; v2 (FP4 pool
    on) keeps them staged and packs the pool's FP4 host sections first
    (_exl3_stage_fp4). The checkpoint params end as 0-byte stubs either way."""
    contract = _layer_contract(layer)
    _exl3_check_config()
    lidx = _exl3_pack_index(layer, layer_key)
    dev = torch.device("cuda", torch.cuda.current_device())
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    pool_on = moe_w2_delta.enabled()
    skipped = getattr(layer, "_moe_w2_pack_skip", False)
    if skipped:
        # loader-skip (plan_pack_skip): params are already 0-byte stubs
        E, N13, K13, N2, K2 = layer._moe_w2_shapes
        param_names = ()
    else:
        # staged path (v2 pool on, or plan_pack_skip missed): read shapes
        # from the CPU-staged params, then stub them (after FP4 staging)
        w13 = layer.w13_weight.data          # [E, 2I, H/2] u8 (cpu)
        w2 = layer.w2_weight.data            # [E, H, I/2] u8
        E, N13, _ = w13.shape
        _, N2, _ = w2.shape
        K13, K2 = N2, N13 // 2               # H, I (mxfp4 layout)
        param_names = ("w13_weight", "w13_weight_scale", "w2_weight",
                       "w2_weight_scale")
    if (E, N13, K13, N2, K2) != (256, 4096, 4096, 4096, 2048):
        raise ValueError(
            "moe_w2 EXL3 serves the DS4 TP1 pack geometry only "
            "(E=256, H=4096, I=2048); this layer has "
            f"E={E}, N13={N13}, K13={K13}, N2={N2}, K2={K2} "
            "(TP-sharded or non-DS4 experts)")
    if pool_on and _EXL3_DELTA_PACK:
        # P6 Δ-pool: staged from the delta pack on disk; the checkpoint
        # experts are loader-skipped (v1-style) — nothing FP4 to stage.
        _exl3_stage_delta(layer_key, lidx, dev, E)
    elif pool_on:
        if skipped:
            raise RuntimeError(
                "moe_w2 EXL3 v2: FP4 pool enabled but layer was loader-"
                "skipped — no checkpoint experts to stage the pool from "
                "(plan_pack_skip must return False when the pool is on)")
        _exl3_stage_fp4(layer, layer_key, dev, E, N13, K13, N2, K2)
        if _EXL3_FP4_CUBIN:
            # the moe_w4_mm cubin is not loaded on the EXL3 path (which uses
            # exl3_mgemm); load + verify it for the v2-perf FP4 apply.
            if not _ensure_ready():
                raise RuntimeError("moe_w2 EXL3 v2 cubin apply: moe_w2 "
                                   "cubins unavailable")
            for _kk in (K13, K2):
                if ("w4", _kk) not in _fns:
                    raise RuntimeError(
                        f"moe_w2 EXL3 v2 cubin apply: moe_w4_mm_k{_kk}_a32."
                        f"cubin not loaded (VLLM_MOE_W2_CUBIT_DIR={_DIR})")
    from vllm.model_executor.layers.quantization.utils.moe_w2_exl3 import (
        Exl3BaseTier)
    limit = contract["swiglu_limit"]
    tier = Exl3BaseTier(
        _EXL3_PACK, layers=[lidx], device=str(dev),
        w13_k=2, w2_k=_EXL3_W2K, num_experts=E, hidden=K13, inter=K2,
        act_limit=float(limit) if limit is not None else float("inf"))
    _EXL3_TIERS[layer_key] = (tier, lidx)
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in param_names:
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info(
        "moe_w2 EXL3 base: layer_key %d <- pack layer %d "
        "(w13_k=2, w2_k=%d, %.2f GiB resident, %d/%d layers loaded%s)",
        layer_key, lidx, tier.w2_k, tier.total_bytes() / 2**30,
        len(_EXL3_TIERS), _layer_cutoff(),
        ", +FP4 need-pool" if pool_on else "")
@moe_w2_mapped_host.cleanup_on_failure
def build_layer_planes(layer, layer_key: int) -> None:
    """Quantize one FusedMoE layer's experts to 2-bit planes (GPU, chunked).

    Reads the CPU-resident checkpoint params (w13_weight [E,2I,K/2] u8 etc.),
    builds fragment-major code planes + scale planes on the GPU, then
    replaces the originals with empty stubs.
    """
    if _EXL3_BASE:
        _exl3_build_layer(layer, layer_key)
        return
    moe_w2_mapped_host.require_supported_builder(layer_key, "mxfp4")
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

    mapped = moe_w2_mapped_host.allocate_canonical_layer(
        layer_key,
        {
            "planes13": (E, N13 * K13 // 4),
            "sc13": (E, N13 * K13 // 32),
            "planes2": (E, N2 * K2 // 4),
            "sc2": (E, N2 * K2 // 32),
        },
    )
    if mapped is None:
        planes13 = torch.empty(
            E, N13 * K13 // 4, dtype=torch.uint8, device=dev)
        sc13 = torch.empty(
            E, N13 * K13 // 32, dtype=torch.uint8, device=dev)
        planes2 = torch.empty(
            E, N2 * K2 // 4, dtype=torch.uint8, device=dev)
        sc2 = torch.empty(
            E, N2 * K2 // 32, dtype=torch.uint8, device=dev)
    else:
        planes13 = mapped["planes13"]
        sc13 = mapped["sc13"]
        planes2 = mapped["planes2"]
        sc2 = mapped["sc2"]

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
            _planes_cache_parts(
                planes13, sc13, planes2, sc2, fp13, fp2,
                allow_separate_delta=True))

    if tier is not None:
        _stage_fp4_host(tier, layer_key, fp13, sc13, fp2, sc2)
        del fp13, fp2
        # (the background manager is started by get_tier when the tier is
        # created; the old "start on layer NUM_LAYERS-1" trigger never fired
        # under PP, where layer_keys are local per rank and never reach 42)

    if mapped is not None:
        # Quantization writes into the canonical host-mapped tensors from the
        # normal CUDA stream. Complete those writes before descriptors retain
        # their stable UVA pointers and graph capture can begin.
        torch.cuda.synchronize(dev)
        free_bytes, _ = torch.cuda.mem_get_info(dev)
        moe_w2_mapped_host.record_constructed(
            layer_key,
            {
                "planes13": planes13,
                "sc13": sc13,
                "planes2": planes2,
                "sc2": sc2,
            },
            free_bytes,
        )

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
    moe_w2_mapped_host.require_supported_builder(layer_key, "fp8")
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
    direct = getattr(layer, "_moe_w2_direct_cache", False)
    if direct:
        if _fast_loader is None:
            raise RuntimeError("moe_w2 direct fast-load coordinator missing")
        if tier is None or layer_key not in tier._store:
            raise RuntimeError(
                f"moe_w2 direct fast-load layer {layer_key} lost its "
                "matching delta cache")
        cached = _fast_loader.take(layer_key)
    else:
        cached = planes_cache.try_load(lidx, planes_cache.expected_sizes(
            E, N13, K13, N2, K2, want_fp4=tier is not None))
    if cached is None:
        return False
    mapped = moe_w2_mapped_host.allocate_canonical_layer(
        layer_key,
        {
            "planes13": (E, N13 * K13 // 4),
            "sc13": (E, N13 * K13 // 32),
            "planes2": (E, N2 * K2 // 4),
            "sc2": (E, N2 * K2 // 32),
        },
    )
    if mapped is None:
        planes13 = cached["planes13"].view(E, -1).to(dev)
        sc13 = cached["sc13"].view(E, -1).to(dev)
        planes2 = cached["planes2"].view(E, -1).to(dev)
        sc2 = cached["sc2"].view(E, -1).to(dev)
    else:
        planes13 = mapped["planes13"]
        sc13 = mapped["sc13"]
        planes2 = mapped["planes2"]
        sc2 = mapped["sc2"]
        planes13.copy_(cached["planes13"].view(E, -1))
        sc13.copy_(cached["sc13"].view(E, -1))
        planes2.copy_(cached["planes2"].view(E, -1))
        sc2.copy_(cached["sc2"].view(E, -1))
    if tier is not None and not direct:
        _stage_fp4_host(tier, layer_key, cached["fp13"].view(E, -1),
                        sc13, cached["fp2"].view(E, -1), sc2)
    if mapped is not None:
        torch.cuda.synchronize(dev)
        free_bytes, _ = torch.cuda.mem_get_info(dev)
        moe_w2_mapped_host.record_constructed(
            layer_key,
            {
                "planes13": planes13,
                "sc13": sc13,
                "planes2": planes2,
                "sc2": sc2,
            },
            free_bytes,
        )
    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2,
                  sc2, N13, K13, N2, K2, E,
                  ("w13_weight", "w13_weight_scale", "w2_weight",
                   "w2_weight_scale"))
    if direct:
        _fast_loader.release_consumed(layer_key)
    logger.info(
        "moe_w2: layer %d planes direct from cache%s",
        lidx, ", delta direct from pack" if direct else "")
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
    moe_w2_mapped_host.require_supported_builder(layer_key, "nvfp4")
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
        planes_cache.store(
            lidx,
            _planes_cache_parts(
                planes13, sc13, planes2, sc2, fp13, fp2))

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

    mapped = moe_w2_mapped_host.layer_enabled(layer_key)
    if not mapped:
        _check_resident_fit(layer_key, dev,
                            planes13.nbytes + sc13.nbytes
                            + planes2.nbytes + sc2.nbytes)
    _LAYERS[layer_key] = dict(
        planes13=planes13, sc13=sc13, planes2=planes2, sc2=sc2,
        N13=N13, K13=K13, N2=N2, K2=K2, E=E, tl_idx=_tl,
        mapped_host=mapped,
        **contract,
    )
    # Release checkpoint copies; keep CUDA stubs so device probes stay happy.
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in param_names:
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info("moe_w2: layer %d planes built (%.2f GiB%s)", layer_key,
                (planes13.nbytes + sc13.nbytes + planes2.nbytes + sc2.nbytes)
                / 2**30, ", canonical mapped host" if mapped else "")


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
def _silu_and_mul_clamp_fp32_kernel(
    input_ptr,
    output_ptr,
    hidden,
    input_stride,
    output_stride,
    limit,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h < hidden
    gate = tl.load(
        input_ptr + row * input_stride + h, mask=mask, other=0.0
    ).to(tl.float32)
    up = tl.load(
        input_ptr + row * input_stride + hidden + h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.clamp(up, -limit, limit)
    glu = gate / (1.0 + tl.exp(-gate))
    tl.store(output_ptr + row * output_stride + h, glu * up, mask=mask)


def _silu_and_mul_clamp_fp32(
    output: torch.Tensor, input_: torch.Tensor, limit: float
) -> None:
    """Match DeepGEMM's native clamp/activation precision before A2 quant."""
    rows, twice_hidden = input_.shape
    hidden = twice_hidden // 2
    assert twice_hidden == hidden * 2
    assert output.shape == (rows, hidden)
    _silu_and_mul_clamp_fp32_kernel[
        (rows, triton.cdiv(hidden, 256))
    ](
        input_,
        output,
        hidden,
        input_.stride(0),
        output.stride(0),
        limit,
        BLOCK_H=256,
        num_warps=4,
    )


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


@triton.jit
def _desc_build_kernel_exl3fp4(
    eids_ptr, npost_ptr, fslot_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    fpoolb, fslot_bytes, off4_s13, off4_c2, off4_s2,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """v2-perf: TWO w4 desc tables ([0]=w13, [1]=w2) for the FP4 need-pool
    OVER an EXL3 base. Base experts are served by the trellis tier (not
    here), so a pair that is NOT FP4-resident gets m=0 (moe_w4_mm early-EXITs)
    -- no base/miss handling. Slots carry their own scales
    ([fp4_13|sc13|fp4_2|sc2]) read via off4_* (same as the base-cache kernel's
    w4 arm)."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    fslot = tl.load(fslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    m4 = tl.where(live & (fslot >= 0), mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    fs = fpoolb + tl.maximum(fslot, 0) * fslot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    for gi in tl.static_range(2):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c, m = fs, fs + off4_s13, a1, as1, c13, m4
        else:
            b, s, a, as_, c, m = fs + off4_c2, fs + off4_s2, a2, as2, c2, m4
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


def _exl3_fp4_apply_cubin(
    ftier,
    layer_key: int,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    act_limit: float,
) -> torch.Tensor:
    """v2-perf: FP4 need-pool contribution via the production moe_w4_mm cubin.
    Reads the 4-bit pool slots directly (fp8-a32 activations, ~2 kernel
    launches/layer) instead of the torch dequant. Non-resident pairs get m=0
    (served by the EXL3 base). Returns [T, H] in x's dtype. Uses the ORIGINAL
    (unmasked) topk — the desc kernel selects FP4-resident experts via
    slot_table; the deterministic unpermute zeroes the non-resident rows."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    g = _EXL3_FP4_GEOM
    N13, K13, N2, K2 = g["N13"], g["K13"], g["N2"], g["K2"]
    off4_s13, off4_c2, off4_s2 = g["off_s13"], g["off_c2"], g["off_s2"]
    T, H = x.shape
    top_k = topk_ids.shape[1]
    E = ftier.E
    dev = x.device
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    mblock = _BLOCK
    sorted_ids, expert_blocks, num_post = moe_align_block_size(
        topk_ids, mblock, E)
    slots = sorted_ids.numel()
    pairs = slots // mblock
    ws = _workspaces(slots, T, dev, inter=K2, hidden=K13, n_experts=E)
    xq = ws["xq"]
    pad_row = xq.shape[0] - 1
    _quant_a32(x, xq[:T], ws["xs"][:T], _G1)
    valid = sorted_ids < T * top_k
    rows = torch.where(valid, sorted_ids // top_k,
                       torch.full_like(sorted_ids, pad_row))
    torch.index_select(xq.view(torch.uint8), 0, rows,
                       out=ws["a1"][:slots].view(torch.uint8))
    torch.index_select(ws["xs"], 0, rows, out=ws["as1"][:slots])
    d = ws["desc"]
    cap = d.shape[1]
    slot_row = ftier.slot_table[layer_key]
    _desc_build_kernel_exl3fp4[(triton.cdiv(pairs, 256),)](
        expert_blocks, num_post, slot_row, d,
        ws["a1"].data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
        ws["a2"].data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
        ftier.pool.data_ptr(), ftier.slot_bytes, off4_s13, off4_c2, off4_s2,
        K13, (K13 // 32) * 4, 4 * K2, K2, (K2 // 32) * 4, 2 * K13,
        E, pairs, cap * 6, mblock, BLOCK=256)
    _launch("w4", K13, d[0], N13, pairs, stream)
    act = ws["act"][:slots]
    _silu_and_mul_clamp_fp32(act, ws["c13"][:slots], act_limit)
    _quant_a32(act, ws["a2"][:slots], ws["as2"][:slots], _G2)
    _launch("w4", K2, d[1], N2, pairs, stream)
    # only FP4-resident pairs contribute; the rest hold stale c2 rows -> mask
    e_pair = expert_blocks.to(torch.long).clamp_(0, E - 1)
    resident = (slot_row[e_pair] >= 0)
    miss_rows = resident.repeat_interleave(mblock)[:slots]
    n_routes = T * top_k
    inverse = ws["inv_sorted"][:n_routes]
    _invert_sorted_ids_kernel[(triton.cdiv(slots, 256),)](
        sorted_ids, inverse, slots, n_routes, BLOCK=256)
    out = torch.empty((T, H), dtype=x.dtype, device=dev)
    _deterministic_unpermute_kernel[(T, triton.cdiv(H, 256))](
        ws["c2"], topk_weights, inverse, miss_rows, out, H,
        ws["c2"].stride(0), topk_weights.stride(0), topk_weights.stride(1),
        out.stride(0), TOP_K=top_k, HAS_ROW_MASK=True, BLOCK_H=256,
        num_warps=4)
    return out


def _exl3_fp4_apply(
    ftier,
    layer_key: int,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fp4_mask: torch.Tensor,
    act_limit: float,
) -> torch.Tensor:
    """Exact FP4 contribution for the experts masked OUT of the EXL3 base
    (i.e. resident in the need-pool). Eager, correctness-first (v2): dequant
    each masked expert's weights from its pool slot ([fp4_13|sc13|fp4_2|sc2])
    in torch and run the SwiGLU expert, mirroring Exl3BaseTier.forward_topk's
    per-token loop and clamps EXACTLY (parity oracle: tools/test_moe_w2_exl3.py
    block_torch(fp4_expert)). Perf (batched dispatch / cubin) is v2-perf, the
    next backlog item; this path is the correctness plumbing. Returns [m, H]
    f32; base-served (t, j) contribute nothing here (they came from EXL3)."""
    import torch.nn.functional as F
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        dequant_fp4_expert)
    g = _EXL3_FP4_GEOM
    assert g is not None, "moe_w2 EXL3 v2: FP4 slot geometry not staged"
    N13, K13, N2, K2 = g["N13"], g["K13"], g["N2"], g["K2"]
    off_s13, off_c2, off_s2 = g["off_s13"], g["off_c2"], g["off_s2"]
    inter = K2                          # w1/w3 output rows (I); w13 = [2I, H]
    hidden = K13                        # H
    dev = x.device
    slot_row = ftier.slot_table[layer_key]        # [E] int32 (-1 = base tier)
    pool = ftier.pool                             # [n_slots, slot_bytes] u8
    xf = x.float()
    y = torch.zeros(xf.shape[0], hidden, dtype=torch.float, device=dev)
    # SYNC-LIGHT: collect every masked (token, slot) hit with ONE nonzero, sort
    # by expert so each expert's hits are contiguous, and read the unique expert
    # list + counts with a single tolist pair — instead of a per-expert
    # `.nonzero()`/`int(slot)` (each a CPU<->GPU sync -> pipeline stall x 43
    # layers x re-forwards, the measured v2 bottleneck). Still: dequant each
    # expert once (LRU cache), apply batched over its tokens, free per expert.
    tok_all, j_all = fp4_mask.nonzero(as_tuple=True)   # [P] (the one sync to size P)
    if tok_all.numel() == 0:
        return y
    experts = topk_ids[tok_all, j_all]                 # [P] expert ids
    order = torch.argsort(experts)                     # group hits by expert (GPU)
    experts = experts[order]
    tok_all = tok_all[order]
    wsel = topk_weights[tok_all, j_all[order]].float()
    uniq, counts = torch.unique_consecutive(experts, return_counts=True)
    uniq_l = uniq.tolist()
    counts_l = counts.tolist()
    slots_l = None                                     # filled lazily on a cache miss
    off = 0
    for i, (e, c) in enumerate(zip(uniq_l, counts_l)):
        ti = tok_all[off:off + c]
        ws = wsel[off:off + c]
        off += c
        key = (layer_key, e)
        w = _EXL3_FP4_WCACHE.get(key)
        if w is None:
            if slots_l is None:
                slots_l = slot_row[uniq].tolist()      # one sync for all misses
            row = pool[slots_l[i]]
            w13 = dequant_fp4_expert(row[:off_s13], row[off_s13:off_c2],
                                     N13, K13).to(torch.bfloat16)
            w2 = dequant_fp4_expert(row[off_c2:off_s2], row[off_s2:],
                                    N2, K2).to(torch.bfloat16)
            w = (w13[:inter].contiguous(), w13[inter:].contiguous(), w2)
            _EXL3_FP4_WCACHE[key] = w
            if len(_EXL3_FP4_WCACHE) > _EXL3_FP4_WCACHE_MAX:
                _EXL3_FP4_WCACHE.pop(next(iter(_EXL3_FP4_WCACHE)))  # evict oldest
        else:
            _EXL3_FP4_WCACHE.pop(key)                  # LRU touch: move to newest
            _EXL3_FP4_WCACHE[key] = w
        w1b, w3b, w2b = w
        # mirror forward_topk: 16-bit GEMMs, f32 activation clamp/silu
        xt = xf[ti].to(torch.bfloat16)                 # [c, H]
        g = (xt @ w1b.t()).float().clamp(max=act_limit)
        u = (xt @ w3b.t()).float().clamp(min=-act_limit, max=act_limit)
        contrib = (F.silu(g) * u).to(torch.bfloat16) @ w2b.t()   # [c, H] bf16
        y.index_add_(0, ti, ws.unsqueeze(1) * contrib.float())
    return y


def _exl3_fp4_validate(ftier, layer_key, x, ids, wts, fp4_mask, act_limit,
                       y_cubin) -> None:
    """Certify the cubin FP4 apply against the torch apply (the validated
    oracle) on the first few live forwards. A non-zero rel-err is EXPECTED
    (the cubin path uses fp8-a32 activations, the torch path bf16), so this
    is a band check (~0.02-0.06), not bit-equality."""
    global _exl3_fp4_val_n
    if _exl3_fp4_val_n >= 60:
        return
    _exl3_fp4_val_n += 1
    y_torch = _exl3_fp4_apply(ftier, layer_key, x, ids, wts, fp4_mask,
                              act_limit)
    yc = y_cubin.float()
    num = (yc - y_torch).square().mean().sqrt()
    den = y_torch.square().mean().sqrt().clamp_min(1e-8)
    logger.info("moe_w2 EXL3 v2 FP4-apply VALIDATE layer %d: rel-err %.4f "
                "(cubin fp8-a32 vs torch bf16; band ~0.02-0.06)",
                layer_key, float(num / den))


def _exl3_delta_apply(tier, lidx: int, ftier, layer_key: int,
                      x: torch.Tensor, topk_ids: torch.Tensor,
                      topk_weights: torch.Tensor,
                      dmask: torch.Tensor) -> torch.Tensor:
    """Δ-pool contribution: full base+Δ block for pool-resident pairs via
    Exl3BaseTier.forward_topk_dual (per-projection second mgemm over slot
    pointer tables). Capture-safe (pointer tables are elementwise int64
    arithmetic on slot_table; fixed shapes; no syncs)."""
    from vllm.model_executor.layers.quantization.utils.moe_w2_exl3 import (
        delta_ptr_tables)
    g = _EXL3_DELTA_GEOM
    assert g is not None, "moe_w2 EXL3 Δ-pool: geometry not staged"
    tabs = delta_ptr_tables(ftier.pool.data_ptr(), ftier.slot_bytes,
                            ftier.slot_table[layer_key], g)
    return tier.forward_topk_dual(lidx, x, topk_ids, topk_weights, dmask,
                                  tabs, dk13=g["dk13"], dk2=g["dk2"],
                                  dmults=g["mults"])


_HAD128 = None


def _exl3_had128(dev):
    global _HAD128
    if _HAD128 is None or _HAD128.device != dev:
        h = torch.ones(1, 1, dtype=torch.float64, device=dev)
        b = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64,
                         device=dev)
        for _ in range(7):
            h = torch.kron(b, h)
        _HAD128 = (h / 128.0 ** 0.5).float()
    return _HAD128


def _exl3_delta_validate(tier, lidx: int, ftier, layer_key: int,
                         x: torch.Tensor, ids: torch.Tensor,
                         wts: torch.Tensor, dmask: torch.Tensor,
                         y_dual: torch.Tensor) -> None:
    """Certify the dual-stream mgemm apply against a torch reference:
    dequantize base+Δ weights (ext.reconstruct + blockwise 128-Hadamards +
    suh/svh) for each pooled pair and run the SwiGLU block in fp32. Band
    check ~0.02 (fp16 mgemm chain vs fp32 torch)."""
    global _exl3_delta_val_n
    if _exl3_delta_val_n >= 12 or not bool(dmask.any()):
        return
    _exl3_delta_val_n += 1
    import torch.nn.functional as F
    g = _EXL3_DELTA_GEOM
    dev = x.device
    H128 = _exl3_had128(dev)

    def dereg(w_inner, suh, svh):
        kdim, ndim = w_inner.shape
        w = w_inner.float()
        w = (H128 @ w.view(kdim // 128, 128, ndim)).reshape(kdim, ndim)
        w = w * suh.float().unsqueeze(1)
        w = (w.view(kdim, ndim // 128, 128) @ H128).reshape(kdim, ndim)
        return w * svh.float().unsqueeze(0)

    def dq(trellis, suh, svh, K, mul1=False):
        kdim, ndim = trellis.shape[0] * 16, trellis.shape[1] * 16
        w = torch.empty(kdim, ndim, dtype=torch.half, device=dev)
        tier.ext.reconstruct(w, trellis.contiguous(), K, False, mul1)
        return dereg(w, suh, svh)

    def slot_part(row, proj, kdim, ndim):
        dkp = g["dk2"] if proj == "d" else g["dk13"]
        otr, osuh, osvh = g["offs"][proj]
        trb = kdim // 16 * (ndim // 16) * dkp * 32
        tr = row[otr:otr + trb].view(torch.int16) \
            .view(kdim // 16, ndim // 16, 16 * dkp)
        suh = row[osuh:osuh + kdim * 2].view(torch.half)
        svh = row[osvh:osvh + ndim * 2].view(torch.half)
        return dq(tr, suh, svh, dkp, mul1=True)  # pack v3 delta = MUL1

    slot_row = ftier.slot_table[layer_key]
    tok, jj = dmask.nonzero(as_tuple=True)
    y_ref = torch.zeros_like(y_dual)
    Hd, Id = tier.H, tier.I
    for t_i, j_i in zip(tok.tolist(), jj.tolist()):
        e = int(ids[t_i, j_i])
        base = {p: dq(tier._store[f"l{lidx}.e{e}.{p}.k{k}.trellis"],
                      tier._store[f"l{lidx}.e{e}.{p}.k{k}.suh"],
                      tier._store[f"l{lidx}.e{e}.{p}.k{k}.svh"], k)
                for p, k in (("w1", tier.w13_k), ("w3", tier.w13_k),
                             ("w2", tier.w2_k))}
        row = ftier.pool[int(slot_row[e])]
        d1 = slot_part(row, "g", Hd, Id)
        d3 = slot_part(row, "u", Hd, Id)
        d2 = slot_part(row, "d", Id, Hd)
        xt = x[t_i].float()
        gg = (xt @ (base["w1"] + d1)).clamp(max=tier.act_limit)
        uu = (xt @ (base["w3"] + d3)).clamp(min=-tier.act_limit,
                                            max=tier.act_limit)
        y_ref[t_i] += float(wts[t_i, j_i]) * ((F.silu(gg) * uu)
                                              @ (base["w2"] + d2))
    num = (y_dual.float() - y_ref).square().mean().sqrt()
    den = y_ref.square().mean().sqrt().clamp_min(1e-8)
    logger.info("moe_w2 EXL3 Δ-pool VALIDATE layer %d: rel-err %.4f "
                "(dual mgemm fp16 vs torch fp32 reference; band ~0.02)",
                layer_key, float(num / den))


def _exl3_prefill_ensure_ok(x: torch.Tensor) -> bool:
    """True when this call may run the synchronous prefill-ensure fetch:
    an ensure-enabled, prefill-sized call on a REAL batch.

    Dummy batches must never promote:
    - the load-time KV-PROFILING forward would fill the pool from the
      dummy prompt's junk routing and (on the eager arm) retain dequant
      state, inflating the measured peak -> KV planning goes negative
      (the v2 boot regression that forced PREFILL_FP4=0);
    - cudagraph warmup/capture must not bake the host-side fetch.
    Both run with forward_context.attn_metadata=None, which is the same
    profile-run marker the MLA/mamba layers key on. Outside any forward
    context (tools/tests) ensure stays allowed."""
    if not (_PREFILL_FP4 and _PREFILL_FP4_ENSURE):
        return False
    if x.shape[0] <= _PREFILL_T:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    try:
        from vllm.forward_context import get_forward_context
        if get_forward_context().attn_metadata is None:
            return False
    except (ImportError, AssertionError):
        pass
    return True


_exl3_ptab_ready: list = []   # one-shot init sentinel (glue fix [V] 08-05)


def _exl3_ensure_ptabs(ftier) -> None:
    """One-time wiring of the decode-wave persistent pointer tables: build
    each EXL3 tier's ptab stack (base rows static, Δ rows from the CURRENT
    slot_table) and register the refresh on the delta tier's mutate hooks.
    Runs on the first eager forward through the wave route (vLLM's profile/
    dummy run precedes any capture); a capture asserting here means that
    ordering broke."""
    if _exl3_ptab_ready:
        return
    assert not torch.cuda.is_current_stream_capturing(), \
        "moe_w2 EXL3 ptab init reached capture before any eager forward"
    g = _EXL3_DELTA_GEOM
    assert g is not None, "moe_w2 EXL3 ptab init: geometry not staged"
    by_tier: dict[int, tuple] = {}
    for lk, (tier, li) in _EXL3_TIERS.items():
        by_tier.setdefault(id(tier), (tier, []))[1].append((lk, li))
    for tier, pairs in by_tier.values():
        tier.init_ptabs(ftier, g, pairs)
    logger.info("moe_w2 EXL3 decode-wave ptab stack ready "
                "(%d tier(s), %d layers, refresh-on-mutation)",
                len(by_tier), len(_EXL3_TIERS))
    _exl3_ptab_ready.append(1)


def _exl3_forward(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    """EXL3 base tier forward: the full routed-expert block from the pack
    via exl3_mgemm (per-token grouped dispatch, weighted reduction
    in-kernel). Mirrors the plane path's return contract exactly:
    routed-only [T, H] in x's dtype (shared experts are orchestrated by
    the MoE runner outside, same as every non-modular apply); _apply_topp
    runs first like the plane path so the env knob keeps its semantics.

    v2 (FP4 need-pool on): experts currently resident in the pool
    (slot_table >= 0) are MASKED out of the EXL3 call and their exact FP4
    contribution is added back — plane replacement, additive over experts
    (parity-proven split additivity 0.0006). The confidence gate + the
    runner's promote->replay loop drive residency between forwards, all
    format-agnostic; this forward only reads slot_table and marks `seen`."""
    if not _EXL3_CUDAGRAPH:
        # eager path: a capture would bake one token set into the host loop.
        assert not torch.cuda.is_current_stream_capturing(), (
            "moe_w2 EXL3 base is eager-only; serve with --enforce-eager or "
            "VLLM_MOE_W2_EXL3_CUDAGRAPH=1 for the capturable path")
    topk_weights, topk_ids = _apply_topp(topk_weights, topk_ids)
    tier, lidx = _EXL3_TIERS[layer_key]
    if _HCAP_ON:
        # 9b step-1 (telemetry-only, default-off): in-graph copy of this
        # layer's MoE input (the residual-stream h_l the offline probes
        # train on). Fixed-shape copy into a persistent buffer — the
        # mark_seen idiom; the runner snapshots it at DECISION time on
        # dumped steps (pre-FP-loop, so replays never overwrite the
        # signal the gate actually saw). Row 0 only (C=1 decode; spec
        # verify rows are a step-2 concern).
        _hcap_buf(layer_key, x).copy_(x[0], non_blocking=True)
    ids = topk_ids.long()
    wts = topk_weights.float()
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    ftier = moe_w2_delta._TIER
    if ftier is None:
        # v1 path: bare EXL3 base, no need-pool / gate
        return tier.forward_topk(lidx, x, ids, wts).to(x.dtype)
    # record this step's routed experts so the gate's force_promote sees the
    # scatter (the background manager never promotes under `need`).
    moe_w2_delta.mark_seen(ftier.seen[layer_key], ids.view(-1))
    if moe_w2_delta.STEP_W:
        # P10: routing-weight scatter for cap-bound promotion ordering
        moe_w2_delta.mark_seen_w(ftier.seen_w[layer_key], ids.view(-1),
                                 wts.view(-1))
    # v2 prefill-ensure (both arms): make the chunk's routed set FP4-resident
    # BEFORE the slot_table read so THIS chunk already serves them at FP4
    # (the 2-bit prefill idiom). Real eager prefills only — never the
    # KV-profiling dummy run or capture (see _exl3_prefill_ensure_ok).
    if _exl3_prefill_ensure_ok(x):
        ftier.ensure_resident(layer_key, ids.view(-1))
    # Track 2.5 decode wave, graph AND eager (glue fix [V] 08-05): M=1 decode
    # via the flat wave kernel over the PERSISTENT ptab stack — no per-step
    # slot_table read, Δ ptr tables, dmask or cats (the base-only zero-svh
    # mask for non-pooled experts is baked into the ptab at pool-MUTATION
    # time by the delta tier's mutate hook). Same base/base+Δ block as
    # forward_topk_unified; capture-safe, no host syncs, one gather/layer.
    if (_EXL3_WAVE and x.shape[0] == 1 and _EXL3_W2K == 1
            and _EXL3_DELTA_PACK and _EXL3_DELTA_GEOM is not None
            and _EXL3_DELTA_GEOM["dk13"] == 2
            and _EXL3_DELTA_GEOM["dk2"] == 3
            and (_EXL3_DELTA_UNIFIED or not _EXL3_CUDAGRAPH)):
        _exl3_ensure_ptabs(ftier)
        if not _exl3_wave_seen:
            logger.info("moe_w2 EXL3 decode-wave route ACTIVE "
                        "(%sM=1, base 2,2,1 cb0 + Δ 2,2,3 cb2, ptab)",
                        "" if _EXL3_CUDAGRAPH else "eager, ")
            _exl3_wave_seen.append(1)
        return tier.forward_topk_wave(lidx, x, ids, wts).to(x.dtype)
    # M8 CAPTURE-SAFE (charter CS [E] 2026-08-10): M ∈ [2, 8] decode on
    # the M=8-native wave INSIDE cudagraphs — fixed G=T ext calls per
    # captured size, groups built on device (no host syncs, static
    # shapes), pointers gathered into persistent buffers, empty groups
    # exited by the m8g canon guard (~4.5 us). Requires the GUARDED
    # cubin set (ext_wave_m8_guarded); plain m8 cubins keep the eager
    # route below only. Pre-capture eager warmups take this same route
    # (warms the dlopen/cuModuleLoad before capture); the unified path
    # no longer serves captured T ∈ [2, 8] decode shapes, so the
    # autotune-inside-capture incident ([K] 01:5x) cannot re-arm — its
    # prefill/piecewise shapes (T > 8) still warm eagerly as before.
    if (_EXL3_WAVE_M8 and 2 <= x.shape[0] <= 8 and _EXL3_W2K == 1
            and _EXL3_DELTA_PACK and _EXL3_DELTA_GEOM is not None
            and _EXL3_DELTA_GEOM["dk13"] == 2
            and _EXL3_DELTA_GEOM["dk2"] == 3
            and _EXL3_CUDAGRAPH
            and tier.ext_wave_m8_guarded()):
        _exl3_ensure_ptabs(ftier)
        y, n_groups = tier.forward_topk_wave_m8_graph(lidx, x, ids, wts)
        if not _exl3_wave_m8g_seen:
            logger.info("moe_w2 EXL3 decode-wave-M8 route ACTIVE "
                        "(graph, fixed groups=%d)", n_groups)
            _exl3_wave_m8g_seen.append(1)
        return y.to(x.dtype)
    # M8 charter F2 ([K] 2026-08-09, §2bis): M ∈ [2, 8] decode on the
    # M=8-native wave — union of the step's routed experts partitioned
    # into groups of 6, ext launched once per group, atomic-accumulating
    # into one zeroed [T, H] fp32 out (forward_topk_wave_m8 owns the
    # zeroing). Geometry gate == the M=1 wave route (pack-v3 serving
    # pair over the same ptab stack). The grouping is data-dependent
    # HOST logic, so this route is EAGER-ONLY — and not merely guarded
    # against capture: under _EXL3_CUDAGRAPH the pre-capture eager
    # warmups would take THIS route and starve the unified path of its
    # first-call autotune, which then fires inside the capture ("GPU
    # assert: operation not permitted when stream is capturing",
    # coop_autotune.cu — measured on the F3 boot, 2026-08-10). With
    # cudagraphs on, T ∈ [2, 8] is served by the capture-safe route
    # above (guarded cubins) or the capture-safe fallbacks below; an
    # ext without the m8 launcher (or unresolved cubins) degrades the
    # same way.
    if (_EXL3_WAVE_M8 and 2 <= x.shape[0] <= 8 and _EXL3_W2K == 1
            and _EXL3_DELTA_PACK and _EXL3_DELTA_GEOM is not None
            and _EXL3_DELTA_GEOM["dk13"] == 2
            and _EXL3_DELTA_GEOM["dk2"] == 3
            and not _EXL3_CUDAGRAPH
            and not torch.cuda.is_current_stream_capturing()
            and tier.ext_wave_m8_available()):
        _exl3_ensure_ptabs(ftier)
        y, n_groups = tier.forward_topk_wave_m8(lidx, x, ids, wts)
        if not _exl3_wave_m8_seen:
            logger.info("moe_w2 EXL3 decode-wave-M8 route ACTIVE "
                        "(groups=%d)", n_groups)
            _exl3_wave_m8_seen.append(1)
        return y.to(x.dtype)
    slot_row = ftier.slot_table[layer_key]        # [E] int32 (-1 = base tier)
    fp4_mask = slot_row[ids] >= 0                 # [m, k] bool
    if _EXL3_CUDAGRAPH:
        # CAPTURE-SAFE (Stage B): fixed-shape, sync-free. Serve the base for
        # ALL experts but zero the pool-resident ones via masked weights (not
        # shape compaction); add the pool-resident experts via the kernel
        # apply (Δ dual-stream mgemm in Δ-pool mode, moe_w4_mm cubin in FP4
        # mode). The gate's out-of-graph replay re-runs this captured graph
        # after promotions, exactly like the 2-bit path.
        if _EXL3_DELTA_PACK and _EXL3_DELTA_UNIFIED:
            # P6-perf: one unified pass serves base AND Δ (6 mgemm/token).
            from vllm.model_executor.layers.quantization.utils \
                .moe_w2_exl3 import delta_ptr_tables
            g = _EXL3_DELTA_GEOM
            tabs = delta_ptr_tables(ftier.pool.data_ptr(), ftier.slot_bytes,
                                    ftier.slot_table[layer_key], g)
            y = tier.forward_topk_unified(lidx, x, ids, wts, fp4_mask,
                                          tabs, dk13=g["dk13"],
                                          dk2=g["dk2"], dmults=g["mults"])
            return y.to(x.dtype)
        wts_base = wts.masked_fill(fp4_mask, 0.0)
        y = tier.forward_topk(lidx, x, ids, wts_base)
        if _EXL3_DELTA_PACK:
            yd = _exl3_delta_apply(tier, lidx, ftier, layer_key, x, ids,
                                   wts, fp4_mask)
            if (_EXL3_DELTA_VALIDATE
                    and not torch.cuda.is_current_stream_capturing()
                    and bool(fp4_mask.any())):
                _exl3_delta_validate(tier, lidx, ftier, layer_key, x, ids,
                                     wts, fp4_mask, yd)
            y = y + yd.float()
        else:
            y = y + _exl3_fp4_apply_cubin(
                ftier, layer_key, x, ids, wts, tier.act_limit).float()
        return y.to(x.dtype)
    # ---- eager path (v2): compaction + optional cubin/torch apply ----
    # (prefill-ensure already ran above; the eager decode-wave mirror is the
    # hoisted ptab branch before the fp4_mask read)
    y = tier.forward_topk(lidx, x, ids, wts, fp4_mask=fp4_mask)
    if _EXL3_DELTA_PACK:
        if bool(fp4_mask.any()):
            yd = _exl3_delta_apply(tier, lidx, ftier, layer_key, x, ids,
                                   wts, fp4_mask)
            if _EXL3_DELTA_VALIDATE:
                _exl3_delta_validate(tier, lidx, ftier, layer_key, x, ids,
                                     wts, fp4_mask, yd)
            y = y + yd.float()
    elif _EXL3_FP4_CUBIN:
        # v2-perf: FP4-resident experts via the moe_w4_mm cubin (base already
        # served by forward_topk with fp4_mask). Non-resident pairs contribute
        # zero (m=0 in the desc); no per-expert torch dequant/sync.
        yfp4 = _exl3_fp4_apply_cubin(ftier, layer_key, x, ids, wts,
                                     tier.act_limit)
        if _EXL3_FP4_VALIDATE and bool(fp4_mask.any()):
            _exl3_fp4_validate(ftier, layer_key, x, ids, wts, fp4_mask,
                               tier.act_limit, yfp4)
        y = y + yfp4.float()
    elif bool(fp4_mask.any()):
        y = y + _exl3_fp4_apply(ftier, layer_key, x, ids, wts, fp4_mask,
                                tier.act_limit)
    return y.to(x.dtype)


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
    if _EXL3_BASE:
        return _exl3_forward(x, topk_weights, topk_ids, layer_key)

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
    if st.get("mapped_host", False):
        moe_w2_mapped_host.note_real_kernel_dispatch(
            layer_key, T, prefill,
            torch.cuda.is_current_stream_capturing())
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
    elif _SWIGLU_CLAMP_FP32:
        _silu_and_mul_clamp_fp32(
            act, ws["c13"][:slots], st["swiglu_limit"])
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
    global _cutoff_cache, _resident_fit_checked, _exl3_config_checked
    global _EXL3_FP4_GEOM
    global _fast_loader, _fast_probe_s, _fast_probe_layers
    # CUDA graphs are already destroyed by gpu_worker before this hook. Drop
    # the last tensor views before cudaFreeHost releases their mapped backing.
    _LAYERS.clear()
    moe_w2_mapped_host.shutdown()
    _WS.clear()
    _EXL3_TIERS.clear()
    _EXL3_FP4_GEOM = None
    _EXL3_FP4_WCACHE.clear()
    _exl3_config_checked = False
    _n_created = 0
    _skip_logged = False
    _stream_logged = False
    _cutoff_cache = None
    _resident_fit_checked = False
    if _fast_loader is not None:
        _fast_loader.close()
    _fast_loader = None
    _fast_probe_s = 0.0
    _fast_probe_layers = 0
    ready.cache_clear()
