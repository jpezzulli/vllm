# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router-lookahead measurement (LOOKA) + prefetch (PILOT) for the moe_w2
BASE cache.

Motivation (measured on colibri, the CPU streaming engine for GLM-5.2):
next-layer expert routing is predictable AHEAD of the layer itself —
applying layer L+1's router to layer L's hidden state recalled 71.6% of the
true top-8, vs 41.3% for "same experts as the previous token". The shipped
draft-affinity prefetcher (VLLM_MOE_W2_PREFETCH) is the 41.3%-class
predictor (token-identity affinity); this module measures — and optionally
acts on — the stronger router-lookahead signal in THIS stack:

  LOOKA (VLLM_MOE_W2_LOOKA=1): counters only, zero behaviour change.
    At each MoE layer's decode forward, (a) score the previous decode
    step's routing for the same layer (predictor [0], the affinity-class
    baseline) and (b) score the prediction made one layer EARLIER by
    applying THIS layer's router to the PREVIOUS layer's expert input
    (predictor [1], router-lookahead). Recalls are accumulated in-graph
    (GPU counters, no syncs) and reported on the KPI line.

  PILOT (VLLM_MOE_W2_PILOT=1, implies the LOOKA machinery): at layer L the
    predicted top-K for layer L+1 is also written to an in-graph pilot log;
    the tier's manager thread consumes it every tick (5 ms against a
    30-60 ms step) and prefetches the predicted NON-RESIDENT experts on the
    side stream — so by the time the step's misses are counted, part of the
    would-be fetch burst is already on the GPU, and the mandatory replay
    (or the next step) finds them resident. Mispredictions cost one cold
    slot each and decay away; the prefetch never emergency-evicts.

The predictor input is layer L's MoE input x_L (the post-attention-LN
hidden) fed to layer L+1's router — one residual short of the true router
input (x_L lacks L's expert contribution). That is exactly the point where
the prediction is available a full layer ahead of the fetch it hides; the
LOOKA counters price that approximation honestly before PILOT is trusted.

Everything is CUDA-graph-safe by construction: the in-graph half touches
only persistent buffers via tensor ops (matmul + sigmoid + topk + compares),
python branching happens at capture time, and the host half (arming, KPI
reads, PILOT consumption) runs on the manager/runner threads outside
capture.
"""

import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOOKA = os.getenv("VLLM_MOE_W2_LOOKA", "0") == "1"
_PILOT = os.getenv("VLLM_MOE_W2_PILOT", "0") == "1"
# top-K predictions kept per position: the head of the router ranking is
# more reliable than the tail (colibri's PILOT_K) — and it bounds both the
# in-graph topk and the per-tick prefetch fan-out.
_PILOT_K = max(1, min(16, int(os.getenv("VLLM_MOE_W2_PILOT_K", "8"))))
# max experts fetched per manager tick from the pilot log
_PILOT_CAP = int(os.getenv("VLLM_MOE_W2_PILOT_CAP", "32"))
# pilot fetches may only displace slots idle for this many manager ticks
# (5 ms each; 200 ≈ 1 s). The per-step hot set is touched every few ticks
# and must never be churned by speculative prefetch.
_PILOT_COLD_TICKS = int(os.getenv("VLLM_MOE_W2_PILOT_COLD_TICKS", "200"))
# pilot consumes only while the pool is COLD (replay EMA %, from kpi_step):
# warm pools are covered by the step's own miss restore and pilot fetches
# would only churn idle slots. 0 = always consume.
_PILOT_MIN_REPLAY = float(os.getenv("VLLM_MOE_W2_PILOT_MIN_REPLAY", "30"))
_TCAP = int(os.getenv("VLLM_MOE_W2_PREFETCH_TCAP", "8"))

_armed = False
_gate_w: dict[int, torch.Tensor] = {}     # layer_key -> [E, H] (live param)
_gate_b: dict[int, torch.Tensor] = {}     # layer_key -> [E] f32 or None
_pred_buf: torch.Tensor | None = None     # [TCAP, PILOT_K] i32 (key k-1 -> k)
_pilot_log: torch.Tensor | None = None    # [n_keys, TCAP, PILOT_K] i32
_pilot_host: torch.Tensor | None = None   # pinned mirror for the tick D2H
# GPU counters, written in-graph: [0]=prev-step hits, [1]=lookahead hits
_hit: torch.Tensor | None = None          # i64 [2]
_tot: torch.Tensor | None = None          # i64 [2]


def enabled() -> bool:
    return _armed and (_LOOKA or _PILOT)


def pilot_enabled() -> bool:
    return _armed and _PILOT


def wants_route_log() -> bool:
    """The predictor-[0] baseline reads the previous step's routing from the
    tier's route_log — arm it even when the affinity prefetcher is off."""
    return _LOOKA or _PILOT


def arm(model, n_keys: int, dev) -> None:
    """Collect the live router (mlp.gate) weights per moe_w2 layer_key and
    allocate the persistent in-graph buffers. Called once from the runner
    after weight load (before any cudagraph capture). Never raises."""
    global _armed, _pred_buf, _pilot_log, _hit, _tot
    if not (_LOOKA or _PILOT) or _armed:
        return
    try:
        from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
        # transformer layer idx -> moe_w2 layer_key (dense layers have none)
        tl_to_key = {
            st.get("tl_idx"): key
            for key, st in moe_w2_cubit._LAYERS.items()
            if st.get("tl_idx") is not None
        }
        pat = re.compile(r"\.layers\.(\d+)\.mlp\.gate\.(weight|"
                         r"e_score_correction_bias)$")
        for name, p in model.named_parameters():
            m = pat.search(name)
            if m is None:
                continue
            key = tl_to_key.get(int(m.group(1)))
            if key is None:
                continue
            if m.group(2) == "weight":
                _gate_w[key] = p.data          # live reference, no copy
            else:
                _gate_b[key] = p.data.float()
        if not _gate_w:
            logger.warning("moe_w2 LOOKA: no mlp.gate weights found — "
                           "disabled (router naming mismatch?)")
            return
        _pred_buf = torch.zeros(_TCAP, _PILOT_K, dtype=torch.int32,
                                device=dev)
        if _PILOT:
            _pilot_log = torch.full((n_keys, _TCAP, _PILOT_K), -1,
                                    dtype=torch.int32, device=dev)
        _hit = torch.zeros(2, dtype=torch.int64, device=dev)
        _tot = torch.zeros(2, dtype=torch.int64, device=dev)
        _armed = True
        logger.info(
            "moe_w2 %s armed: %d routers, pred top-%d, T_cap %d%s",
            "PILOT (router-lookahead prefetch)" if _PILOT else
            "LOOKA (router-lookahead counters)",
            len(_gate_w), _PILOT_K, _TCAP,
            f", pilot cap {_PILOT_CAP}/tick" if _PILOT else "")
    except Exception as e:  # noqa: BLE001 - measurement must never kill boot
        logger.warning("moe_w2 LOOKA arm failed: %s", e)


def record(layer_key: int, x: torch.Tensor, topk_ids: torch.Tensor,
           route_log: torch.Tensor | None) -> None:
    """In-graph hook, called from the moe_w2 decode forward of every BASE
    layer BEFORE the route_log is overwritten. Pure tensor ops on persistent
    buffers (capture-safe); `layer_key` is a python int, so the branches
    below specialize per layer at capture time."""
    if not _armed:
        return
    T = x.shape[0]
    if T > _TCAP:
        return
    k_true = topk_ids.shape[1]
    true = topk_ids[:, :k_true].int()
    # [0] previous decode step, same layer (the affinity-class baseline).
    # route_log still holds LAST step's ids for this layer here.
    if route_log is not None:
        prev = route_log[layer_key, :T, :k_true]
        m0 = (true.unsqueeze(2) == prev.unsqueeze(1)).any(dim=2)
        _hit[0] += m0.sum()
        _tot[0] += true.numel()
    # [1] router-lookahead: the prediction targeting THIS layer, written
    # into _pred_buf by the previous layer's record() call (same step; the
    # keys run in order inside one forward). Exists iff this layer's gate
    # was collected and there IS a previous base layer.
    if layer_key > 0 and layer_key in _gate_w:
        pred = _pred_buf[:T]
        m1 = (true.unsqueeze(2) == pred.unsqueeze(1)).any(dim=2)
        _hit[1] += m1.sum()
        _tot[1] += true.numel()
    # predict layer_key+1's routing from THIS layer's expert input
    w = _gate_w.get(layer_key + 1)
    if w is not None:
        logits = (x.to(w.dtype) @ w.t()).float()
        scores = torch.sigmoid(logits)
        b = _gate_b.get(layer_key + 1)
        if b is not None:
            scores = scores + b
        pred_ids = torch.topk(scores, _PILOT_K, dim=-1).indices.int()
        _pred_buf[:T].copy_(pred_ids)
        if _pilot_log is not None:
            _pilot_log[layer_key + 1, :T].copy_(pred_ids)


def tick_consume(tier) -> int:
    """PILOT host half, called from the tier's manager tick (outside
    capture): read the pilot log, prefetch predicted non-resident experts.
    The log is tiny (n_keys * TCAP * K * 4 B).

    Hard-won rules from the live bring-up (each violation halved decode):
      - CONSUME-ONCE: the log is invalidated after the read — otherwise
        every 5 ms tick re-fetches the same stale predictions forever
        (29k fetches/500 steps measured). The fill_ races benignly with
        the in-flight step's writes (a lost prediction = one tick delay).
      - SIDE-STREAM D2H: `.to('cpu')` on the manager thread runs on the
        DEFAULT stream — synchronizing it every tick stalls the in-flight
        decode graphs (measured: flat ~20 tok/s regardless of pool warmth).
        Copy through the tier's side stream into a pinned buffer instead.
      - COLD PHASE ONLY: on a WARM pool the step's own miss restore covers
        the working set and pilot fetches only churn idle slots for ~zero
        upside; consume only while the replay EMA says the pool is cold.
      - NEVER CHURN THE HOT SET: predictions may only displace slots idle
        >= _PILOT_COLD_TICKS. colibri's PILOT was an OS readahead HINT
        with zero eviction cost; ours takes a slot, so it must only take
        genuinely idle ones."""
    global _pilot_host
    if _pilot_log is None:
        return 0
    if 100.0 * tier._replay_ema < _PILOT_MIN_REPLAY:
        return 0
    if _pilot_host is None:
        _pilot_host = torch.empty_like(_pilot_log, device="cpu",
                                       pin_memory=True)
    with torch.cuda.stream(tier._stream):
        _pilot_host.copy_(_pilot_log, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(tier._stream)
        _pilot_log.fill_(-1)
    ev.synchronize()
    flat = _pilot_host.flatten()
    valid = (flat >= 0) & (flat < tier.E)
    if not bool(valid.any()):
        return 0
    n_keys = _pilot_host.shape[0]
    per_layer = _pilot_host.shape[1] * _pilot_host.shape[2]
    li_all = (torch.arange(n_keys, dtype=torch.int64)
              .repeat_interleave(per_layer))
    keys = torch.unique(li_all[valid] * tier.E + flat[valid].long())
    pairs: list[tuple[int, int]] = []
    for k in keys.tolist():
        li, e = divmod(k, tier.E)
        if int(tier._mirror[li, e]) < 0 and li in tier._store:
            pairs.append((li, e))
            if len(pairs) >= _PILOT_CAP:
                break
    if not pairs:
        return 0
    return tier.prefetch_pairs(pairs, cold_ticks=_PILOT_COLD_TICKS)


def kpi_summary() -> str:
    """Recall summary for the KPI line (host thread; one small D2H)."""
    if not _armed or _tot is None:
        return ""
    tot = _tot.tolist()
    hit = _hit.tolist()
    if tot[0] == 0 and tot[1] == 0:
        return ""
    r0 = 100.0 * hit[0] / max(tot[0], 1)
    r1 = 100.0 * hit[1] / max(tot[1], 1)
    return (f"; LOOKA recall: prev-step {r0:.1f}% / lookahead {r1:.1f}% "
            f"(top-{_PILOT_K}, n={tot[1]})")
