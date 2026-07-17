# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confidence-gated FP4 re-forward for the 2-bit MoE path (directive 2 / Step B).

When the 2-bit base emits a LOW-CONFIDENCE decode token, this gate re-runs the
step with the token's routed experts pulled up to FP4 (via the delta tier's
`force_promote`) and re-decides. Offline validation on a coding corpus
(gate_validate.py) showed that gating on `max_prob <= 0.67` (~30% of tokens)
recovers ~90% of the 2-bit->FP4 top-1 agreement gap and ~61% of the PPL gap;
`max_prob` is the cleanest signal (matches AUROC 0.916).

This module is the *decision + orchestration* half (pure, env-gated, no graph
surgery). The *re-forward* itself is one extra CUDA-graph replay driven by the
model runner, which reads the updated `slot_table` and recomputes the promoted
experts at FP4. Everything is OFF unless `VLLM_MOE_W2_GATE=1`, so the prod
serving path is byte-for-byte unchanged by default.

Why the orchestration is out-of-graph (see CONFIDENCE_GATE_NEXT_SESSION.md):
the trigger (`max_prob` of THIS step's logits) is a runtime branch on a GPU
value, the forced promotion is synchronous + variable-size, and the re-run is a
2nd forward — none of which fit the captured one-graph-per-step cadence. The
re-forward CAN be a graph replay; only steps (a) read confidence, (b) force
promote, (c) trigger the replay are eager.

Env knobs:
  VLLM_MOE_W2_GATE         0 (default) | 1     master switch
  VLLM_MOE_W2_GATE_SIGNAL  max_prob (default) | margin
  VLLM_MOE_W2_GATE_TAU     fire if signal <= TAU. Default 0.60 for max_prob, 1.5
                           nats for margin. Pure quality<->latency knob. At 0.60
                           (measured, coding): fires ~16% of steps, precision ~46%
                           (FP4 differs from 2-bit there), 4.2x lift over the 10.8%
                           base disagreement, ~68% recall -- the efficiency knee
                           before added re-runs go mostly redundant. Raise toward
                           0.70-0.80 for more recall once a functional eval confirms
                           the FP4 upgrades are correct; lower to 0.50 if marginal.
  VLLM_MOE_W2_GATE_MAX_PROMOTE  cap experts force-promoted per fired step
                                (default 64; 0 = unlimited). The cap is a
                                STABILIZER, not just a latency bound —
                                measured both ways on DS4 (2026-07-13):
                                - uncapped, LONG-form generation (GPQA
                                  Diamond, drifting working sets): pool
                                  churns wholesale on every fire and the
                                  FP4 set mutates mid-answer — 60.6% vs
                                  70.2% capped (McNemar p=0.002); small
                                  pools also blow up short-form token
                                  length (+110% on an 85-slot pool).
                                - capped 64, SHORT-form (GSM8K): capped
                                  and uncapped tie at big pools (97.0 vs
                                  97.5, n.s.); capped never measured
                                  worse than uncapped end-to-end.
                                Incremental promotion + need-ranking
                                accumulates the true hard core across
                                fires; promotions persist. Raise/0 only
                                for short-form serving with a pool >= the
                                per-step routed set, where fires then
                                cover their whole step (and see the
                                GLM-5.2 collapse note: unlimited fires
                                measured 200-1400 promotes = ~6 GiB H2D
                                per fire = 56->3 tok/s).
  VLLM_MOE_W2_GATE_TRACE   0 (default) | 1 log each fire/re-forward.
  VLLM_MOE_W2_GATE_FP_MAX  max promote->replay iterations per fired step
                           (default 3). A single replay is only FIRST-order:
                           upgraded early layers re-route later layers onto
                           still-cold experts inside the replay, which then
                           decide the token at 2-bit (measured on GPQA:
                           tau=1.00 66.2% vs 72.2% when a big pool hid the
                           residue; GSM8K showed no gap - stable routing).
                           The loop iterates until no rank promotes; steady
                           state costs nothing (first check promotes 0).
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.getenv("VLLM_MOE_W2_GATE", "0") == "1"
_SIGNAL = os.getenv("VLLM_MOE_W2_GATE_SIGNAL", "max_prob")
_DEFAULT_TAU = {"max_prob": 0.60, "margin": 1.5}
_TAU = float(os.getenv("VLLM_MOE_W2_GATE_TAU", str(_DEFAULT_TAU.get(_SIGNAL, 0.60))))
# Default 0 = UNLIMITED. The cap is a pure PERF knob (bounds a fire's H2D
# tail); quality-wise it must be neutral. Configs whose pool cannot hold
# one step's routed union are refused at boot (the FIRE FLOOR hardstop in
# moe_w2_delta.check_pool_floor); FORCE_POOL=1 configs run degraded and
# should set an explicit cap.
_MAX_PROMOTE = int(os.getenv("VLLM_MOE_W2_GATE_MAX_PROMOTE", "64"))
_FIRE_FP_MAX = max(int(os.getenv("VLLM_MOE_W2_GATE_FP_MAX", "3")), 1)


def fire_fp_max() -> int:
    """Bound on promote->replay iterations per fired step (see header)."""
    return _FIRE_FP_MAX
_TRACE = os.getenv("VLLM_MOE_W2_GATE_TRACE", "0") == "1"
# Measurement mode: on a fired step, COUNT routed experts (delta._need) instead of
# promoting/re-forwarding -> study whether 2-bit difficulty concentrates on few
# experts. Zero serving perturbation; read [need] lines from the delta trace.
_CAPTURE = os.getenv("VLLM_MOE_W2_GATE_CAPTURE", "0") == "1"
# Optional runtime-tunable threshold: if VLLM_MOE_W2_GATE_TAU_FILE points at a
# file, its float contents override TAU (mtime-cached, re-read on change). Lets a
# threshold/latency sweep run in ONE server without restarts. A value that can
# never fire (e.g. max_prob<=0.0) effectively disables the gate (baseline).
_TAU_FILE = os.getenv("VLLM_MOE_W2_GATE_TAU_FILE", "")
_tau_dyn = _TAU
_tau_mtime = -1.0
# Diagnostic: when 0, a fired step force-promotes (warms cache) but SKIPS the
# 2nd forward — isolates re-forward correctness from force_promote. Default 1.
_REFORWARD = os.getenv("VLLM_MOE_W2_GATE_REFORWARD", "1") == "1"
# S4 verify-row masking (VLLM_MOE_W2_GATE_SPEC_MASK=1, default off): on MTP
# verify steps the row aggregation skips rows the sampler cannot emit — the
# min-over-ALL-rows firing otherwise pays replays for uncertainty on rows
# that get DISCARDED after the first rejected draft. Knowledge-ported from
# the fin-03 gate-signal session (commit 6e4af0052 there): measured on DS4
# 1x PRO6000, MTP k=2, tau=0.60: re-forwards 24% -> 1.2% on prose
# (-31% -> -4% tok/s) and 16.9% -> 7.5% on code, quality flat. Their wider
# study also stands as the verdict AGAINST porting the ML signal machinery:
# no scalar or fitted combination beat max_prob materially (live labels
# 10.9% -> 10.3% fire@recall90), so max_prob stays the only signal here.
_SPEC_MASK = os.getenv("VLLM_MOE_W2_GATE_SPEC_MASK", "0") == "1"

# observability (cheap; only mutated when the gate is enabled)
_n_steps = 0
_n_fired = 0
_n_reforwarded = 0
_n_promoted = 0


def enabled() -> bool:
    return _ENABLED


def disable(reason: str) -> None:
    """Turn the gate off at runtime (boot-time config-coherence guard):
    a gate without a promotable tier pays its per-step decision sync for
    nothing. Loud by design."""
    global _ENABLED
    if _ENABLED:
        logger.warning("moe_w2 gate DISABLED: %s", reason)
    _ENABLED = False


def signal() -> str:
    return _SIGNAL


def _current_tau() -> float:
    """TAU, optionally overridden live by VLLM_MOE_W2_GATE_TAU_FILE (mtime-cached)."""
    global _tau_dyn, _tau_mtime
    if not _TAU_FILE:
        return _TAU
    try:
        m = os.path.getmtime(_TAU_FILE)
        if m != _tau_mtime:
            _tau_mtime = m
            with open(_TAU_FILE) as f:
                _tau_dyn = float(f.read().strip())
    except (OSError, ValueError):
        pass
    return _tau_dyn


def threshold() -> float:
    return _current_tau()


def reforward_enabled() -> bool:
    return _REFORWARD


def _spec_relevant_mask(logits: torch.Tensor, spec) -> torch.Tensor | None:
    """S4: rows the sampler can actually emit on an MTP verify step — the
    prefix of would-be-accepted drafts (greedy: draft == argmax of the
    previous row), the first rejection, and the bonus row only when every
    draft is accepted. Pure GPU ops (argmax + per-request cumprod over
    <= k elements), no host syncs; the decision sync below stays the only
    one. Returns None (no masking) when shapes don't line up — masking is
    an optimization, never a correctness dependency."""
    try:
        n_rows = logits.shape[0]
        num_draft = spec.num_draft_tokens          # python list per request
        if sum(num_draft) == 0:
            return None
        if n_rows != sum(num_draft) + len(num_draft):
            return None
        draft_ids = spec.draft_token_ids
        am = logits.argmax(dim=-1)
        mask = torch.zeros(n_rows, dtype=torch.bool, device=logits.device)
        row = 0
        dpos = 0
        for nd in num_draft:
            if nd == 0:
                mask[row] = True
                row += 1
                continue
            seg = am[row:row + nd]
            drafts = draft_ids[dpos:dpos + nd]
            acc = seg == drafts
            prefix = torch.cumprod(acc.int(), dim=0).bool()
            rel = torch.ones(nd, dtype=torch.bool, device=logits.device)
            if nd > 1:
                rel[1:] = prefix[:-1]              # row i needs 0..i-1 accepted
            mask[row:row + nd] = rel
            mask[row + nd] = prefix[-1]            # bonus row iff all accepted
            row += nd + 1
            dpos += nd
        return mask
    except Exception:  # noqa: BLE001 - masking is an optimization only
        return None


def should_reforward(logits: torch.Tensor, spec=None) -> bool:
    """Decide whether to re-forward this decode step at FP4.

    `logits` is the per-request next-token logits [num_reqs, vocab] from the
    1st (2-bit) forward. Fires when ANY request's top-1 is low-confidence -- the
    whole batch shares one CUDA graph, so a re-forward recomputes all rows
    together. Costs ONE GPU->CPU sync (the `.item()` below), incurred only when
    the gate is enabled.

    `spec` is the step's SpecDecodeMetadata (or None): with
    VLLM_MOE_W2_GATE_SPEC_MASK=1 the min-aggregation skips MTP verify rows
    the sampler cannot reach (see _SPEC_MASK above).

    `margin` and `max_prob` are computed directly from logits without a full
    softmax: margin = top1_logit - top2_logit == log p1 - log p2 (the softmax
    normaliser cancels), and max_prob = exp(top1_logit - logsumexp(logits)).
    """
    global _n_steps, _n_fired
    _n_steps += 1
    # Step-boundary signal for the tier managers: this is the one gate call
    # guaranteed once per decode step, so delta-only configs (no base tier,
    # hence no runner step_begin) still get event-driven manager passes
    # instead of the legacy free-running poll.
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    moe_w2_delta.wake_all()
    if logits is None or logits.numel() == 0:
        return False
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    tau = _current_tau()
    top2 = torch.topk(logits, 2, dim=-1).values  # [R, 2]
    if _SIGNAL == "margin":
        rows = top2[:, 0] - top2[:, 1]
    else:  # max_prob
        lse = torch.logsumexp(logits, dim=-1)
        rows = torch.exp(top2[:, 0] - lse)
    mask = None
    if _SPEC_MASK and spec is not None:
        mask = _spec_relevant_mask(logits, spec)
    if mask is not None:
        # masked rows are pushed to +inf so the min ignores them; an
        # all-masked batch (cannot happen: row 0 of each request is always
        # reachable) would fall through to no-fire, which is safe.
        rows = torch.where(mask, rows, torch.full_like(rows, float("inf")))
    worst = rows.min()
    fire = bool((worst <= tau).item())
    if fire:
        _n_fired += 1
        if _TRACE:
            logger.info("[gate] fire: %s worst=%.3f <= tau=%.3f (step %d)",
                        _SIGNAL, float(worst), tau, _n_steps)
    return fire


def step_promote_budget():
    """The per-STEP promotion budget for a fired step (None = unlimited).
    The runner threads it through the fixed-point loop so GATE_MAX_PROMOTE
    caps the STEP as documented, not each promote->replay pass."""
    return _MAX_PROMOTE if _MAX_PROMOTE > 0 else None


def force_promote_step(layers=None, max_promote="default") -> int:
    """Pull this step's COLD routed experts up to FP4 via the delta tier.
    Returns the number promoted (0 if the tier is absent / nothing cold).

    `max_promote`: remaining budget for THIS call ("default" = the full
    GATE_MAX_PROMOTE — legacy per-call semantics for callers outside the
    runner's FP loop; None = unlimited; 0 short-circuits to no promotion).

    MEASUREMENT mode (VLLM_MOE_W2_GATE_CAPTURE=1): instead of promoting, only
    COUNT this low-confidence step's routed experts (tier.mark_need_only) and
    return 0 -- so the caller skips the re-forward. Lets us study whether 2-bit
    difficulty concentrates on a small expert set with zero serving perturbation."""
    global _n_reforwarded, _n_promoted
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    tier = moe_w2_delta._TIER
    if tier is None:
        return 0
    if _CAPTURE:
        tier.mark_need_only(layers=layers)
        return 0
    cap = step_promote_budget() if max_promote == "default" else max_promote
    if cap is not None and cap <= 0:
        return 0
    n = tier.force_promote(layers=layers, max_promote=cap)
    if n > 0:
        _n_reforwarded += 1
        _n_promoted += n
        if _TRACE:
            logger.info("[gate] force-promoted %d experts -> re-forward", n)
    return n


def stats() -> dict:
    return dict(steps=_n_steps, fired=_n_fired, reforwarded=_n_reforwarded,
                promoted=_n_promoted, signal=_SIGNAL, tau=_TAU,
                fire_rate=(_n_fired / _n_steps if _n_steps else 0.0))
