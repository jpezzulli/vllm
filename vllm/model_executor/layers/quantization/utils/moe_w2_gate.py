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
  VLLM_MOE_W2_GATE_AUDIT   probability p (default 0 = off): on an armed step
                           that did NOT fire, force the fire path with prob p
                           ("audit fire"). The replay's re-decided logits are
                           used as usual (strictly-better quality), and with
                           GATE_DUMP set the pre/post argmax comparison yields
                           UNBIASED live labels for the silent-flip rate
                           P(argmax changes under promotion | signal > tau) —
                           the class no logit threshold can see (P5 measured
                           3.65% @ tau0.60 offline; this measures it live).
                           [O] 2026-07-31, gate mechanism v2 probe P9.
  VLLM_MOE_W2_GATE_DUMP    path (default "" = off): append one JSONL row per
                           FIRED/AUDITED step: worst signal value, per-row
                           signal, pre/post per-row argmax, flips, promotes,
                           replays, audit flag. Measurement mode: adds argmax
                           syncs on fired steps only; leave unset for formal
                           runs.
  VLLM_MOE_W2_GATE_FIRE_TIMING  0 (default) | 1: with GATE_DUMP set, the
                           runner times every promote->replay iteration of a
                           fired step (host wall clock, explicit GPU syncs at
                           the phase boundaries) and attaches the per-iteration
                           breakdown to the step's JSONL row ("timing"):
                           promote_ms (with the delta tier's snap/select/read/
                           h2d/map split), n promoted, replay_ms. Measurement
                           mode for the P11 fire-stall decomposition (plan item
                           4, 2026-08-01) — the added syncs serialize what the
                           FP loop already serializes, but leave unset for
                           formal runs.
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

import json
import os
import random
import time

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
# C>1 aggregation (Faza-3 item 7, [P] 2026-08-03; default = the FROZEN
# batch-min): VLLM_MOE_W2_GATE_AGG=min fires when ANY row is uncertain
# (replay cost scales with batch — measured expensive at C>1);
# =q:<float> fires when the q-quantile of row signals is <= tau, i.e.
# only when ENOUGH of the batch is uncertain (Gupta-style quantile
# aggregation). Evaluation knob for batched serving; single-row decode
# is unaffected (quantile of 1 row == min).
_AGG = os.getenv("VLLM_MOE_W2_GATE_AGG", "min")
_AGG_Q = float(_AGG.split(":", 1)[1]) if _AGG.startswith("q:") else 0.0
# P9 audit + dump (see header). Both default-off; the disabled path adds
# only falsy checks to the existing decision.
_AUDIT_P = float(os.getenv("VLLM_MOE_W2_GATE_AUDIT", "0") or "0")
_DUMP_PATH = os.getenv("VLLM_MOE_W2_GATE_DUMP", "")
# P11 fire-stall timing (see header). Default-off; requires DUMP for output.
_FIRE_TIMING = os.getenv("VLLM_MOE_W2_GATE_FIRE_TIMING", "0") == "1"


def fire_timing_enabled() -> bool:
    return _FIRE_TIMING and bool(_DUMP_PATH)


# 9b step-1 hidden-state capture ([P] 2026-08-03, default-off): with
# VLLM_MOE_W2_GATE_HCAP_DIR set (and DUMP on), the runner snapshots the
# EXL3 forward's per-layer h_l buffer at DECISION time on every dumped
# (fired/audited) step and hands it here; shards of {steps, h} tensors
# flush to the dir every _HCAP_FLUSH records. Offline consumers join on
# the dump JSONL's `step` field. Telemetry-only, never in formal runs.
_HCAP_DIR = os.getenv("VLLM_MOE_W2_GATE_HCAP_DIR", "")
_HCAP_FLUSH = int(os.getenv("VLLM_MOE_W2_GATE_HCAP_FLUSH", "256"))
_hcap_steps: list[int] = []
_hcap_tensors: list[torch.Tensor] = []
_hcap_shard = 0


def hcap_enabled() -> bool:
    return bool(_HCAP_DIR) and bool(_DUMP_PATH)


def hcap_store(step: int, h: torch.Tensor) -> None:
    """Queue one decision-time h snapshot ([n_layers, H] half, GPU);
    moved to CPU here (fired steps only — measurement mode)."""
    global _hcap_shard
    try:
        _hcap_steps.append(int(step))
        _hcap_tensors.append(h.to("cpu", non_blocking=False))
        if len(_hcap_steps) >= _HCAP_FLUSH:
            os.makedirs(_HCAP_DIR, exist_ok=True)
            path = os.path.join(_HCAP_DIR, f"hcap-{_hcap_shard:05d}.pt")
            torch.save(dict(steps=list(_hcap_steps),
                            h=torch.stack(_hcap_tensors)), path)
            _hcap_shard += 1
            _hcap_steps.clear()
            _hcap_tensors.clear()
    except Exception:  # noqa: BLE001 - observability must not break serving
        pass


def hcap_flush() -> None:
    """Final partial-shard flush (shutdown/atexit path)."""
    global _hcap_shard
    if _HCAP_DIR and _hcap_steps:
        try:
            os.makedirs(_HCAP_DIR, exist_ok=True)
            path = os.path.join(_HCAP_DIR, f"hcap-{_hcap_shard:05d}.pt")
            torch.save(dict(steps=list(_hcap_steps),
                            h=torch.stack(_hcap_tensors)), path)
            _hcap_shard += 1
            _hcap_steps.clear()
            _hcap_tensors.clear()
        except Exception:  # noqa: BLE001
            pass

# observability (cheap; only mutated when the gate is enabled)
_n_steps = 0
_n_fired = 0
_n_reforwarded = 0
_n_promoted = 0
_n_audit = 0
# last-decision snapshot for dump_step (single-threaded step execution:
# should_reforward -> [replay loop] -> dump_step, no interleaving)
_last_worst = float("nan")
_last_audit = False
_last_spec = False
_last_rows: torch.Tensor | None = None


def shutdown() -> None:
    """Reset mutable gate state between in-process engines."""
    global _ENABLED, _tau_dyn, _tau_mtime, _mp_dyn, _mp_mtime
    global _n_steps, _n_fired, _n_reforwarded, _n_promoted
    global _n_audit, _last_worst, _last_audit, _last_spec, _last_rows
    hcap_flush()
    _ENABLED = os.getenv("VLLM_MOE_W2_GATE", "0") == "1"
    _tau_dyn = _TAU
    _tau_mtime = -1.0
    _mp_dyn = _MAX_PROMOTE
    _mp_mtime = -1.0
    _n_steps = 0
    _n_fired = 0
    _n_reforwarded = 0
    _n_promoted = 0
    _n_audit = 0
    _last_worst = float("nan")
    _last_audit = False
    _last_spec = False
    _last_rows = None


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
    global _n_steps, _n_fired, _n_audit
    global _last_worst, _last_audit, _last_spec, _last_rows
    _n_steps += 1
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
    if _AGG_Q > 0.0 and rows.numel() > 1:
        # quantile aggregation for C>1: the decision statistic is the
        # q-quantile of finite row signals (masked +inf rows excluded by
        # clamping the quantile index into the finite prefix after sort).
        srt = rows.sort().values
        n_fin = torch.isfinite(srt).sum()
        qi = (n_fin.float() * _AGG_Q).long().clamp(min=0,
                                                   max=rows.numel() - 1)
        worst = srt[qi]
    else:
        worst = rows.min()
    # single GPU->CPU sync, same count as the old `(worst <= tau).item()`
    # (and one FEWER when TRACE is on, which used to float(worst) again)
    worst_f = float(worst.item())
    fire = worst_f <= tau
    audit = False
    if not fire and _AUDIT_P > 0.0 and random.random() < _AUDIT_P:
        # P9 audit: force the fire path on a confident step to obtain an
        # unbiased live label for P(flip | signal > tau). Replay logits
        # are consumed as usual (strictly-better re-decision).
        fire = True
        audit = True
    _last_worst = worst_f
    _last_audit = audit
    _last_spec = spec is not None
    _last_rows = rows.detach() if _DUMP_PATH else None
    if fire:
        if audit:
            _n_audit += 1
        else:
            _n_fired += 1
        if _TRACE:
            logger.info("[gate] %s: %s worst=%.3f %s tau=%.3f (step %d)",
                        "audit" if audit else "fire", _SIGNAL, worst_f,
                        "> " if audit else "<=", tau, _n_steps)
    return fire


# Optional runtime-tunable budget: VLLM_MOE_W2_GATE_MAX_PROMOTE_FILE points
# at a file whose integer contents override GATE_MAX_PROMOTE (mtime-cached,
# the TAU_FILE idiom) — a budget sweep runs on ONE warm server.
_MAX_PROMOTE_FILE = os.getenv("VLLM_MOE_W2_GATE_MAX_PROMOTE_FILE", "")
_mp_dyn = _MAX_PROMOTE
_mp_mtime = -1.0


def _current_max_promote() -> int:
    global _mp_dyn, _mp_mtime
    if not _MAX_PROMOTE_FILE:
        return _MAX_PROMOTE
    try:
        m = os.path.getmtime(_MAX_PROMOTE_FILE)
        if m != _mp_mtime:
            _mp_mtime = m
            with open(_MAX_PROMOTE_FILE) as f:
                _mp_dyn = int(f.read().strip())
    except (OSError, ValueError):
        pass
    return _mp_dyn


def step_promote_budget():
    """The per-STEP promotion budget for a fired step (None = unlimited).
    The runner threads it through the fixed-point loop so GATE_MAX_PROMOTE
    caps the STEP as documented, not each promote->replay pass."""
    mp = _current_max_promote()
    return mp if mp > 0 else None


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


def dump_enabled() -> bool:
    return bool(_DUMP_PATH)


def dump_step(pre_top1: torch.Tensor, post_top1: torch.Tensor,
              n_promoted: int, n_replays: int, timing=None) -> None:
    """Append one JSONL record for a fired/audited step (see header).
    Measurement-only path: never raises into serving. `pre_top1` /
    `post_top1` are per-row argmax of the logits before the first and
    after the last replay; equal tensors (n_replays==0) are still dumped
    (fires that could not promote are informative). `timing` (P11): the
    runner's per-iteration stall breakdown, attached verbatim."""
    if not _DUMP_PATH:
        return
    try:
        pre = pre_top1.tolist()
        post = post_top1.tolist()
        rows = _last_rows.tolist() if _last_rows is not None else []
        rec = dict(
            t=round(time.time(), 3), step=_n_steps, signal=_SIGNAL,
            tau=_current_tau(), worst=_last_worst, audit=_last_audit,
            spec=_last_spec, n_promoted=int(n_promoted),
            n_replays=int(n_replays), pre=pre, post=post,
            flips=sum(1 for a, b in zip(pre, post) if a != b),
            rows=[round(float(v), 6) for v in rows])
        if timing is not None:
            rec["timing"] = timing
        with open(_DUMP_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 - observability must not break serving
        pass


def stats() -> dict:
    return dict(steps=_n_steps, fired=_n_fired, reforwarded=_n_reforwarded,
                promoted=_n_promoted, audited=_n_audit, signal=_SIGNAL,
                tau=_TAU, audit_p=_AUDIT_P,
                fire_rate=(_n_fired / _n_steps if _n_steps else 0.0))
