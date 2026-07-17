# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 delta tier for the 1-GPU 2-bit MoE path (quality restoration).

Hot routed experts get their FULL e2m1 nibble planes cached in a small GPU
pool and dispatched to the `moe_w4_mm` kernel; everyone else stays on the
resident 2-bit base (`moe_w2_mm`). Block-32 scale planes are shared by both
tiers (kept on GPU since load).

Pieces:
  - host store: fragment-major FP4 planes per (layer, expert) in PINNED
    memory (built once at load from the checkpoint bytes, D2H);
  - GPU pool: VLLM_MOE_W2_DELTA_GB worth of 12.6 MiB expert slots
    (w13 8.4 MiB + w2 4.2 MiB packed back-to-back per slot);
  - slot table: int32 [layers, 256] on GPU (-1 = base tier), read by the
    desc-build kernel inside CUDA graphs;
  - manager thread: consumes the forward's last-seen expert flags
    (event-synced D2H), promotes seen-but-uncached experts (H2D on a side
    stream, capped per pass), evicts only experts cold for >= 2 passes.
    Passes are EVENT-DRIVEN: the runner signals step boundaries (wake_all
    via step_begin / the gate decision) and the manager runs at most one
    pass per signal, rate-limited by VLLM_MOE_W2_DELTA_TICK_MS; a wall
    clock timeout only provides liveness for configs that never signal.

Consistency model (deliberate): the table update is racy versus graph
replay — the worst case is one step reading the OLD tier for an expert,
which is numerically safe (both tiers are valid weights). Evicting only
cold slots keeps pool rewrites away from in-flight reads.
"""

import os
import threading
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Pool size: a number in GiB, or "auto" (also accepts -1) to defer the pool
# allocation until AFTER the KV cache is allocated and size it from the VRAM
# actually free then (minus a reserve for cudagraph capture + workspaces).
# Auto resolves the delta-vs-KV headroom trade at extreme context lengths:
# at 512K the KV eats the whole card and auto lands at 0 slots (the manual
# DELTA_GB=0 rule); at short context it recovers the usual 1-2 GiB pool.
_GB_RAW = os.getenv("VLLM_MOE_W2_DELTA_GB", "2.0").strip().lower()
_AUTO = _GB_RAW in ("auto", "-1", "-1.0")
_GB = 0.0 if _AUTO else float(_GB_RAW)
# Auto-mode knobs: VRAM to leave free for capture/workspaces, and an optional
# cap on the auto-sized pool (0 = uncapped).
_RESERVE_GB = float(os.getenv("VLLM_MOE_W2_DELTA_RESERVE_GB", "3.0"))
_MAX_GB = float(os.getenv("VLLM_MOE_W2_DELTA_MAX_GB", "0"))
_PROMOTE_PER_TICK = int(os.getenv("VLLM_MOE_W2_DELTA_PROMOTE", "8"))
# Hotness/need decay and heat-dump cadences are WALL-CLOCK based: passes are
# step-driven now, so tick-count periods would stretch with slow traffic
# (1000 ticks meant 5 s at the old 5 ms poll but a minute at 18 steps/s).
# The decay exponent scales with the actual elapsed time, so irregular pass
# spacing keeps the same half-life.
_DECAY_EVERY_S = max(float(os.getenv("VLLM_MOE_W2_DELTA_DECAY_S", "5")), 0.1)
_POOL_HEAT_EVERY_S = max(
    float(os.getenv("VLLM_MOE_W2_POOL_HEAT_EVERY_S", "10")), 1.0)
# Minimum spacing between manager passes. The manager is event-driven (one
# pass per step-boundary wake, see _loop/wake_all); this knob only rate
# limits pathological wake storms and paces the LEGACY polling mode used
# until the first wake arrives. It is no longer a fixed period: the old
# 5 ms free-running poll cost ~200 passes/s of GIL + CUDA-event syncs +
# slot churn against the forward thread (measured on GLM-5.2 TP2 base
# cache: removing it bought +40% decode), while slowing the period starved
# eviction recency (owners looked "seen" for the whole window) and broke
# long-context retrieval. Step-driven passes give fresh recency at step
# granularity with zero standing overhead.
_TICK_S = float(os.getenv("VLLM_MOE_W2_DELTA_TICK_MS", "5")) / 1e3
# Liveness fallback for wake-driven mode: an idle server still gets a pass
# this often (pending heat dumps, decay, trace), and a tier that never
# receives wakes keeps the legacy _TICK_S poll instead.
_FALLBACK_S = max(
    float(os.getenv("VLLM_MOE_W2_TICK_FALLBACK_MS", "250")) / 1e3, 0.05)

# Observability of the precision tiering (default OFF; behaviour-neutral — only
# adds logging). Useful for studying the delta in practice: which experts are
# FP4 right now, and how the working set churns.
#   VLLM_MOE_W2_DELTA_TRACE=0  silent (default)
#                          =1  periodic coverage/churn summary + per-layer
#                              FP4 histogram, every _TRACE_EVERY ticks
#                          =2  + one line per promotion/eviction (verbose)
#   VLLM_MOE_W2_DELTA_TRACE_EVERY=N   ticks between summaries (default 64)
#   VLLM_MOE_W2_DELTA_DUMP=<path>     also write the full precision map
#                                     (which expert is FP4 vs 2-bit) as JSON
#                                     at each summary, atomically (tail-able).
_TRACE = int(os.getenv("VLLM_MOE_W2_DELTA_TRACE", "0"))
_TRACE_EVERY = max(int(os.getenv("VLLM_MOE_W2_DELTA_TRACE_EVERY", "64")), 1)
_DUMP_PATH = os.getenv("VLLM_MOE_W2_DELTA_DUMP", "")

# Routing-trace capture for offline policy study (gated, off by default): record
# each tick's seen (layer,expert) frame and periodically write a .npy of
# [frame, layer, expert] rows. Replay it through candidate promote/evict
# policies in a simulator instead of restarting the 159B model each round.
_CAPTURE = os.getenv("VLLM_MOE_W2_DELTA_CAPTURE", "")
_CAPTURE_TICKS = int(os.getenv("VLLM_MOE_W2_DELTA_CAPTURE_TICKS", "20000"))

# Promotion/eviction policy (chosen via offline trace replay; see tools/delta_sim.py).
# "need" (gate-driven, the right default for a memory-bound decoder): the FP4 pool
# is filled ONLY by the confidence gate's force_promote -- an expert enters FP4
# *because a low-confidence token routed to it* (2-bit was insufficient and forced
# a re-run), never because it is merely hot. This matters because decode is
# HBM-bandwidth-bound and 2-bit is HALF the bytes of FP4: promoting a hot expert to
# FP4 makes the most-read weights SLOWER for no quality reason. Under "need" the
# background manager does NOT promote; it only ages/evicts, keeping the experts with
# the highest (recency-decayed) NEED score and letting everything else stay 2-bit
# (fast). Requires the gate on (VLLM_MOE_W2_GATE=1) to generate the need signal.
# "freq": promote the globally-hottest candidates and evict the least-frequently
# used slot -- maximizes FP4 COVERAGE/hit-rate (good when the pool >= working set so
# the extra FP4 bytes are amortized), but spends FP4 on experts 2-bit handled fine.
# "lru" = old behaviour (promote in order, evict coldest).
_POLICY = os.getenv("VLLM_MOE_W2_DELTA_POLICY", "freq")
_DECAY = float(os.getenv("VLLM_MOE_W2_DELTA_DECAY", "0.5"))

# Token-weighted hit-rate: when observability is on, the forward records per-expert
# routing COUNTS (not a binary flag) so the logged hit-rate reflects the fraction
# of token->expert ROUTINGS served at FP4 — the honest number. A binary-flag
# hit-rate under-counts badly, because the cached hot experts absorb
# disproportionately many tokens (a one-token expert and a 500-token expert count
# the same under a flag). Off by default -> the prod serving path is unchanged.
_COUNT = (_TRACE > 0) or bool(_CAPTURE)

# ---- GPU-pool warm-start (the "learning cache", colibri's .coli_usage) ----
# The tiered store's heat.json warms the HOST arena, but the GPU slot pool
# still converged from scratch every boot (misses + replays until the hot
# set assembles — the "pool still converging" phase of every bring-up). The
# base tier now persists its pool OWNERSHIP (freq-ranked (layer, expert)
# pairs) and preloads it before cudagraph capture on the next boot: the
# first decode step starts at yesterday's coverage instead of 0%.
#   VLLM_MOE_W2_POOL_HEAT      1 (default) dump + preload when a dir is known
#                              0 disables both
#   VLLM_MOE_W2_POOL_HEAT_DIR  where the JSON lives; defaults to
#                              VLLM_MOE_W2_STORE_DIR, then
#                              VLLM_MOE_W2_PLANES_CACHE (first set wins)
#   VLLM_MOE_W2_POOL_HEAT_EVERY_S  seconds between dumps (default 10;
#                              atomic tmp+rename)
#   VLLM_MOE_W2_POOL_HEAT_FILL max fraction of the pool preloaded (default
#                              0.9 — the first steps' working set should
#                              fetch into FREE slots, not evict preloads)
_POOL_HEAT = os.getenv("VLLM_MOE_W2_POOL_HEAT", "1") == "1"
_POOL_HEAT_DIR = (os.getenv("VLLM_MOE_W2_POOL_HEAT_DIR")
                  or os.getenv("VLLM_MOE_W2_STORE_DIR")
                  or os.getenv("VLLM_MOE_W2_PLANES_CACHE") or "")
_POOL_HEAT_FILL = min(max(
    float(os.getenv("VLLM_MOE_W2_POOL_HEAT_FILL", "0.9")), 0.0), 1.0)
_POOL_HEAT_VERSION = "pool-heat-v1"

# ---- lazy-promotion hysteresis (colibri's REPIN anti-ping-pong) ----------
# When the pool is FULL, a lazy background promotion evicts the least-
# valuable slot — with a full pool and a drifting working set this can
# ping-pong (promote X evicting Y, next tick promote Y evicting X), and
# every swap perturbs which experts serve at which tier mid-decode. With
# VLLM_MOE_W2_PROMO_HYST=h (>1), a candidate only displaces into a full
# pool when its recency-decayed freq exceeds h * (weakest eligible
# victim's) + 4 (colibri's 25%+4 rule at h=1.25). 0/1 (default) = off.
# Mandatory paths (miss restore, gate force_promote, prefill
# ensure_resident) are NEVER gated — correctness beats churn.
_PROMO_HYST = float(os.getenv("VLLM_MOE_W2_PROMO_HYST", "0"))

# ---- speculation guard (colibri's cold-cache DRAFT auto-off) -------------
# Speculative decode inflates the per-step expert union (verify batches
# route 1+k tokens), so on a COLD pool it multiplies miss-replays — colibri
# measured MTP as a net time LOSS until the cache warmed, and auto-disabled
# drafts. Here: when the windowed replay rate (the KPI) exceeds the
# threshold, the runner stops SCHEDULING drafts (the guard suppresses
# take_draft_token_ids); speculation resumes once the pool warms below
# thresh/2 (hysteresis). Value = replay %%. Default ON at 60 (measured on
# GLM-5.2 TP2: latches within seconds of a cold boot, releases at 30% once
# the pool converges — decode then jumped 41->53 tok/s from MTP — and
# re-latches around working-set shifts). Inert without the base cache;
# speculation itself is lossless, so the guard never changes outputs.
# 0 = off (schedule drafts unconditionally, pre-guard behaviour).
_SPEC_GUARD = float(os.getenv("VLLM_MOE_W2_SPEC_GUARD", "60"))
_SPEC_GUARD_EMA = float(os.getenv("VLLM_MOE_W2_SPEC_GUARD_EMA", "0.02"))
# Suppression must PROVE it warms the pool: if the replay EMA has not
# dropped by MIN_DROP points after PROBE suppressed steps, high replay is
# this config's steady state (working set > pool), not a cold start — the
# guard resumes drafts (at MTP acceptance ~3 they pay even at replay 100%:
# measured 14.2 vs 13.3 tok/s on GLM TP2) and backs off for COOLDOWN steps
# instead of latching forever on an unreachable resume threshold.
_GUARD_PROBE = max(int(os.getenv("VLLM_MOE_W2_SPEC_GUARD_PROBE", "500")), 50)
_GUARD_MIN_DROP = float(os.getenv("VLLM_MOE_W2_SPEC_GUARD_MIN_DROP", "5"))
_GUARD_COOLDOWN = max(
    int(os.getenv("VLLM_MOE_W2_SPEC_GUARD_COOLDOWN", "25000")), 0)

# Per-expert FP4 plane sizes for the SINGLE-GPU (TP1) layout. Under tensor
# parallelism the experts shard, so the real per-rank planes are smaller; the
# plane builder passes the per-rank sizes to get_tier()/DeltaTier and every
# consumer reads the per-instance self.{w13_bytes,w2_bytes,slot_bytes}. These
# module constants stay as the TP1 default / fallback (byte-identical to the
# original single-GPU path).
W13_BYTES = 4096 * 4096 // 2          # 8.0 MiB (TP1)
W2_BYTES = 4096 * 2048 // 2           # 4.0 MiB (TP1)
SLOT_BYTES = W13_BYTES + W2_BYTES     # 12.0 MiB per expert (TP1)


class DeltaTier:
    def __init__(self, n_layers: int, n_experts: int, dev,
                 w13_bytes: int = W13_BYTES, w2_bytes: int = W2_BYTES,
                 pool_gb: float | None = None, policy: str | None = None,
                 tag: str = "delta", host_pinned: bool = True):
        self.n_layers = n_layers
        self.E = n_experts
        # Per-instance policy/tag: with the base cache and the FP4 tier
        # coexisting, the base tier wants freq/lru (hot-set convergence)
        # while the FP4 tier wants "need" (gate-filled only) — a shared
        # module-level policy cannot express that. `tag` disambiguates the
        # two tiers' log lines.
        self._policy = policy if policy is not None else _POLICY
        self._tag = tag
        # Pinned host store is right for tiers that promote continuously (the
        # base cache's misses, the standalone delta's lazy manager). The FP4
        # need-pool OVER the base promotes only on gate fires — pageable
        # memory there saves ~360 GiB of pinned RAM on GLM TP2/TP4 (pinning
        # that much alongside the base store + load staging exhausts a 1 TB
        # host: measured OOM at boot), at the cost of a bounce-buffer copy on
        # the rare promote.
        self._host_pinned = host_pinned
        if isinstance(dev, torch.device) and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        # Per-rank FP4 plane sizes (== the TP1 module constants on a single GPU;
        # halved under TP2, quartered under TP4 as the experts shard). All slot
        # math, host staging, and the desc-kernel pool indexing read these so the
        # tier is correct under tensor parallelism.
        self.w13_bytes = w13_bytes
        self.w2_bytes = w2_bytes
        self.slot_bytes = w13_bytes + w2_bytes
        # Auto mode: the pool is NOT allocated here (weight load runs before
        # the KV cache is planned). finalize_auto() -- driven by the worker
        # right after initialize_kv_cache -- sizes it from the VRAM actually
        # free once KV has taken its share, and always before any cudagraph
        # capture (the desc kernel bakes pool pointers into the graph).
        # `pool_gb` overrides the module-level env sizing (used by the BASE
        # cache tier, which has its own env knob and never auto-defers).
        _gb = _GB if pool_gb is None else float(pool_gb)
        self._auto_pending = _AUTO and pool_gb is None
        self.n_slots = 0 if self._auto_pending else max(
            int(_gb * 2**30) // self.slot_bytes, 8)
        self.pool = torch.empty(self.n_slots, self.slot_bytes, dtype=torch.uint8,
                                device=dev)
        # device table read by the desc kernel; host mirror for the manager
        self.slot_table = torch.full((n_layers, n_experts), -1,
                                     dtype=torch.int32, device=dev)
        self._mirror = torch.full((n_layers, n_experts), -1,
                                  dtype=torch.int32)
        # slot -> (layer, expert, last_seen_tick) as three flat CPU tensors;
        # layer -1 = free. Tensor form keeps the eviction/refresh paths fully
        # vectorized: the old list-of-tuples cost three O(n_slots) python
        # comprehensions per eviction batch — i.e. per force_promote, i.e.
        # per REPLAYED STEP on base-cache configs (~9k slots each).
        self._alloc_owner(self.n_slots)
        self._free = list(range(self.n_slots))
        # routing signal written by the forward (graph-replayed scatter): token
        # COUNTS per expert when observability is on (int32, for token-weighted
        # hit-rate), else a cheap binary flag (uint8). Read by the manager only;
        # the desc kernel reads slot_table, never this.
        _seen_dtype = torch.int32 if _COUNT else torch.uint8
        self.seen = torch.zeros(n_layers, n_experts, dtype=_seen_dtype,
                                device=dev)
        self._seen_host = torch.zeros_like(self.seen, device="cpu",
                                           pin_memory=True)
        # Host store behind a backend interface: classic pinned/pageable
        # tensors (default), or an on-disk pack file with the kernel page
        # cache as the RAM tier (VLLM_MOE_W2_STORE_DIR) — see moe_w2_store.
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_store)
        self._store = moe_w2_store.make_store(
            tag, n_layers, n_experts, self.slot_bytes, pinned=host_pinned)
        self._stream = torch.cuda.Stream(dev)
        # Guards pool/slot_table/_mirror/_owner/_free/_freq mutations. In steady
        # state only the manager thread mutates them (uncontended). The
        # confidence-gated re-forward (force_promote) mutates from the FORWARD
        # thread, so the two must be serialized. The desc kernel only READS
        # slot_table (never takes the lock), so steady-state decode is unaffected.
        self._lock = threading.Lock()
        # Serializes the seen-snapshot sequence (D2H copy_ -> event sync ->
        # nonzero) across the manager tick and the forward-thread paths
        # (force_promote / ensure_resident / mark_need_only). They share ONE
        # pinned _seen_host and ONE side stream; torch's two-pass nonzero
        # overruns its output when the input mutates between passes — a
        # concurrent copy_ from the other thread does exactly that. Measured
        # on GLM long-prefill needles: TensorAdvancedIndexing.cpp:3008
        # internal assert -> glibc heap corruption -> dead worker; a torn
        # snapshot could also evict an in-flight expert (bad bytes served).
        self._snap_lock = threading.Lock()
        self._tick = 0
        self._stop = False
        self._thread = None
        # Step-boundary wakeup (wake()/wake_all()): coalescing event; the
        # _wake_driven latch flips the loop from legacy polling to pure
        # event cadence the moment the first signal arrives.
        self._wake = threading.Event()
        self._wake_driven = False
        self._last_capture = 0.0    # graph-capture grace (see notify_capture)
        # observability counters: cumulative + per-summary window
        self._n_promoted = 0
        self._n_evicted = 0
        self._win_promoted = 0
        self._win_evicted = 0
        self._last_summary_tick = 0
        self._win_hits = 0.0     # token-weighted FP4-served routings this window
        self._win_active = 0.0   # token-weighted total routings this window
        self._win_hits_d = 0     # distinct FP4-served experts this window
        self._win_active_d = 0   # distinct active experts this window
        self._cap_frames = []
        self._cap_done = False
        # store-membership mask cache for the vectorized candidate filter
        self._store_mask_cache: torch.Tensor | None = None
        self._store_mask_n = -1
        # spec-guard re-probe state (see kpi_step): suppression must prove
        # it warms the pool within _GUARD_PROBE steps or it resumes drafts
        # and backs off.
        self._guard_since = 0
        self._guard_ema0 = 0.0
        self._guard_cooldown_until = 0
        # per-step KPI counters (window + cumulative), fed by kpi_step()
        self._kpi_steps = 0
        self._kpi_miss_pairs = 0
        self._kpi_replays = 0
        self._kpi_c_steps = 0
        self._kpi_c_replays = 0
        self._kpi_unfixed = 0     # experts replay could NOT restore (window)
        self._kpi_deferred = 0    # slotless warm-up fetches (no reader; window)
        self._kpi_2nd = 0         # extra replays for second-order misses
        self._kpi_fp_giveup = 0   # steps that accepted second-order residue
        self._kpi_fp_resid = 0    # residual missing pairs in those steps
        # Slots touched since step_begin(): promoted or hit by any pass of
        # the CURRENT step. Never evictable (even in the emergency pass) —
        # without this, a fixed-point iteration can evict pass-k's fetches
        # to serve pass-k+1 (their seen marks are zeroed after each
        # snapshot) and ping-pong past the replay cap.
        self._step_pins: set[int] = set()
        # Prefill-layer pin scope (ensure_resident): pins live only until
        # the NEXT ensure_resident call (the layer's eager GEMMs are done
        # by then). Step-scoped pins would accumulate the whole prefill
        # working set (61 layers x routed >> pool) and starve every later
        # layer's fetch; no scope at all lets the emergency eviction take
        # the CURRENT layer's slots between its fetch and its GEMMs.
        self._layer_pins: set[int] = set()
        # Residency coupling (BASE tier only, split-FP4 over the base
        # cache): the coexisting FP4 tier whose refinement slots read THIS
        # tier's base slots — its mapped experts are eviction-blocked here.
        # Set by get_tier() when the need-pool is created in split mode.
        self._coupled_fp4 = None
        # Set by check_pool_floor when this NEED-pool cannot hold one
        # step's routed union: gate fires must take the EAGER LAYER-WISE
        # replay path (ensure_resident per MoE layer) instead of the
        # graph replay — see internal/EAGER_FIRE_NEXT_SESSION.md.
        self._sub_floor = False
        # Draft-affinity prefetch (VLLM_MOE_W2_PREFETCH=1, base tier only):
        # route_log = in-graph [n_layers, T_cap, K_cap] routing log written
        # by the forward glue; _aff = token->experts affinity table folded
        # from it post-step; draft_prefetch() predicts+fetches at step start.
        self.route_log: torch.Tensor | None = None
        self._aff: torch.Tensor | None = None
        self._aff_k = 8
        self._last_ids: torch.Tensor | None = None
        self._kpi_prefetched = 0
        # spec-guard state: EMA of the per-step replay indicator (fed by
        # kpi_step) + the current suppression latch (read by the runner).
        self._replay_ema = 0.0
        self._spec_suppressed = False
        # pool warm-start bookkeeping (dump cadence + one-shot preload flag).
        # The previous run's heat file is read AT TIER CREATION and stashed:
        # the manager thread ticks all through weight load and its periodic
        # dump would otherwise overwrite the file with the (still empty)
        # boot pool long before the worker-driven preload reads it —
        # measured: a 8.9k-owner file clobbered to 0 within seconds of boot.
        # Dumps stay blocked until the stash is consumed (or absent).
        self._heat_preloaded = False
        self._heat_pending: list | None = None
        if _POOL_HEAT and _POOL_HEAT_DIR and tag == "base":
            self._heat_pending = self._read_heat_file()
        # recency-decayed routing frequency per expert (drives the freq policy)
        self._freq = torch.zeros(n_layers, n_experts, dtype=torch.float32)
        self._last_decay_t = time.monotonic()
        self._heat_last_dump_t = 0.0
        # NEED signal (gate-driven policy): how often the confidence gate flagged a
        # step routing to this expert (i.e. 2-bit was insufficient). Recency-decayed
        # like _freq; the eviction key under _POLICY == "need".
        self._need = torch.zeros(n_layers, n_experts, dtype=torch.float32)
        if self._auto_pending:
            logger.info("moe_w2 delta tier: auto-sizing deferred until after "
                        "KV-cache allocation (slot %.1f MiB, reserve %.1f GiB)",
                        self.slot_bytes / 2**20, _RESERVE_GB)
        else:
            logger.info("moe_w2 delta tier: %d slots x %.1f MiB (%.2f GiB pool)",
                        self.n_slots, self.slot_bytes / 2**20,
                        self.n_slots * self.slot_bytes / 2**30)
        if _TRACE:
            logger.info("moe_w2 delta trace ON: level %d, every %d ticks%s",
                        _TRACE, _TRACE_EVERY,
                        f", dump -> {_DUMP_PATH}" if _DUMP_PATH else "")
        if _CAPTURE:
            logger.info("moe_w2 delta CAPTURE ON -> %s (dump every 200 frames)",
                        _CAPTURE)

    # ---- owner bookkeeping (tensor form) ----------------------------------

    def _alloc_owner(self, n: int) -> None:
        """slot -> (layer, expert, last_seen_tick) as three flat CPU
        tensors; layer -1 = free. Tensor form keeps eviction, hysteresis
        and recency refresh fully vectorized — the old list-of-tuples cost
        three O(n_slots) python comprehensions per eviction batch, which
        runs once per REPLAYED step on base-cache configs."""
        self._owner_li = torch.full((n,), -1, dtype=torch.long)
        self._owner_ei = torch.full((n,), -1, dtype=torch.long)
        self._owner_tick = torch.zeros(n, dtype=torch.long)

    def _own(self, slot: int, li: int, ei: int) -> None:
        """Reserve a slot for (li, ei) at the current tick (lock held)."""
        self._owner_li[slot] = li
        self._owner_ei[slot] = ei
        self._owner_tick[slot] = self._tick

    @property
    def _owner(self):
        """Read-only compatibility view (tests/tools): [(li, ei, tick)].
        Internal code uses the tensor fields directly."""
        return list(zip(self._owner_li.tolist(), self._owner_ei.tolist(),
                        self._owner_tick.tolist()))

    def _store_mask(self) -> torch.Tensor:
        """Boolean [n_layers] mask of layers present in the host store;
        cached until the store grows (which only happens at load time)."""
        n = len(self._store)
        if self._store_mask_cache is None or n != self._store_mask_n:
            m = torch.zeros(self.n_layers, dtype=torch.bool)
            for li in range(self.n_layers):
                if li in self._store:
                    m[li] = True
            self._store_mask_cache, self._store_mask_n = m, n
        return self._store_mask_cache

    # ---- load-time -------------------------------------------------------

    def finalize_auto(self) -> None:
        """Size + allocate the auto pool from the VRAM free AFTER KV-cache
        allocation (VLLM_MOE_W2_DELTA_GB=auto). Driven by the worker's
        initialize_from_config, i.e. after the KV tensors exist and BEFORE any
        cudagraph capture — the desc kernel bakes `pool`/`slot_table` pointers
        into the graph, so the pool must not be reallocated after capture.

        Sizing: free VRAM minus _RESERVE_GB (capture + workspace headroom),
        optionally capped by _MAX_GB, floored at 0 slots (extreme-context
        configs where KV takes the whole card -> tier inert, exactly like the
        manual DELTA_GB=0 rule, but without the manual step). No-op unless
        auto mode is pending."""
        if not self._auto_pending:
            return
        self._auto_pending = False
        free_b, _ = torch.cuda.mem_get_info(self.dev)
        budget = free_b - int(_RESERVE_GB * 2**30)
        if _MAX_GB > 0:
            budget = min(budget, int(_MAX_GB * 2**30))
        n = max(budget // self.slot_bytes, 0)
        if n == 0:
            # Nothing to cache into -> behave exactly like manual DELTA_GB=0:
            # release the host store too (tens of GiB of host RAM the tier
            # can never use; candidates require li in the store, so the
            # manager and force_promote turn inert).
            with self._lock:
                self._store.release()
            logger.info(
                "moe_w2 delta tier AUTO: %.2f GiB free after KV < reserve "
                "%.1f GiB -> pool disabled (0 slots, pure 2-bit; host store "
                "released)",
                free_b / 2**30, _RESERVE_GB)
            return
        self.n_slots = int(n)
        self.pool = torch.empty(self.n_slots, self.slot_bytes,
                                dtype=torch.uint8, device=self.dev)
        self._alloc_owner(self.n_slots)
        self._free = list(range(self.n_slots))
        logger.info(
            "moe_w2 delta tier AUTO: %d slots x %.1f MiB (%.2f GiB pool; "
            "%.2f GiB was free after KV, reserve %.1f GiB)",
            self.n_slots, self.slot_bytes / 2**20,
            self.n_slots * self.slot_bytes / 2**30, free_b / 2**30,
            _RESERVE_GB)

    def add_layer_host_planes(self, layer_key: int, w13_plane_gpu, w2_plane_gpu):
        """Stage a layer's fragment-major FP4 planes into pinned host memory.

        Called from the plane builder while the FP4 planes are transiently
        on GPU; w13/w2 are [E, bytes] u8.
        """
        self.add_layer_host_sections(layer_key,
                                     (w13_plane_gpu,), (w2_plane_gpu,))

    def add_layer_host_sections(self, layer_key: int, parts13, parts2):
        """Stage a layer whose slot sections arrive as SEPARATE GPU tensors
        (e.g. [fp4_13|sc13] / [fp4_2|sc2] for the over-base FP4 tier): copy
        each part D2H into its slice of the host row — a GPU-side cat of
        multi-GiB planes is exactly the transient that OOMs a 32 GB card
        during load. With the pack-file store a layer already on disk is
        skipped entirely (persistent quantization cache)."""
        self._store.add_layer(layer_key, (*parts13, *parts2))

    def start(self):
        if self._thread is not None:   # idempotent: started once at tier creation
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="moe-w2-delta")
        self._thread.start()

    # ---- manager loop ----------------------------------------------------

    def _loop(self):
        # Event-driven cadence: block until a step-boundary wake (or the
        # liveness fallback), enforce the _TICK_S rate limit, run ONE pass.
        # Tick counts advance per pass, so tick-denominated ages ("cold
        # >= 2 ticks", the seen window, decay/heat cadences) now track
        # steps — matching their original intent of "in-flight graph
        # protection" — instead of wall-clock poll iterations.
        last = 0.0
        while not self._stop:
            if self._wake_driven:
                self._wake.wait(timeout=_FALLBACK_S)
            else:
                # No wake ever received (no base tier / gate armed in the
                # runner): keep the legacy fixed-period poll.
                self._wake.wait(timeout=_TICK_S)
            self._wake.clear()
            gap = _TICK_S - (time.monotonic() - last)
            if gap > 0:
                time.sleep(gap)
            last = time.monotonic()
            try:
                torch.cuda.set_device(self.dev)
                self._tick_once()
            except Exception as e:  # noqa: BLE001 - never kill serving
                logger.warning("delta tick failed: %s", e)
                time.sleep(1.0)

    def wake(self):
        """Step-boundary signal (runner thread): run one manager pass as
        soon as the rate limiter allows. Lock-free and coalescing — a burst
        of calls between two passes collapses into one."""
        self._wake_driven = True
        self._wake.set()

    def notify_capture(self):
        """Forward calls this while stream capture is active: the manager
        idles through the whole capture phase plus a grace window (captures
        run with thread_local error mode as the primary guard; this avoids
        even benign allocator interleaving)."""
        self._last_capture = time.monotonic()

    def _tick_once(self):
        if time.monotonic() - self._last_capture < 5.0:
            return
        self._tick += 1
        with self._snap_lock:
            with torch.cuda.stream(self._stream):
                self._seen_host.copy_(self.seen, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._stream)
            ev.synchronize()
            seen = self._seen_host.nonzero()
            # token counts for the hit-rate below — read under the snap lock
            # so a concurrent snapshot can't swap the values underneath.
            cnt_raw = self._seen_host[seen[:, 0], seen[:, 1]]
        if seen.numel() == 0:
            return
        if _CAPTURE and not self._cap_done:
            self._cap_frames.append((self._tick, seen.to(torch.int16).clone()))
            n = len(self._cap_frames)
            if n % 200 == 0 or n >= _CAPTURE_TICKS:
                self._dump_capture(final=n >= _CAPTURE_TICKS)
        # hit-rate: of this tick's routings, how many hit an FP4 slot. `cnt` is
        # token counts (count-mode) or 1s (binary) per active expert -> the
        # token-weighted ratio is the honest one; the distinct ratio is the old
        # flag-based number, logged alongside for comparison.
        cnt = cnt_raw.to(torch.float64)
        cached = self._mirror[seen[:, 0], seen[:, 1]] >= 0
        self._win_hits += float((cnt * cached).sum())
        self._win_active += float(cnt.sum())
        self._win_hits_d += int(cached.sum())
        self._win_active_d += int(seen.shape[0])
        # Mutate shared tier state under the lock (serialized with a concurrent
        # gate-driven force_promote on the forward thread).
        with self._lock:
            li_idx, ei_idx = seen[:, 0], seen[:, 1]
            # recency-decayed routing frequency (the hotness signal)
            self._freq[li_idx, ei_idx] += 1.0
            # refresh last_seen for cached owners; collect promotion
            # candidates — all vectorized (no per-pair python)
            slots = self._mirror[li_idx, ei_idx].long()
            hit = slots >= 0
            if bool(hit.any()):
                self._owner_tick[slots[hit]] = self._tick
            cand = [tuple(p) for p in
                    seen[~hit & self._store_mask()[li_idx]].tolist()]
            # "need" policy: the background manager does NOT promote — FP4 is filled
            # only by the gate's force_promote (an expert 2-bit handled fine never
            # gets pulled to the slower FP4 path). freq/lru: promote the hottest
            # candidates first so the limited pool tracks genuinely hot experts
            # across ALL layers (vs the layer-sorted order that starved past layer 0).
            if self._policy != "need":
                if self._policy == "freq" and len(cand) > 1:
                    ca = torch.tensor(cand)
                    order = torch.argsort(self._freq[ca[:, 0], ca[:, 1]],
                                          descending=True)
                    cand = [cand[i] for i in order.tolist()]
                # Lazy-promotion hysteresis (opt-in): into a FULL pool, only
                # candidates clearly hotter than the weakest eligible victim
                # may displace (anti ping-pong; colibri's 25%+4 REPIN rule).
                # One vectorized floor per tick; free-slot promotions and all
                # mandatory paths are unaffected.
                if _PROMO_HYST > 1.0 and not self._free and cand:
                    floor = self._promo_floor()
                    if floor is not None:
                        bar = floor * _PROMO_HYST + 4.0
                        cand = [(li, ei) for li, ei in cand
                                if float(self._freq[li, ei]) > bar]
                promoted = 0
                for li, ei in cand:
                    if promoted >= _PROMOTE_PER_TICK:
                        break
                    slot = self._take_slot()
                    if slot is None:
                        break
                    self._promote(li, ei, slot)
                    promoted += 1
            now = time.monotonic()
            if now - self._last_decay_t >= _DECAY_EVERY_S:
                # exponent scales with real elapsed time, so the half-life is
                # invariant to pass spacing (step-driven passes are irregular)
                f = _DECAY ** ((now - self._last_decay_t) / _DECAY_EVERY_S)
                self._freq *= f  # keep the frequency signal recent + bounded
                self._need *= f  # need decays too -> tracks RECENT 2-bit misses
                self._last_decay_t = now
        # reset flags for the next window (racy with the forward's scatter
        # of ones — a lost flag only delays promotion by one tick)
        if self._tick % 4 == 0:
            self.seen.zero_()
        # PILOT (router-lookahead) consumption: prefetch the experts the
        # in-graph predictor flagged for upcoming layers. 5 ms ticks against
        # a 30-60 ms step: fetched bytes typically land before the step's
        # replay (or the next step) needs them.
        from vllm.model_executor.layers.quantization.utils import moe_w2_looka
        if moe_w2_looka.pilot_enabled() and self._tag == "base":
            try:
                moe_w2_looka.tick_consume(self)
            except Exception as e:  # noqa: BLE001 - prefetch is best-effort
                logger.warning_once("moe_w2 PILOT tick failed: %s", e)
        # pool warm-start: periodic freq-ranked ownership dump (atomic)
        if (_POOL_HEAT and self._tag == "base" and _POOL_HEAT_DIR
                and time.monotonic() - self._heat_last_dump_t
                >= _POOL_HEAT_EVERY_S):
            self._heat_last_dump_t = time.monotonic()
            self._dump_pool_heat()
        if _TRACE and self._tick - self._last_summary_tick >= _TRACE_EVERY:
            self._log_summary()
            self._last_summary_tick = self._tick

    def _promo_floor(self) -> float | None:
        """freq of the weakest EVICTABLE slot (hysteresis reference): owners
        not in the current seen window, >=2 ticks cold, not step-pinned.
        Lock held by caller. None when nothing is evictable (promotions
        will fail to take a slot anyway)."""
        li, ei = self._owner_li, self._owner_ei
        tk = self._owner_tick
        lic, eic = li.clamp(min=0), ei.clamp(min=0)
        blocked = ((li < 0) | self._seen_host[lic, eic].to(torch.bool)
                   | ((self._tick - tk) < 2))
        if self._step_pins:
            blocked[list(self._step_pins)] = True
        if self._layer_pins:
            blocked[list(self._layer_pins)] = True
        if bool(blocked.all()):
            return None
        key = self._freq[lic, eic].double()
        key[blocked] = float("inf")
        return float(key.min())

    # ---- GPU-pool warm-start (persist + preload the hot ownership) --------

    def _heat_path(self) -> str:
        from vllm.model_executor.layers.quantization.utils.moe_w2_store \
            import _rank_suffix
        return os.path.join(
            _POOL_HEAT_DIR, f"pool-heat.{self._tag}.{_rank_suffix()}.json")

    def _heat_meta(self) -> dict:
        return dict(version=_POOL_HEAT_VERSION, tag=self._tag,
                    n_layers=self.n_layers, E=self.E,
                    slot_bytes=self.slot_bytes)

    def _read_heat_file(self) -> list | None:
        """Parse + validate the heat file into an owner list (None on any
        miss). Called at tier creation, before the manager can clobber it."""
        import json
        try:
            path = self._heat_path()
            if not os.path.exists(path):
                return None
            with open(path) as f:
                snap = json.load(f)
            if snap.get("meta") != self._heat_meta():
                logger.warning(
                    "moe_w2 pool-heat: %s is for another model/config "
                    "(%s vs %s) — ignored", path, snap.get("meta"),
                    self._heat_meta())
                return None
            pairs = [(int(li), int(ei)) for li, ei in snap.get("owners", [])
                     if 0 <= int(li) < self.n_layers and 0 <= int(ei) < self.E]
            logger.info("moe_w2 [%s] pool-heat: %d hot experts stashed from "
                        "%s (preload runs after KV-cache init)",
                        self._tag, len(pairs), path)
            return pairs
        except Exception as e:  # noqa: BLE001 - warm-start is best-effort
            logger.warning("moe_w2 pool-heat read failed: %s", e)
            return None

    def _dump_pool_heat(self) -> None:
        """Write the pool's current owners, hottest first (freq is the
        recency-decayed routing frequency), atomically. ~100 KB of JSON for
        a 10k-slot pool; a missed dump only costs preload freshness.

        Blocked while a stashed preload is UNCONSUMED (the manager ticks all
        through weight load — dumping there would persist an empty boot pool
        over the previous run's real heat), and skipped for empty pools."""
        import json
        if self._heat_pending is not None and not self._heat_preloaded:
            return
        try:
            with self._lock:
                live = self._owner_li >= 0
                lil, eil = self._owner_li[live], self._owner_ei[live]
                owners = list(zip(lil.tolist(), eil.tolist(),
                                  self._freq[lil, eil].tolist()))
            if not owners:
                return
            owners.sort(key=lambda r: -r[2])
            snap = dict(meta=self._heat_meta(),
                        owners=[[li, ei] for li, ei, _f in owners])
            os.makedirs(_POOL_HEAT_DIR, exist_ok=True)
            path = self._heat_path()
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(snap, f)
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001 - observability path
            logger.warning_once("moe_w2 pool-heat dump failed: %s", e)

    def preload_pool(self) -> int:
        """One-shot boot preload of the pool from the last run's heat dump
        (the colibri 'learning cache': the engine starts with YOUR hot
        experts already resident instead of converging from 0% every boot).
        Driven by the worker AFTER weight load + KV allocation and BEFORE
        cudagraph capture (slot_table writes must precede graph bake-in).
        Consumes the owner list stashed at tier creation (_heat_pending —
        the file itself may have been re-dumped since). Fills at most
        _POOL_HEAT_FILL of the pool, hottest first, leaving free slots for
        the first steps' fresh working set. Never raises."""
        import time as _time
        pairs = self._heat_pending
        if self._heat_preloaded or not pairs or self.n_slots == 0:
            self._heat_preloaded = True    # unblock the periodic dumps
            return 0
        self._heat_preloaded = True
        try:
            budget = int(self.n_slots * _POOL_HEAT_FILL)
            pairs = pairs[:budget]
            t0 = _time.perf_counter()
            total = 0
            # chunked: rows_for stages through a pinned buffer sized to the
            # batch — 256-row chunks keep it ~1-3 GiB at GLM/Kimi slot sizes
            for i in range(0, len(pairs), 256):
                total += self.prefetch_pairs(pairs[i:i + 256])
            logger.info(
                "moe_w2 [%s] pool warm-start: %d/%d experts preloaded "
                "in %.1f s (%.1f%% of the pool starts WARM)",
                self._tag, total, len(pairs),
                _time.perf_counter() - t0,
                100.0 * total / max(self.n_slots, 1))
            return total
        except Exception as e:  # noqa: BLE001 - warm-start is best-effort
            logger.warning("moe_w2 pool-heat preload failed: %s", e)
            return 0

    def _take_slots_batch(self, k: int, emergency: bool = False,
                          min_cold: int = 2,
                          admission_keys=None) -> list[int]:
        """Take up to k slots (lock held by caller): free list first, then ONE
        vectorized eviction pass over all slots. Replaces the old per-slot
        python scan per promotion — O(n_slots) per TAKEN slot — which at GLM
        scale (4k slots x hundreds of gate promotions per fire) burned seconds
        of GIL time per fired step and starved the forward thread.

        `emergency=True` (synchronous runner-thread callers only — never the
        background manager) adds two fallback eviction passes when the first
        pass cannot cover k: pass 2 relaxes the 2-tick coldness bound but
        keeps the seen-window exclusion; pass 3 also drops the seen window
        (a recency HEURISTIC — with per-step manager passes the snapshot
        holds one full step's routing, which under an MTP-verify gate storm
        approaches the pool size and starved pass 2 into leaving zeroed
        contributions: measured as corrupted math on DS4 three-tier,
        tau→storm). Pass 3 keeps only the correctness contracts: step pins
        (the CURRENT step's reads/fetches) and split-FP4 residency coupling.
        See the pass comments below.

        Eviction policy is unchanged: least-valuable slot by _POLICY key
        (need / freq / lru), restricted to slots whose owner is not active in
        the current seen window (read directly from the _seen_host snapshot —
        the call sites always passed a set built from exactly that) and cold
        >= 2 ticks, so in-flight graph reads never hit a rewritten slot.
        Victims are unmapped here (graphs stop dispatching w4 before bytes
        change); the caller reserves _owner for each returned slot.

        `admission_keys` (opportunistic tiers ONLY — the FP4 need-pool,
        never the base cache): per-candidate priority values aligned with
        the caller's candidate order (descending). Free slots are granted
        unconditionally; an EVICTION is granted only while the candidate's
        priority EXCEEDS the victim's key — pool value never decreases.
        Without this, an uncapped gate fire at a full pool bulk-evicted
        the accumulated high-need core for once-routed candidates
        (wholesale churn; the old MAX_PROMOTE cap merely masked it)."""
        out: list[int] = []
        while self._free and len(out) < k:
            out.append(self._free.pop())
        k_evict = k - len(out)
        if k_evict <= 0:
            return out
        if admission_keys is not None and len(out) >= len(admission_keys):
            return out
        li, ei = self._owner_li, self._owner_ei
        tk = self._owner_tick
        lic, eic = li.clamp(min=0), ei.clamp(min=0)
        if self._policy == "need":
            key = self._need[lic, eic].double()
        elif self._policy == "freq":
            key = self._freq[lic, eic].double()
        else:
            # double, not clone: the tensorized owner-tick is int64 and the
            # inf sentinel below cannot be represented there (lru is not a
            # production policy for these tiers; surfaced by unit tests)
            key = tk.double()
        # Exclusion tiers. HARD (correctness): free markers, step-pinned
        # slots (fetched or hit by a pass of the CURRENT step — a replay is
        # about to read them), and the coupled-FP4 block added below. SOFT
        # (recency heuristic): owners active in the seen window — likely to
        # be routed again soon, but nothing in-flight reads them once the
        # step that marked them is over and they are not step-pinned.
        hard = torch.zeros_like(li, dtype=torch.bool)
        hard |= (li < 0)
        if self._step_pins:
            hard[list(self._step_pins)] = True
        if self._layer_pins:
            hard[list(self._layer_pins)] = True
        blocked = hard | self._seen_host[lic, eic].to(torch.bool)
        # Residency coupling (split-FP4 over the base cache): never evict a
        # BASE slot whose expert is mapped in the coupled FP4 tier — the
        # split kernel reads its refinement against THESE base codes+scales.
        # HARD exclusion: these experts are exactly the gate's recent
        # promotions (the quality-critical set), and in split mode a
        # base-evicted pair downgrades to a plain MISS — zeroed inside a
        # gate replay (which never refetches). Measured as corrupted math
        # on gate-fired tokens when this was soft. The cost is bounded by
        # the (small) FP4 pool size. The mirror read is lockless (the other
        # tier mutates it under ITS lock); a torn value is benign: dispatch
        # requires residency in BOTH slot tables, so a stale exclusion only
        # delays one eviction and a missed one downgrades that expert to a
        # base miss -> the standard fetch+replay restores it.
        if self._coupled_fp4 is not None:
            cpl = self._coupled_fp4._mirror[lic, eic] >= 0
            hard |= cpl
            blocked |= cpl
        # Pass 1: only >=min_cold-tick-cold victims (never disturbs slots a
        # CONCURRENT in-flight graph might still read — the background
        # manager's constraint; speculative prefetchers pass a much higher
        # bound so they can never churn the hot set). Passes 2-3
        # (emergency): the synchronous callers (force_promote /
        # ensure_resident, runner thread, no forward in flight) relax first
        # the coldness bound, then the seen window, rather than leave a
        # missing expert UNRESTORED — a replay that keeps zeroed
        # contributions is a silent quality hit and a nondeterminism
        # source, strictly worse than evicting a warm-but-idle slot. The
        # seen window must be droppable: one step of an MTP-verify gate
        # storm can mark nearly the whole pool (DS4: 3 tok x top-8 x 61
        # layers vs an 11 GiB pool), and an all-blocked pass 2 silently
        # zeroes hundreds of experts per step (measured: corrupted math).
        passes = [blocked | ((self._tick - tk) < min_cold)]
        if emergency:
            passes.append(blocked)
            passes.append(hard)
        taken: set[int] = set()
        for ineligible in passes:
            need = k - len(out)
            if need <= 0:
                break
            mask = ineligible.clone()
            if taken:
                mask[list(taken)] = True
            kk = key.clone()
            kk[mask] = float("inf")
            take = min(need, int((~mask).sum()))
            if take <= 0:
                continue
            victims = torch.topk(kk, take, largest=False)
            vidx = victims.indices.tolist()
            vkey = victims.values.tolist()
            for s, vk in zip(vidx, vkey):
                if admission_keys is not None:
                    # value-monotone admission: the slot granted next serves
                    # candidate #len(out) (callers zip candidates with the
                    # returned slots in order). Victims come cheapest-first
                    # and candidate priorities descend, so the first losing
                    # pairing ends the whole take (later pairs only worse).
                    if len(out) >= len(admission_keys) or \
                            not (admission_keys[len(out)] > vk):
                        return out
                vli = int(self._owner_li[s])
                vei = int(self._owner_ei[s])
                self.slot_table[vli, vei] = -1
                self._mirror[vli, vei] = -1
                self._n_evicted += 1
                self._win_evicted += 1
                if _TRACE >= 2:
                    logger.info(
                        "[%s] evict   L%-2d E%-3d  slot %-4d (cold %d ticks)",
                        self._tag, vli, vei, s,
                        self._tick - int(self._owner_tick[s]))
                taken.add(s)
                out.append(s)
        return out

    def _take_slot(self, seen_set=None):
        """Single-slot wrapper (kept for the unit tests / external callers)."""
        slots = self._take_slots_batch(1)
        return slots[0] if slots else None

    def _promote(self, li, ei, slot):
        row = self._store.rows_for([(li, ei)])[0]
        with torch.cuda.stream(self._stream):
            self.pool[slot].copy_(row, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self._stream)
        ev.synchronize()           # bytes resident BEFORE mapping
        # The MANAGER thread runs this: its slot_table write must order
        # against graph replays on the forward stream. Today that ordering
        # comes from the thread's current stream being the LEGACY default
        # stream (device-wide serializing) — make that explicit instead of
        # accidental (review 2.5: a vLLM move to per-thread streams would
        # silently break the mapping-after-bytes guarantee).
        with torch.cuda.stream(torch.cuda.default_stream(self.dev)):
            self.slot_table[li, ei] = slot
        self._mirror[li, ei] = slot
        self._own(slot, li, ei)
        self._n_promoted += 1
        self._win_promoted += 1
        if _TRACE >= 2:
            logger.info("[%s] promote L%-2d E%-3d  slot %-4d (tick %d)",
                        self._tag, li, ei, slot, self._tick)

    # ---- confidence-gated re-forward (directive 2 / Step B) --------------

    def step_begin(self) -> None:
        """Open a new step's pin scope (runner: before the first miss read;
        prefill: at each ensure_resident). Slots touched after this call are
        pinned against eviction until the next step_begin — the fixed-point
        replay's passes must never cannibalize each other's fetches.

        Doubles as the manager's step-boundary signal: every live tier
        (base AND fp4) gets one pass per step instead of a free-running
        poll."""
        with self._lock:
            self._step_pins.clear()
            self._layer_pins.clear()
        wake_all()

    def step_end(self) -> None:
        """Close this step's SEEN window (runner: after the replay loop /
        the gate decision consumed it). `seen` must scope to ONE step: the
        eviction exclusion and force_promote's hit-pinning derive from it,
        and the old free-running manager implicitly kept it that narrow by
        zeroing every ~20 ms of wall clock. Event-driven passes turned that
        into "last 4 steps", so a PREFILL's marks (61 ensure_resident
        layers, thousands of pairs) survived into the first decode step's
        force_promote — which then pinned every hit slot, leaving eviction
        NO victims (measured: [pool 2123, pinned 2123], hundreds of
        unpromotable experts per step, corrupted first decode tokens).
        Decode steps with misses are unaffected (force_promote already
        zeroes after its snapshot); this closes the window for miss-free
        and prefill steps. A mark lost to an in-flight scatter only delays
        one lazy promotion (the manager's own idiom).

        ALSO closes the step's PIN scope. step_begin (the other clear
        site) only runs on the base-cache runner path, so on delta-only
        configs (GPU-resident planes + FP4 tier: the DS4 1-GPU/TP2 and
        GLM TP4 quality recipes) the gate's force_promote hit-pins
        accumulated FOREVER — once the pool filled, every eviction pass
        (incl. the emergency pass, whose only hard exclusion is these
        pins) found zero victims and the gate silently stopped promoting
        (measured on the fin-03 line at tau=1.0: 22.9k fires -> 2
        re-forwards, [pool 341, pinned 341]). The step is fully executed
        when the runner calls this (replays done; nothing reads the pool
        until the next step's passes re-pin what they touch), so clearing
        here restores the one-step scope on every config. Ported as
        knowledge from fin-03 commit 6e4af0052 (its gate line diverged
        pre-fire-floor; the fix concept carries verbatim)."""
        self.seen.zero_()
        with self._lock:
            self._step_pins.clear()
            self._layer_pins.clear()

    # ---- draft-affinity prefetch (VLLM_MOE_W2_PREFETCH=1) ------------------

    def draft_prefetch(self, cur_ids: torch.Tensor) -> int:
        """Called by the runner at the START of a decode step (outside
        capture), with the step's REAL input token ids — under MTP these are
        exactly last step's sampled+draft tokens, so this IS the draft
        signal. Two actions:

        1. fold the PREVIOUS step's in-graph route_log into the
           token->experts affinity table (routing is strongly
           token-identity-correlated — the same fact that makes 19%
           coverage serve 96% of routings);
        2. predict this step's routed set from the table and fetch the
           non-resident predictions on the side stream, OVERLAPPING the
           forward: layers deep enough to run after the mapping hit
           directly, earlier ones find the bytes resident when the
           fixed-point replay re-runs — either way the fetch leaves the
           critical path.

        Best-effort by design: never emergency-evicts, capped per step,
        wrong predictions cost one cold slot each and decay away."""
        if self.route_log is None:
            return 0
        # route_log may be armed for LOOKA/PILOT only — the affinity
        # predictor still needs its own opt-in.
        if os.getenv("VLLM_MOE_W2_PREFETCH", "0") != "1":
            return 0
        t_cap = self.route_log.shape[1]
        ids = cur_ids[:t_cap].detach().to("cpu", non_blocking=False).long()
        # 1) fold previous step's log for the ids that produced it
        if self._last_ids is not None and self._last_ids.numel() > 0:
            k = self._aff_k
            log = (self.route_log[:, :self._last_ids.shape[0], :k]
                   .to("cpu", non_blocking=False).to(torch.int16))
            need = int(self._last_ids.max()) + 1
            if self._aff is None or self._aff.shape[0] < need:
                size = 1 << (need - 1).bit_length()
                grown = torch.full((size, self.n_layers, k), -1,
                                   dtype=torch.int16)
                if self._aff is not None:
                    grown[:self._aff.shape[0]] = self._aff
                self._aff = grown
            self._aff[self._last_ids] = log.permute(1, 0, 2)
        self._last_ids = ids
        if self._aff is None:
            return 0
        # 2) predict + prefetch
        vids = ids[ids < self._aff.shape[0]]
        if vids.numel() == 0:
            return 0
        pred = self._aff[vids]                     # [T, L, k]
        cap = int(os.getenv("VLLM_MOE_W2_PREFETCH_CAP", "32"))
        pairs = []
        for li in range(self.n_layers):
            es = pred[:, li, :].flatten()
            es = es[es >= 0]
            if es.numel() == 0:
                continue
            for e in torch.unique(es).tolist():
                if int(self._mirror[li, e]) < 0 and li in self._store:
                    pairs.append((li, int(e)))
                    if len(pairs) >= cap:
                        break
            if len(pairs) >= cap:
                break
        return self.prefetch_pairs(pairs)

    def prefetch_pairs(self, pairs: list, cold_ticks: int = 0) -> int:
        """Best-effort fetch of COLD (layer, expert) pairs into the pool
        (non-emergency slots, step-pinned, side-stream H2D, single sync
        before mapping). The shared tail of the affinity prefetcher and the
        PILOT router-lookahead consumer; also the pool warm-start's fetch
        primitive. Caller must NOT hold the lock.

        `cold_ticks` > 0 restricts eviction victims to slots idle at least
        that many ticks (speculative callers must not churn the hot set);
        free slots are always eligible."""
        if not pairs:
            return 0
        with self._lock:
            pairs = [(li, ei) for li, ei in pairs
                     if int(self._mirror[li, ei]) < 0 and li in self._store]
            if not pairs:
                return 0
            slots = self._take_slots_batch(len(pairs),
                                           min_cold=max(cold_ticks, 2))
            plan = [(p, s) for p, s in zip(pairs, slots)]
            if not plan:
                return 0
            rows = self._store.rows_for([p for p, _ in plan])
            for ((li, ei), slot), row in zip(plan, rows):
                self._own(slot, li, ei)
                self._step_pins.add(slot)
                with torch.cuda.stream(self._stream):
                    self.pool[slot].copy_(row, non_blocking=True)
            with torch.cuda.stream(self._stream):
                ev = torch.cuda.Event()
                ev.record(self._stream)
            ev.synchronize()
            for (li, ei), slot in plan:
                self.slot_table[li, ei] = slot
                self._mirror[li, ei] = slot
                self._freq[li, ei] += 1.0
            self._n_promoted += len(plan)
            self._win_promoted += len(plan)
            self._kpi_prefetched += len(plan)
        return len(plan)

    def force_promote(self, layers=None, max_promote=None,
                      pin: bool = True) -> int:
        """Synchronously pull this step's COLD routed experts up to FP4, for a
        confidence-gated re-forward (directive 2 / Step B).

        Reads `seen` (the forward's routed-expert scatter) to find routed
        (layer, expert) pairs still on the 2-bit base (slot_table == -1), copies
        their FP4 planes H2D on the side stream, blocks ONCE on a single event,
        then maps them into `slot_table`. A subsequent CUDA-graph REPLAY then
        recomputes exactly those experts at FP4 "for free". Promotions persist
        (a superset of lazy promotion), so a flagged step also warms the cache.

        Unlike the background `_promote`, this runs on the FORWARD thread, so all
        pool/table mutations are serialized with the manager via `self._lock`.
        Slot writes stay on the default (forward) stream and pool copies on the
        side stream — matching `_promote`/`_take_slot` so in-flight graph reads
        never observe a half-rewritten slot (eviction only touches >=2-tick-cold
        slots). Must NOT be called during graph capture.

        Args:
            layers: optional iterable of layer keys to restrict to (default all).
            max_promote: optional cap on experts promoted this call.
            pin: step-pin the touched slots (hits + fetches). Pass False
                ONLY when no replay follows this call in the current step
                (the runner's LAST fetch of the fixed-point loop): those
                fetches serve future steps, and pinning them would grow
                the step's pin set to the whole working set — on MTP
                verify (3 pos x top-8 x 61 layers) that is the entire
                14 GiB pool, leaving the eviction no victims (measured:
                [pool 2123, pinned 2123], hundreds of unpromotable
                experts, corrupted math).
        Returns:
            number of experts newly promoted to FP4.
        """
        if len(self._store) == 0:
            return 0
        # snapshot the forward's routed-expert scatter. The side stream must
        # WAIT on the forward (main) stream first so the snapshot includes THIS
        # step's mark_seen scatter — cross-stream ordering is not automatic, and
        # a snapshot racing ahead would miss this step's cold experts.
        main = torch.cuda.current_stream(self.dev)
        with self._snap_lock:
            with torch.cuda.stream(self._stream):
                self._stream.wait_stream(main)
                self._seen_host.copy_(self.seen, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._stream)
            ev.synchronize()
            seen = self._seen_host.nonzero()
        if seen.numel() == 0:
            return 0
        # Bound the working set to RECENT steps: `seen` otherwise accumulates
        # up to 4 manager ticks of routings (the manager zeroes it lazily), so
        # on deep/wide models a single fire tried to force-promote every
        # expert routed in the whole window (GLM-5.2: 75 layers x top-8 ->
        # 600+/step, measured 200-1400 per fire = up to ~6 GiB synchronous
        # H2D). Zeroing after the snapshot is the manager's own idiom; a flag
        # lost to the in-flight scatter race only delays a lazy promotion.
        self.seen.zero_()
        layer_filter = set(layers) if layers is not None else None
        with self._lock:
            li_idx, ei_idx = seen[:, 0], seen[:, 1]
            slots = self._mirror[li_idx, ei_idx].long()
            hit = slots >= 0
            # Refresh last_seen for CACHED owners routed this window
            # (mirrors _tick_once). force_promote zeroes `seen` after its
            # snapshot, so a manager pass racing this step would otherwise
            # see an empty window, find these slots "cold" (stale tick),
            # evict + rewrite one WHILE the imminent replay reads it —
            # measured as rare cross-request greedy nondeterminism. The
            # tick refresh keeps them coldness-protected for >=2 ticks.
            if bool(hit.any()):
                hs = slots[hit]
                self._owner_tick[hs] = self._tick
                if pin:
                    self._step_pins.update(hs.tolist())
            keep = torch.ones(seen.shape[0], dtype=torch.bool)
            if layer_filter is not None:
                lf = torch.zeros(self.n_layers, dtype=torch.bool)
                lf[list(layer_filter)] = True
                keep = lf[li_idx]
            # NEED signal: this step was gate-flagged (2-bit low-confidence), so
            # every expert active in it gets a need bump -- INCLUDING ones already
            # FP4 (so repeat offenders accumulate need and resist eviction). The
            # true culprits are the experts consistently present across fires;
            # decay washes out the coincidental ones.
            self._need[li_idx[keep], ei_idx[keep]] += 1.0
            cand = [tuple(p) for p in
                    seen[keep & ~hit & self._store_mask()[li_idx]].tolist()]
            if not cand:
                return 0
            # capped promote prioritizes the most-NEEDED experts under the gate-driven
            # policy (repeat offenders first); hottest-first otherwise.
            if len(cand) > 1:
                ca = torch.tensor(cand)
                rank = self._need if self._policy == "need" else self._freq
                order = torch.argsort(rank[ca[:, 0], ca[:, 1]], descending=True)
                cand = [cand[i] for i in order.tolist()]
            if max_promote is not None:
                cand = cand[:max_promote]
            # take ALL slots in one vectorized batch (evictions unmap on the
            # forward stream), issue all copies on the side stream, then a
            # SINGLE sync before mapping — bytes resident before any graph
            # replay can read them. The batch returns distinct slots, and each
            # gets its _owner reserved before the copies, so a concurrent
            # manager tick can never hand one of them out again (two experts
            # -> one slot -> pool corruption; see the force_promote history).
            # emergency=True: this is the synchronous runner-thread path with
            # no forward in flight — leaving a miss UNRESTORED is worse than
            # evicting a warm-but-idle slot (see _take_slots_batch).
            # NO admission control here — the FIRE CONTRACT forbids it
            # (2026-07-13): the gate fired because THIS token is uncertain
            # and the imminent replay must see the step's routed set
            # upgraded. Step pins + the seen window already restrict
            # victims to experts NOT routed in this step, so unconditional
            # promotion cannot corrupt the fired step; a "bad" eviction
            # only costs future re-promotions (speed). At high tau this
            # converges to native regardless of eviction policy — an
            # admission veto breaks exactly that limit (the uncertain
            # token would keep its 2-bit expert forever). Standing
            # coverage for NON-fired steps is a POOL SIZE knob, not a
            # promotion-policy knob. (admission_keys plumbing remains in
            # _take_slots_batch for lazy/manager promotions, which carry
            # no fire contract.)
            slots = self._take_slots_batch(len(cand), emergency=True)
            plan = [((li, ei), slot) for (li, ei), slot in zip(cand, slots)]
            if not plan:
                return 0
            # one batched host read (pack-file store: mmap -> pinned stage;
            # pinned store: zero-copy views), THEN the H2D copies — all stage
            # rows stay valid until the single sync below.
            rows = self._store.rows_for([p for p, _ in plan])
            for ((li, ei), slot), row in zip(plan, rows):
                self._own(slot, li, ei)
                if pin:
                    self._step_pins.add(slot)
                with torch.cuda.stream(self._stream):
                    self.pool[slot].copy_(row, non_blocking=True)
            with torch.cuda.stream(self._stream):
                ev = torch.cuda.Event()
                ev.record(self._stream)
            ev.synchronize()
            for (li, ei), slot in plan:
                self.slot_table[li, ei] = slot
                self._mirror[li, ei] = slot
                self._freq[li, ei] += 1.0
            self._n_promoted += len(plan)
            self._win_promoted += len(plan)
        if len(plan) < len(cand):
            if self._policy == "need":
                # Opportunistic tier: the shortfall is admission control
                # doing its job (candidates below the pool's value floor
                # stay 2-bit — correct, just not upgraded). Not a quality
                # event; visible via the deferred KPI.
                self._kpi_deferred += len(cand) - len(plan)
            elif not pin:
                # No reader follows this call in the current step (see the
                # pin doc): a slotless candidate is a DEFERRED warm-up
                # fetch, not a quality event — the next step that routes it
                # refetches under its own (pinned) mandatory pass. Expected
                # whenever pool ~= one step's working set.
                self._kpi_deferred += len(cand) - len(plan)
            else:
                # QUALITY KPI: some of this step's missing experts got NO
                # slot (free list empty + every victim ineligible) and a
                # replay WILL re-read the step — their contributions stay
                # zeroed: a silent quality drop even at MISS_TOL=0, and
                # (pool-content-dependent) a source of run-to-run greedy
                # nondeterminism. The fix is a bigger pool, not a knob.
                self._kpi_unfixed += len(cand) - len(plan)
                npin = len(self._step_pins)
                ncpl = (int((self._coupled_fp4._mirror[
                    self._owner_li.clamp(min=0),
                    self._owner_ei.clamp(min=0)] >= 0).sum())
                        if self._coupled_fp4 is not None else 0)
                logger.warning(
                    "moe_w2 [%s]: %d missing experts could not be promoted "
                    "(pool too tight to evict) — replay keeps their zeroed "
                    "contributions. Raise the pool GiB; occurrences counted "
                    "in the KPI line. [pool %d, pinned %d, fp4-coupled %d, "
                    "free %d, wanted %d]", self._tag, len(cand) - len(plan),
                    self.n_slots, npin, ncpl, len(self._free), len(cand))
        if _TRACE >= 2:
            logger.info("[%s] force-promote %d experts (gate)",
                        self._tag, len(plan))
        return len(plan)

    def ensure_resident(self, layer_key: int, ids: torch.Tensor) -> int:
        """Synchronously make the given experts of ONE layer resident (base
        cache, prefill path): fetch every (layer_key, e) not in the pool,
        blocking until the bytes are on GPU. Runs on the forward thread OUTSIDE
        cudagraph capture (prefill is eager), serialized with the manager via
        the lock. Marks the ids seen first so the batched eviction never picks
        this layer's in-flight experts as victims. Returns experts fetched."""
        if layer_key not in self._store:
            return 0
        ids = ids.unique().long()
        mark_seen(self.seen[layer_key], ids.to(self.dev))
        # snapshot seen (protects eviction) exactly like force_promote
        main = torch.cuda.current_stream(self.dev)
        with self._snap_lock:
            with torch.cuda.stream(self._stream):
                self._stream.wait_stream(main)
                self._seen_host.copy_(self.seen, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._stream)
            ev.synchronize()
        with self._lock:
            # open THIS layer's pin scope (releases the previous layer's —
            # its GEMMs are done; see _layer_pins in __init__)
            self._layer_pins.clear()
            slots = self._mirror[layer_key].long()[ids.cpu()]
            hit = slots >= 0
            if bool(hit.any()):
                # tick-refresh + LAYER-pin cached hits: the eager GEMMs
                # read them right after this call, and the emergency
                # eviction (this very call's fetch, pass 3) may otherwise
                # take them — seen/coldness are soft exclusions there.
                hs = slots[hit]
                self._owner_tick[hs] = self._tick
                self._layer_pins.update(hs.tolist())
            cand = [(layer_key, int(e)) for e in ids.cpu()[~hit].tolist()]
            if not cand:
                return 0
            # emergency=True: prefill MUST have its whole layer resident —
            # an unfetched expert here zeroes contributions for EVERY token
            # of the chunk (the existing pool-too-small warning path).
            slots = self._take_slots_batch(len(cand), emergency=True)
            plan = [((li, ei), slot) for (li, ei), slot in zip(cand, slots)]
            if not plan:
                return 0
            # scan=True: prefill working sets are one-shot — the tiered
            # store may warm FREE arena slots with them but must not evict
            # its decode hot set (a long prefill would wipe the arena).
            rows = self._store.rows_for([p for p, _ in plan], scan=True)
            for ((li, ei), slot), row in zip(plan, rows):
                self._own(slot, li, ei)
                self._layer_pins.add(slot)
                with torch.cuda.stream(self._stream):
                    self.pool[slot].copy_(row, non_blocking=True)
            with torch.cuda.stream(self._stream):
                ev = torch.cuda.Event()
                ev.record(self._stream)
            ev.synchronize()
            for (li, ei), slot in plan:
                self.slot_table[li, ei] = slot
                self._mirror[li, ei] = slot
                self._freq[li, ei] += 1.0
            self._n_promoted += len(plan)
            self._win_promoted += len(plan)
        if len(plan) < len(cand):
            logger.warning_once(
                "moe_w2 base cache: pool too small for one prefill layer "
                "(%d experts unfetched) — increase VLLM_MOE_W2_BASE_CACHE_GB",
                len(cand) - len(plan))
        return len(plan)

    def mark_need_only(self, layers=None) -> int:
        """MEASUREMENT ONLY: bump _need for THIS step's routed experts (a low-conf,
        gate-fired step) WITHOUT promoting anything. Lets us study whether 2-bit
        difficulty concentrates on a small expert set (=> a small persistent FP4
        pool can cover the 'hard' experts) before committing to a pool policy. No
        slot/pool mutation, no H2D copy, no re-forward -> zero serving perturbation
        beyond the seen snapshot. _freq (all-routing) keeps accruing in _tick_once,
        so _need/_freq gives per-expert over-representation in low-confidence steps."""
        if len(self._store) == 0:
            return 0
        main = torch.cuda.current_stream(self.dev)
        with self._snap_lock:
            with torch.cuda.stream(self._stream):
                self._stream.wait_stream(main)
                self._seen_host.copy_(self.seen, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._stream)
            ev.synchronize()
            seen = self._seen_host.nonzero()
        if seen.numel() == 0:
            return 0
        lf = set(layers) if layers is not None else None
        n = 0
        with self._lock:
            for li, ei in seen.tolist():
                if lf is not None and li not in lf:
                    continue
                self._need[li, ei] += 1.0
                n += 1
        return n

    def stats(self):
        cached = int((self._mirror >= 0).sum())
        return dict(slots=self.n_slots, cached=cached, tick=self._tick,
                    promoted=self._n_promoted, evicted=self._n_evicted)

    # ---- per-step KPI (base cache) ----------------------------------------

    def kpi_step(self, miss_pairs: int, replayed: bool) -> None:
        """Fed by the runner once per executed step (TP-max miss count and
        whether the step was replayed). The windowed replay rate is THE
        pool-sizing KPI: replays double the step, so tok/s tracks the
        fraction of zero-miss steps, which falls off a cliff with pool
        coverage — NOT the (much flatter) token hit-rate. Runner thread
        only, no lock needed."""
        self._kpi_steps += 1
        self._kpi_miss_pairs += miss_pairs
        self._kpi_replays += int(replayed)
        self._kpi_c_steps += 1
        self._kpi_c_replays += int(replayed)
        # Replay-rate EMA: maintained unconditionally — it is the shared
        # "pool warmth" signal (spec-guard latch below, PILOT's cold-phase
        # gate in moe_w2_looka.tick_consume).
        self._replay_ema += _SPEC_GUARD_EMA * (
            float(replayed) - self._replay_ema)
        # spec-guard: hysteresis latch on the EMA. The runner reads
        # _spec_suppressed in take_draft_token_ids — while latched, drafts
        # are not scheduled, verify batches shrink to 1 token, and the cold
        # pool warms at the pure-decode rate instead of replaying
        # k+1-token unions (colibri's DRAFT auto-off, reversible).
        if _SPEC_GUARD > 0:
            pct = 100.0 * self._replay_ema
            if (not self._spec_suppressed and pct > _SPEC_GUARD
                    and self._kpi_c_steps >= self._guard_cooldown_until):
                self._spec_suppressed = True
                self._guard_since = self._kpi_c_steps
                self._guard_ema0 = pct
                logger.info(
                    "moe_w2 [%s] SPEC-GUARD: replay EMA %.0f%% > %.0f%% — "
                    "draft scheduling suppressed while the pool warms "
                    "(resumes < %.0f%%, or after a %d-step probe without "
                    "progress)", self._tag, pct, _SPEC_GUARD,
                    _SPEC_GUARD / 2, _GUARD_PROBE)
            elif self._spec_suppressed:
                warmed = pct < _SPEC_GUARD / 2
                probe_over = (self._kpi_c_steps - self._guard_since
                              >= _GUARD_PROBE)
                if warmed:
                    self._spec_suppressed = False
                    logger.info(
                        "moe_w2 [%s] SPEC-GUARD: replay EMA %.0f%% < %.0f%% "
                        "— draft scheduling resumed", self._tag, pct,
                        _SPEC_GUARD / 2)
                elif probe_over and self._guard_ema0 - pct < _GUARD_MIN_DROP:
                    # No warm-up progress: steady-state high-replay regime,
                    # not a cold start. Drafts back on, guard backs off.
                    self._spec_suppressed = False
                    self._guard_cooldown_until = (
                        self._kpi_c_steps + _GUARD_COOLDOWN)
                    logger.info(
                        "moe_w2 [%s] SPEC-GUARD: no warm-up progress after "
                        "%d suppressed steps (EMA %.0f%% -> %.0f%%) — "
                        "steady-state replay regime, drafts resumed "
                        "(guard backs off %d steps)", self._tag,
                        self._kpi_c_steps - self._guard_since,
                        self._guard_ema0, pct, _GUARD_COOLDOWN)
                elif probe_over:
                    # warming (EMA falling): extend the probe window
                    self._guard_since = self._kpi_c_steps
                    self._guard_ema0 = pct
        if _KPI_EVERY <= 0 or self._kpi_steps < _KPI_EVERY:
            return
        cov_total = max(len(self._store), 1) * self.E
        unfixed = (f"; UNRESTORED experts: {self._kpi_unfixed} "
                   "(pool too tight — quality at risk)"
                   if self._kpi_unfixed else "")
        if self._kpi_deferred:
            unfixed += (f"; deferred warm-ups: {self._kpi_deferred} "
                        "(benign; pool ~= step working set)")
        if self._kpi_2nd:
            unfixed += (f"; second-order replays: {self._kpi_2nd}")
        if self._kpi_fp_giveup:
            unfixed += (
                f"; fp-residue: {self._kpi_fp_giveup} steps "
                f"(avg {self._kpi_fp_resid / self._kpi_fp_giveup:.0f} pairs)")
        if self._kpi_prefetched:
            unfixed += (f"; draft-prefetched: {self._kpi_prefetched}")
        if _SPEC_GUARD > 0:
            unfixed += (f"; replay EMA {100.0 * self._replay_ema:.0f}%"
                        f"{' (SPEC SUPPRESSED)' if self._spec_suppressed else ''}")
        from vllm.model_executor.layers.quantization.utils import moe_w2_looka
        unfixed += moe_w2_looka.kpi_summary()
        logger.info(
            "[%s] KPI: replay %.1f%% of last %d steps (avg %.1f missing "
            "pairs/step; cumulative %.1f%% of %d) — pool %d slots = %.1f%% "
            "of experts%s. Rising replay%% => raise the pool "
            "(VLLM_MOE_W2_BASE_CACHE_GB) before touching anything else.",
            self._tag, 100.0 * self._kpi_replays / self._kpi_steps,
            self._kpi_steps, self._kpi_miss_pairs / self._kpi_steps,
            100.0 * self._kpi_c_replays / max(self._kpi_c_steps, 1),
            self._kpi_c_steps, self.n_slots,
            100.0 * self.n_slots / cov_total, unfixed)
        self._kpi_steps = self._kpi_miss_pairs = self._kpi_replays = 0
        self._kpi_unfixed = 0
        self._kpi_deferred = 0
        self._kpi_2nd = 0
        self._kpi_fp_giveup = 0
        self._kpi_fp_resid = 0
        self._kpi_prefetched = 0

    def kpi_fp(self, replays: int, residual: int) -> None:
        """Runner reports the step's fixed-point outcome: total replay
        passes and the residual max-miss after the loop. residual==0 (or
        within tol) = clean fixed point. residual > FP_THRESH = the
        adaptive break: working set moving, second-order residue accepted
        after the mandatory first-order restore (expected on fresh prose;
        KPI-counted, not a warning). residual in (tol, FP_THRESH] = the
        loop hit FP_MAX while within closing distance — pathological
        ping-pong, logged loudly."""
        if replays > 1:
            self._kpi_2nd += replays - 1
        if residual <= base_miss_tol():
            return
        self._kpi_fp_giveup += 1
        self._kpi_fp_resid += residual
        if residual <= fp_thresh():
            self._kpi_unfixed += 1
            logger.warning(
                "moe_w2 [%s]: fixed-point replay hit FP_MAX with %d misses "
                "remaining (within thresh) — ping-pong; step kept zeroed "
                "contributions.", self._tag, residual)

    # ---- observability ---------------------------------------------------

    def precision_of(self, layer: int, expert: int) -> str:
        """Live tier of one expert: 'fp4' (delta-cached) or '2bit' (base)."""
        return "fp4" if int(self._mirror[layer, expert]) >= 0 else "2bit"

    def precision_map(self) -> dict:
        """{layer: [expert ids currently in FP4]}. Anything not listed is on
        the resident 2-bit base — i.e. the live precision of every expert."""
        out = {}
        cov = self._mirror >= 0
        for li in range(self.n_layers):
            ex = cov[li].nonzero().flatten().tolist()
            if ex:
                out[li] = ex
        return out

    def _log_summary(self):
        cov = self._mirror >= 0
        cached = int(cov.sum())
        # Under pipeline parallelism this rank hosts only ITS layers (local
        # layer_keys); normalize coverage by the layers actually staged here
        # (len(self._store)) rather than the full slot_table (n_layers*E), so
        # the reported %experts is honest per-rank. On TP/1-GPU every layer is
        # hosted on each rank -> len(self._store) == n_layers -> unchanged.
        total = max(len(self._store), 1) * self.E
        hr = 100.0 * self._win_hits / max(self._win_active, 1.0)
        hrd = 100.0 * self._win_hits_d / max(self._win_active_d, 1)
        logger.info(
            "[%s] tick %d: %d/%d slots, covering %d/%d experts (%.1f%%); "
            "hit-rate %.1f%% tokens / %.1f%% experts; window +%d/-%d, cumulative +%d/-%d",
            self._tag, self._tick, cached, self.n_slots, cached, total,
            100.0 * cached / max(total, 1), hr, hrd, self._win_promoted,
            self._win_evicted, self._n_promoted, self._n_evicted)
        per_layer = cov.sum(dim=1).tolist()
        hist = " ".join(f"L{li}:{int(c)}" for li, c in enumerate(per_layer) if c)
        if hist:
            logger.info("[%s] experts per layer: %s", self._tag, hist)
        if hasattr(self._store, "stats"):
            st = self._store.stats()
            if "arena_slots" in st:
                # tiered backend: the fetch split ram-hit vs NVMe is the
                # arena-coverage curve — the whole point of the tier.
                tot = max(st["hit_rows"] + st["miss_rows"], 1)
                logger.info(
                    "[%s] tiered store: arena %d/%d slots | fetch rows "
                    "%d ram + %d nvme (%.1f%% ram) | nvme %.2f GiB | "
                    "call ms p50/p99: hit %.2f/%.2f, miss %.1f/%.1f",
                    self._tag, st["arena_used"], st["arena_slots"],
                    st["hit_rows"], st["miss_rows"],
                    100.0 * st["hit_rows"] / tot,
                    st["miss_bytes"] / 2**30,
                    st["hit_p50_ms"], st["hit_p99_ms"],
                    st["miss_p50_ms"], st["miss_p99_ms"])
            else:
                logger.info("[%s] pack store: %d row reads, %.2f GiB total",
                            self._tag, st["reads"], st["read_bytes"] / 2**30)
        # CONCENTRATION study: compare how top-heavy low-confidence routing (_need,
        # from the gate via mark_need_only) is vs overall routing (_freq). If the
        # top few % of experts hold MOST of the _need mass while _freq is spread,
        # 2-bit difficulty concentrates -> a small persistent FP4 set suffices. If
        # _need is as spread as _freq, difficulty is context-driven (no small set).
        nd = self._need.flatten()
        if float(nd.sum()) > 0:
            fr = self._freq.flatten()

            def topmass(v, p):
                vs = torch.sort(v, descending=True).values
                k = max(1, int(vs.numel() * p))
                return 100.0 * float(vs[:k].sum()) / max(float(v.sum()), 1e-9)
            logger.info(
                "[need] low-conf routing top1%%/5%%/10%% = %.0f/%.0f/%.0f  |  "
                "all routing top1%%/5%%/10%% = %.0f/%.0f/%.0f  |  experts need>0: %d/%d",
                topmass(nd, .01), topmass(nd, .05), topmass(nd, .10),
                topmass(fr, .01), topmass(fr, .05), topmass(fr, .10),
                int((nd > 0).sum()), nd.numel())
        self._win_promoted = self._win_evicted = 0
        self._win_hits = self._win_active = 0.0
        self._win_hits_d = self._win_active_d = 0
        if _DUMP_PATH:
            self._dump(_DUMP_PATH)

    def _dump(self, path: str):
        import json
        snap = dict(tick=self._tick, n_slots=self.n_slots,
                    cached=int((self._mirror >= 0).sum()),
                    promoted_total=self._n_promoted,
                    evicted_total=self._n_evicted,
                    fp4_by_layer=self.precision_map())
        if hasattr(self._store, "stats"):
            snap["store"] = self._store.stats()
        try:  # atomic write so a tail/watcher never reads a half file
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(snap, f)
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001 - observability must not kill serving
            logger.warning("delta dump to %s failed: %s", path, e)

    def _dump_capture(self, final=False):
        import numpy as np
        rows = []
        for tk, fr in self._cap_frames:
            a = fr.numpy()
            if a.size == 0:
                continue
            idx = np.full((a.shape[0], 1), tk, dtype=np.int32)
            rows.append(np.hstack([idx, a.astype(np.int32)]))
        arr = np.vstack(rows) if rows else np.zeros((0, 3), np.int32)
        try:
            np.save(_CAPTURE, arr)
            logger.info("delta capture: %d frames, %d activations -> %s%s",
                        len(self._cap_frames), arr.shape[0], _CAPTURE,
                        " (final)" if final else "")
        except Exception as e:  # noqa: BLE001 - capture must not kill serving
            logger.warning("delta capture save failed: %s", e)
        if final:
            self._cap_done = True
            self._cap_frames = []


def mark_seen(seen_row, ids):
    """Record routed experts into a layer's seen row from the forward. Token
    COUNTS when observability is on (token-weighted hit-rate / capture), else a
    cheap binary flag. `ids` = flattened topk_ids (int64). Graph-capture-safe."""
    if _COUNT:
        seen_row.index_add_(0, ids, torch.ones_like(ids, dtype=seen_row.dtype))
    else:
        seen_row.index_fill_(0, ids, 1)


_TIER: DeltaTier | None = None

# ---------------------------------------------------------------------------
# BASE cache (inverted delta): the 2-bit BASE planes live in pinned host RAM
# and the GPU holds only a cache of hot experts — for models whose 2-bit
# planes alone exceed VRAM (GLM-5.2 on 2 GPUs: ~189 GiB of planes vs 192 GB).
# Reuses the DeltaTier machinery wholesale (pool, slot table read in-graph,
# manager prefetch, batched eviction); slot CONTENT differs (2-bit codes +
# UE8M0 scales, four sections per expert) and a miss cannot be served by any
# resident fallback — the desc kernel zeroes the pair and bumps a miss
# counter, and the runner re-runs the step after a synchronous fetch.
#
# The FP4 delta tier CAN coexist with the base cache (explicit opt-in:
# VLLM_MOE_W2_DELTA_GB=<GiB> set in the environment; "auto" unsupported
# here). It then acts as the quality-recovery tier for host-resident bases:
# a small gate-filled ("need" policy) FP4 pool whose slots carry their OWN
# block-32 scales ([fp4_13|sc13|fp4_2|sc2] — with the base host-resident
# there are no GPU-resident scale planes to share). The desc kernel reads
# BOTH slot tables with priority FP4 > 2-bit slot > miss; each tier has its
# own `seen` tensor (the forward marks both), own manager, own policy.
_BASE_GB = float(os.getenv("VLLM_MOE_W2_BASE_CACHE_GB", "0"))
_BASE_TIER: DeltaTier | None = None

# Miss tolerance: a decode step with <= TOL missing routed (layer, expert)
# pairs keeps its logits (the missing pairs contributed zero) instead of
# replaying the graph. Rationale: at 99.9% token hit-rate a 600-pair step
# still has a ~45% chance of >=1 miss, and mandatory replays collapsed
# GLM-5.2 TP4 from 56.7 to 18.2 tok/s at 74% coverage — while dropping k of
# ~600 weighted expert contributions is the same approximation class the
# FP4 delta/gate already trades in. Missing experts are STILL fetched (they
# join the pool for subsequent steps). 0 = strict (always replay).
# The _FILE variant is mtime-cached and re-read on change, so a tolerance
# sweep runs in ONE server (same idiom as the gate's TAU_FILE).
_BASE_MISS_TOL = int(os.getenv("VLLM_MOE_W2_BASE_MISS_TOL", "0"))
_BASE_MISS_TOL_FILE = os.getenv("VLLM_MOE_W2_BASE_MISS_TOL_FILE", "")
_base_tol_dyn = _BASE_MISS_TOL
_base_tol_mtime = -1.0


def base_miss_tol() -> int:
    global _base_tol_dyn, _base_tol_mtime
    if not _BASE_MISS_TOL_FILE:
        return _BASE_MISS_TOL
    try:
        m = os.path.getmtime(_BASE_MISS_TOL_FILE)
        if m != _base_tol_mtime:
            _base_tol_mtime = m
            with open(_BASE_MISS_TOL_FILE) as f:
                _base_tol_dyn = int(f.read().strip())
            logger.info("moe_w2 base cache: miss tolerance -> %d",
                        _base_tol_dyn)
    except (OSError, ValueError):
        pass
    return _base_tol_dyn


# Adaptive fixed-point replay policy. Pass 1 is MANDATORY above the
# tolerance band (it restores every first-order miss — the bulk of the
# quality). Further passes chase SECOND-ORDER misses (corrected layers
# re-routing onto unfetched experts) and only run while the step is within
# FP_THRESH of miss-free: that yields strict, bit-deterministic fixed
# points for extra passes, while on shifting content (fresh prose: 100+
# missing pairs, residue stays large) the loop stops at 2 forwards/step —
# the old single-replay cost — instead of burning 3x+ chasing a moving
# target. The accepted residue is second-order only, small, and
# KPI-visible ("fp-residue"). DEFAULT 0 = mandatory pass only: measured
# A/B (GLM TP2 46 GiB pool, runtime thresh flip, same server, fox 384
# med 5): thresh=16 -> 19.2 tok/s, thresh=0 -> 28.3 at unchanged quality
# probes (needle 36K PASS, arithmetic ok) — the second-order chase cost
# ~35% decode for residue the pre-fixed-point stack silently kept
# anyway, plus first-order restoration on top. DS4 1x5090 14 GiB:
# thresh-insensitive (31.2 vs 31.4 — residue >16 either way). Raise for
# bit-determinism studies on converged working sets. THRESH is
# file-tunable at runtime (sweep idiom of TAU/TOL); FP_MAX bounds
# pathological ping-pong.
_FP_MAX = int(os.getenv("VLLM_MOE_W2_FP_MAX", "8"))
_FP_THRESH = int(os.getenv("VLLM_MOE_W2_FP_THRESH", "0"))
_FP_THRESH_FILE = os.getenv("VLLM_MOE_W2_FP_THRESH_FILE", "")
_fp_thresh_dyn = _FP_THRESH
_fp_thresh_mtime = -1.0


def fp_thresh() -> int:
    global _fp_thresh_dyn, _fp_thresh_mtime
    if not _FP_THRESH_FILE:
        return _FP_THRESH
    try:
        m = os.path.getmtime(_FP_THRESH_FILE)
        if m != _fp_thresh_mtime:
            _fp_thresh_mtime = m
            with open(_FP_THRESH_FILE) as f:
                _fp_thresh_dyn = int(f.read().strip())
            logger.info("moe_w2 base cache: fixed-point thresh -> %d",
                        _fp_thresh_dyn)
    except (OSError, ValueError):
        pass
    return _fp_thresh_dyn


def fp_continue(passes: int, max_miss: int) -> bool:
    """Should the runner run another replay pass? (adaptive policy above)"""
    if max_miss <= base_miss_tol():
        return False            # inside the tolerance band (or miss-free)
    if passes == 0:
        return True             # first-order restore is mandatory
    if passes >= _FP_MAX:
        return False            # hard bound on ping-pong
    return max_miss <= fp_thresh()


# Base-cache KPI cadence: every N runner steps log the per-STEP replay rate,
# avg missing pairs/step and pool coverage. Always on (INFO, one line per
# window) because pool sizing is the dominant base-cache perf knob and the
# per-step replay rate is the number that actually predicts tok/s — token
# hit-rate hides it (measured on DS4 1x5090: 96.5% token hit = replay almost
# every step = 32.7 tok/s; 98.8% = large zero-miss fraction = 43.4 tok/s,
# +33% from 3 GiB of pool). 0 disables the log (counters still kept).
_KPI_EVERY = int(os.getenv("VLLM_MOE_W2_KPI_EVERY", "500"))

# Was VLLM_MOE_W2_DELTA_GB set explicitly (vs the "2.0" default)? Coexistence
# with the base cache must be opt-in: the historical base-cache configs never
# set DELTA_GB and must not silently grow an FP4 pool out of the default.
_GB_EXPLICIT = "VLLM_MOE_W2_DELTA_GB" in os.environ

# SPLIT FP4 (VLLM_MOE_W2_DELTA_SPLIT=1, default off): the delta tier stores
# RADIX-5 QUINTAL planes (2.5 bits/elem) instead of full e2m1 nibble planes
# and dispatches moe_w4q_mm, which reads them alongside the 2-bit base and
# reconstructs e2m1 BIT-EXACTLY — all 16 nibbles, zeros included (see
# moe_w2_planes.pack_quintal_fragment_major; the historical 2-bit
# refinement merged mag 0 into 0.5 and measurably decayed GSM8K with pool
# coverage — internal/SPLIT_FP4_ZERO_LOSS_NEXT_SESSION.md). 5/8 of the
# nibble bytes per expert — 1.6x experts/GiB
# (over the base cache: 1.7x — the quintal slot also drops the private
# scale sections, the base slot's serve both GEMMs) at ~+11/6/6% kernel
# time vs moe_w4s at K=4096/2048/512 (decode ALU; slot read bytes 10 B vs
# 8 B per lane-k64).
#
# GPU-resident-base configs read the resident planes directly. Over the
# BASE CACHE the base codes+scales come from the base tier's pool slot, so
# split serving is RESIDENCY-COUPLED: the desc kernel routes a pair to w4q
# only when the expert is resident in BOTH slot tables (FP4-mapped but
# base-missing counts as a miss -> the standard fetch+replay restores it),
# and the base tier's eviction hard-excludes experts mapped in the FP4
# tier (_coupled_fp4) so a mapped quintal row never outlives its base row.
_SPLIT = os.getenv("VLLM_MOE_W2_DELTA_SPLIT", "0") == "1"


def split_enabled() -> bool:
    return _SPLIT


def enabled() -> bool:
    if os.getenv("VLLM_MOE_W2_DELTA", "1") != "1":
        return False
    if base_enabled():
        # FP4 need-pool OVER the base cache: explicit GiB only (auto's
        # after-KV sizing belongs to the base pool math, not this tier).
        return _GB_EXPLICIT and _GB > 0
    return _GB > 0 or _AUTO


def base_enabled() -> bool:
    return _BASE_GB > 0


def check_pool_floor(top_k: int, n_spec: int, max_num_seqs: int) -> None:
    """Boot-time guard: refuse to serve when a pool is sized below its
    WORKING-SET floor (VLLM_MOE_W2_FORCE_POOL=1 downgrades to a warning).

    BASE cache: a pool smaller than the per-step routed working set cannot
    hold a step's experts even with perfect eviction — the emergency path
    then leaves routed experts ZEROED (silent quality corruption; measured
    on DS4+MTP2: 11 GiB pool -> corrupted math, 14 GiB -> clean). Floor:
        pairs = moe_layers x top_k x (1 + n_spec) x sqrt(min(seqs, 4))
    (sqrt: concurrent decodes overlap heavily on hot experts). HARD floor
    1.15x pairs (below the measured-bad anchor stays below it), comfort
    1.40x (the measured-good anchor stays above it).

    FP4 need-pool (gate): pool < one step's routed set means a fire can
    NEVER cover its own step (partial upgrades regardless of
    GATE_MAX_PROMOTE) — degraded recovery, not corruption: WARN only.
    """
    import math
    force = os.getenv("VLLM_MOE_W2_FORCE_POOL", "0") == "1"
    seq_f = math.sqrt(float(min(max(max_num_seqs, 1), 4)))
    t = _BASE_TIER
    if t is not None and len(t._store) > 0:
        pairs = int(len(t._store) * top_k * (1 + n_spec) * seq_f)
        hard = int(1.15 * pairs)
        comfort = int(1.40 * pairs)
        gib = t.n_slots * t.slot_bytes / 2**30
        need_gib = hard * t.slot_bytes / 2**30
        if t.n_slots < hard:
            msg = (
                f"moe_w2 BASE cache pool is BELOW the working-set floor: "
                f"{t.n_slots} slots ({gib:.1f} GiB) < {hard} required "
                f"(~{need_gib:.1f} GiB) for {len(t._store)} MoE layers x "
                f"top-{top_k} x (1+{n_spec} spec) x {max_num_seqs} seqs. "
                f"Steps will keep ZEROED expert contributions (silent "
                f"quality corruption). Raise VLLM_MOE_W2_BASE_CACHE_GB, "
                f"reduce num_speculative_tokens/max_num_seqs, or set "
                f"VLLM_MOE_W2_FORCE_POOL=1 to serve anyway.")
            if not force:
                raise ValueError(msg)
            logger.warning("%s (FORCED past the check)", msg)
        elif t.n_slots < comfort:
            logger.warning(
                "moe_w2 BASE cache pool is tight: %d slots (%.1f GiB) < "
                "comfort %d for this step working set — expect UNRESTORED "
                "bursts at prefill->decode transitions (watch the KPI "
                "line).", t.n_slots, gib, comfort)
        else:
            logger.info(
                "moe_w2 BASE cache pool floor check OK: %d slots (%.1f "
                "GiB) >= comfort %d (step working set %d pairs).",
                t.n_slots, gib, comfort, pairs)
    d = _TIER
    if d is not None and gate_armed() and len(d._store) > 0:
        # FIRE FLOOR (hard): the fire contract - "this step was uncertain,
        # re-decide it with its routed set upgraded" - requires the pool to
        # hold the step's routed union ACROSS ALL of the fire's fixed-point
        # iterations (pins persist through the step, so every pass's hits
        # and fetches stay unevictable while later passes promote the
        # re-routed remainder). Components:
        #   - one seq's union: moe_layers x top_k x (1 + n_spec);
        #   - seqs: LINEAR (worst case: independent prompts share nothing —
        #     measured on GPQA C=2: demand 2590 slots vs sqrt-scaled
        #     estimate 1899, pool 2048 fully pinned + 542 denied);
        #   - fixed-point growth: each extra replay re-routes onto new
        #     experts; measured ~0.32 x union per iteration on GPQA ->
        #     factor 1 + 0.35 x (GATE_FP_MAX - 1).
        # Below the floor a fire can only partially upgrade the very token
        # it exists to fix, and tau->1 does NOT converge: structurally
        # broken, not merely slow (measured: sub-floor pools 94.5% /
        # +110% token inflation vs 96.5-97.5% above; 12 GiB at C=2 GPQA
        # 66.2-67.7% vs 72.2% for a floor-satisfying 24 GiB).
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_gate)
        fp_growth = 1.0 + 0.35 * max(moe_w2_gate.fire_fp_max() - 1, 0)
        fire_floor = int(len(d._store) * top_k * (1 + n_spec)
                         * min(max(max_num_seqs, 1), 4) * fp_growth)
        if d.n_slots < fire_floor:
            # Sub-floor pools serve in DEGRADED mode: a fire can only
            # PARTIALLY upgrade its step, and the measured-sound operating
            # point is the INCREMENTAL cap (GATE_MAX_PROMOTE default 64) —
            # uncapped sub-floor churns the pool wholesale (measured 94.5%
            # / +110% tokens; GPQA 60.6% vs 70.2% capped, McNemar p=0.002).
            # The flag below is READ BY NOTHING yet: the eager layer-wise
            # fire (full contract at any pool size) is a SPEC, not a
            # feature — see internal/EAGER_FIRE_NEXT_SESSION.md. The
            # warning must not promise it (review finding 2.4).
            d._sub_floor = True
            logger.warning(
                "moe_w2 FP4 need-pool is below the FIRE FLOOR: %d slots "
                "< %d (%d MoE layers x top-%d x (1+%d spec) x %d seqs x "
                "%.2f FP-growth = ~%.1f GiB). Serving continues in "
                "DEGRADED mode: gate fires can only PARTIALLY upgrade "
                "their step; promotions stay capped by GATE_MAX_PROMOTE "
                "(default 64 = the measured-best sub-floor operating "
                "point; do NOT set 0/unlimited here — measured churn "
                "regression). tau->1 does NOT converge to native on a "
                "sub-floor pool. For full graph-mode fires raise "
                "VLLM_MOE_W2_DELTA_GB or reduce num_speculative_tokens/"
                "max_num_seqs.",
                d.n_slots, fire_floor, len(d._store), top_k, n_spec,
                max_num_seqs, fp_growth,
                fire_floor * d.slot_bytes / 2**30)

    # ---- config-coherence guards (review finding 2.3): contradictory
    # env combinations used to fail silently or pay dead per-step costs.
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_cubit, moe_w2_gate)
    if moe_w2_gate.enabled() and _TIER is None:
        # gate without a delta tier: the decision (topk + logsumexp +
        # .item() sync) would run EVERY step and force_promote would
        # no-op forever — pure hot-path waste. Disable loudly.
        moe_w2_gate.disable(
            "VLLM_MOE_W2_GATE=1 but no FP4 delta tier exists "
            "(VLLM_MOE_W2_DELTA_GB unset/0) — the gate cannot promote "
            "anything; disabling to avoid a dead per-step sync.")
    decode_cap = max(max_num_seqs, 1) * (1 + n_spec)
    if moe_w2_cubit._PREFILL_T < decode_cap:
        raise ValueError(
            f"VLLM_MOE_W2_PREFILL_T={moe_w2_cubit._PREFILL_T} is below "
            f"the largest decode step of this config (max_num_seqs x "
            f"(1 + num_speculative_tokens) = {decode_cap}): captured "
            "decode graphs would bake the PREFILL code path (wrong tier "
            "machinery inside full-graph replays). Raise PREFILL_T to "
            f">= {decode_cap} (default 96) or shrink the config.")
    try:
        from vllm.config import get_current_vllm_config
        pc = get_current_vllm_config().parallel_config
        if _BASE_TIER is not None and (
                getattr(pc, "use_ubatching", False)
                or getattr(pc, "ubatch_size", 0)):
            raise ValueError(
                "BASE cache x micro-batching (ubatch/DBO) is unsupported: "
                "the in-graph miss counter is zeroed at layer 0 of EVERY "
                "forward, so the second ubatch erases the first ubatch's "
                "misses -> the runner skips the restore replay and serves "
                "ZEROED expert contributions. Disable ubatching or the "
                "base cache.")
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 - config introspection best-effort
        pass


def gate_armed() -> bool:
    from vllm.model_executor.layers.quantization.utils import moe_w2_gate
    return moe_w2_gate.enabled()


def spec_suppressed() -> bool:
    """Spec-guard latch (VLLM_MOE_W2_SPEC_GUARD): True while the base pool
    is too cold for speculation to pay — the runner then skips scheduling
    drafts. Always False when the guard or the base cache is off."""
    t = _BASE_TIER
    return t is not None and t._spec_suppressed


def wake_all() -> None:
    """Step-boundary broadcast: nudge every live tier's manager (base +
    fp4/delta) to run one pass. Called once per decode step from
    step_begin (base-cache configs) and the gate decision (delta-only
    configs); lock-free, coalescing, safe before tiers exist."""
    t = _BASE_TIER
    if t is not None:
        t.wake()
    t = _TIER
    if t is not None:
        t.wake()


def get_base_tier(n_layers: int, n_experts: int, dev,
                  w13_bytes: int, w2_bytes: int) -> DeltaTier:
    """Base-cache tier singleton. `w13_bytes`/`w2_bytes` are the PACKED 2-bit
    sections per expert (codes13+sc13 / codes2+sc2), so slot_bytes matches the
    host rows staged by the plane builder. The pool is allocated immediately
    (explicit env sizing, no auto-defer) and the manager starts prefetching
    as soon as host planes exist."""
    global _BASE_TIER
    if _BASE_TIER is None:
        _BASE_TIER = DeltaTier(n_layers, n_experts, dev,
                               w13_bytes=w13_bytes, w2_bytes=w2_bytes,
                               pool_gb=_BASE_GB, tag="base")
        # decode misses counted by the desc kernel (atomic, in-graph); zeroed
        # in-graph at the first layer of every forward, read by the runner
        # after logits to decide the fetch+replay.
        _BASE_TIER.miss_count = torch.zeros(1, dtype=torch.int32,
                                            device=_BASE_TIER.dev)
        from vllm.model_executor.layers.quantization.utils import moe_w2_looka
        if (os.getenv("VLLM_MOE_W2_PREFETCH", "0") == "1"
                or moe_w2_looka.wants_route_log()):
            # in-graph routing log: the draft-affinity prefetcher folds it
            # into the token->experts table, and LOOKA's predictor-[0]
            # baseline scores it (previous step's routing per layer).
            _BASE_TIER.route_log = torch.zeros(
                n_layers, int(os.getenv("VLLM_MOE_W2_PREFETCH_TCAP", "8")),
                16, dtype=torch.int32, device=_BASE_TIER.dev)
            logger.info("moe_w2 BASE cache: route_log armed %s (prefetch=%s"
                        ", looka/pilot=%s)", tuple(_BASE_TIER.route_log.shape),
                        os.getenv("VLLM_MOE_W2_PREFETCH", "0"),
                        moe_w2_looka.wants_route_log())
        # Mechanical minimum only (one prefill layer's worst case): the
        # CALIBRATED working-set floor (real top_k/spec/seqs) is verdicted
        # by check_pool_floor at end-of-load — the old layers*16 heuristic
        # here double-floored valid configs (e.g. DS4 no-spec single-seq:
        # calibrated floor 304 slots, heuristic demanded 704).
        assert _BASE_TIER.n_slots >= 2 * n_experts, (
            f"moe_w2 base cache: pool of {_BASE_TIER.n_slots} slots cannot "
            f"hold even one prefill layer (2*E={2 * n_experts}); raise "
            f"VLLM_MOE_W2_BASE_CACHE_GB")
        _BASE_TIER.start()
        cov = 100.0 * _BASE_TIER.n_slots / (n_layers * n_experts)
        logger.info("moe_w2 BASE cache: %d slots x %.2f MiB (%.1f GiB pool) — "
                    "2-bit base is HOST-resident; pool covers %.1f%% of "
                    "%d experts. POOL SIZE IS THE DOMINANT PERF KNOB "
                    "(replays are per-step: DS4 15%%->19%% coverage measured "
                    "+33%% decode) — watch the '[base] KPI' line.",
                    _BASE_TIER.n_slots, _BASE_TIER.slot_bytes / 2**20,
                    _BASE_TIER.n_slots * _BASE_TIER.slot_bytes / 2**30,
                    cov, n_layers * n_experts)
    return _BASE_TIER


def get_tier(n_layers=None, n_experts=256, dev=None,
             w13_bytes=None, w2_bytes=None) -> DeltaTier | None:
    global _TIER
    if not enabled():
        return None
    if _TIER is None:
        if n_layers is None:
            # one slot-table row per built layer_key: the main stack and,
            # when the cutoff includes it, the MTP drafter MoE
            from vllm.model_executor.layers.quantization.utils import (
                moe_w2_cubit)
            n_layers = moe_w2_cubit._layer_cutoff() + 1
        # The plane builder passes the per-rank FP4 plane sizes (smaller under
        # TP); fall back to the TP1 module constants when unspecified.
        # Over the base cache the tier defaults to the "need" policy: the pool
        # is a QUALITY tier filled only by the confidence gate — a freq-filled
        # pool would duplicate the base tier's hot set at 2x the read bytes.
        # An explicit VLLM_MOE_W2_DELTA_POLICY still wins.
        policy = None
        if base_enabled():
            policy = os.getenv("VLLM_MOE_W2_DELTA_POLICY", "need")
        # Split quintal slots have a DIFFERENT geometry than full-FP4
        # slots; a distinct pack tag ("fp4q"; the superseded 2-bit
        # refinement used "fp4s") keeps the modes' pack files apart — a
        # shared pack dir otherwise ping-pong-rebuilds one file under the
        # other config's feet (the page-cache store reads it live at
        # serve time -> garbage FP4 rows on BOTH; measured).
        fp4_tag = "fp4q" if split_enabled() else "fp4"
        _TIER = DeltaTier(
            n_layers, n_experts, dev or torch.device("cuda"),
            w13_bytes=W13_BYTES if w13_bytes is None else w13_bytes,
            w2_bytes=W2_BYTES if w2_bytes is None else w2_bytes,
            policy=policy, tag=fp4_tag if base_enabled() else "delta",
            host_pinned=not base_enabled())
        if base_enabled() and split_enabled() and _BASE_TIER is not None:
            # residency coupling: the base tier must not evict slots the
            # FP4 tier's refinement rows are mapped against
            _BASE_TIER._coupled_fp4 = _TIER
            logger.info("moe_w2 delta: split-FP4 residency coupling armed "
                        "(base evictions exclude FP4-mapped experts)")
        # Start the background manager as soon as the tier exists. It idles until
        # experts are actually routed (seen empty -> early return) and only
        # promotes layers whose host planes are already staged, so an early start
        # is safe. This fires correctly under PIPELINE PARALLELISM, where
        # layer_keys are LOCAL per rank and never reach NUM_LAYERS-1 -> the old
        # "start on the last layer built" trigger never ran and the tier sat
        # inactive (pool allocated but no promotions).
        _TIER.start()
    return _TIER
