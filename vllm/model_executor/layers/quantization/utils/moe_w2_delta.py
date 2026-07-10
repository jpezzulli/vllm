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
    stream, capped per tick), evicts only experts cold for >= 2 ticks.

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
_TICK_S = float(os.getenv("VLLM_MOE_W2_DELTA_TICK_MS", "5")) / 1e3

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
_DECAY_TICKS = max(int(os.getenv("VLLM_MOE_W2_DELTA_DECAY_TICKS", "1000")), 1)

# Token-weighted hit-rate: when observability is on, the forward records per-expert
# routing COUNTS (not a binary flag) so the logged hit-rate reflects the fraction
# of token->expert ROUTINGS served at FP4 — the honest number. A binary-flag
# hit-rate under-counts badly, because the cached hot experts absorb
# disproportionately many tokens (a one-token expert and a 500-token expert count
# the same under a flag). Off by default -> the prod serving path is unchanged.
_COUNT = (_TRACE > 0) or bool(_CAPTURE)

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
        # slot -> (layer, expert, last_seen_tick); -1 layer = free
        self._owner = [(-1, -1, 0)] * self.n_slots
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
        # per-step KPI counters (window + cumulative), fed by kpi_step()
        self._kpi_steps = 0
        self._kpi_miss_pairs = 0
        self._kpi_replays = 0
        self._kpi_c_steps = 0
        self._kpi_c_replays = 0
        self._kpi_unfixed = 0     # experts replay could NOT restore (window)
        self._kpi_2nd = 0         # extra replays for second-order misses
        # Slots touched since step_begin(): promoted or hit by any pass of
        # the CURRENT step. Never evictable (even in the emergency pass) —
        # without this, a fixed-point iteration can evict pass-k's fetches
        # to serve pass-k+1 (their seen marks are zeroed after each
        # snapshot) and ping-pong past the replay cap.
        self._step_pins: set[int] = set()
        # Draft-affinity prefetch (VLLM_MOE_W2_PREFETCH=1, base tier only):
        # route_log = in-graph [n_layers, T_cap, K_cap] routing log written
        # by the forward glue; _aff = token->experts affinity table folded
        # from it post-step; draft_prefetch() predicts+fetches at step start.
        self.route_log: torch.Tensor | None = None
        self._aff: torch.Tensor | None = None
        self._aff_k = 8
        self._last_ids: torch.Tensor | None = None
        self._kpi_prefetched = 0
        # recency-decayed routing frequency per expert (drives the freq policy)
        self._freq = torch.zeros(n_layers, n_experts, dtype=torch.float32)
        # NEED signal (gate-driven policy): how often the confidence gate flagged a
        # step routing to this expert (i.e. 2-bit was insufficient). Recency-decayed
        # like _freq; the eviction key under _POLICY == "need".
        self._need = torch.zeros(n_layers, n_experts, dtype=torch.float32)
        self._last_decay = 0
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
        self._owner = [(-1, -1, 0)] * self.n_slots
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
        while not self._stop:
            try:
                torch.cuda.set_device(self.dev)
                self._tick_once()
            except Exception as e:  # noqa: BLE001 - never kill serving
                logger.warning("delta tick failed: %s", e)
                time.sleep(1.0)
            time.sleep(_TICK_S)

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
            # recency-decayed routing frequency (the hotness signal)
            self._freq[seen[:, 0], seen[:, 1]] += 1.0
            # refresh last_seen for cached owners; collect promotion candidates
            cand = []
            seen_set = set()
            for li, ei in seen.tolist():
                seen_set.add((li, ei))
                s = int(self._mirror[li, ei])
                if s >= 0:
                    la, ex, _ = self._owner[s]
                    self._owner[s] = (la, ex, self._tick)
                elif li in self._store:
                    cand.append((li, ei))
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
                promoted = 0
                for li, ei in cand:
                    if promoted >= _PROMOTE_PER_TICK:
                        break
                    slot = self._take_slot(seen_set)
                    if slot is None:
                        break
                    self._promote(li, ei, slot)
                    promoted += 1
            if self._tick - self._last_decay >= _DECAY_TICKS:
                self._freq *= _DECAY  # keep the frequency signal recent + bounded
                self._need *= _DECAY  # need decays too -> tracks RECENT 2-bit misses
                self._last_decay = self._tick
        # reset flags for the next window (racy with the forward's scatter
        # of ones — a lost flag only delays promotion by one tick)
        if self._tick % 4 == 0:
            self.seen.zero_()
        if _TRACE and self._tick - self._last_summary_tick >= _TRACE_EVERY:
            self._log_summary()
            self._last_summary_tick = self._tick

    def _take_slots_batch(self, k: int, emergency: bool = False) -> list[int]:
        """Take up to k slots (lock held by caller): free list first, then ONE
        vectorized eviction pass over all slots. Replaces the old per-slot
        python scan per promotion — O(n_slots) per TAKEN slot — which at GLM
        scale (4k slots x hundreds of gate promotions per fire) burned seconds
        of GIL time per fired step and starved the forward thread.

        `emergency=True` (synchronous runner-thread callers only — never the
        background manager) adds a second eviction pass that relaxes the
        2-tick coldness bound when the first pass cannot cover k, keeping
        the seen-window exclusion. See the pass comments below.

        Eviction policy is unchanged: least-valuable slot by _POLICY key
        (need / freq / lru), restricted to slots whose owner is not active in
        the current seen window (read directly from the _seen_host snapshot —
        the call sites always passed a set built from exactly that) and cold
        >= 2 ticks, so in-flight graph reads never hit a rewritten slot.
        Victims are unmapped here (graphs stop dispatching w4 before bytes
        change); the caller reserves _owner for each returned slot."""
        out: list[int] = []
        while self._free and len(out) < k:
            out.append(self._free.pop())
        k_evict = k - len(out)
        if k_evict <= 0:
            return out
        li = torch.tensor([o[0] for o in self._owner], dtype=torch.long)
        ei = torch.tensor([o[1] for o in self._owner], dtype=torch.long)
        tk = torch.tensor([o[2] for o in self._owner], dtype=torch.float64)
        lic, eic = li.clamp(min=0), ei.clamp(min=0)
        if self._policy == "need":
            key = self._need[lic, eic].double()
        elif self._policy == "freq":
            key = self._freq[lic, eic].double()
        else:
            key = tk.clone()
        # Hard exclusions: free markers, owners active in the current seen
        # window (their slots may be read by this step's graph/replay), and
        # step-pinned slots (touched by any pass of the current step).
        blocked = (li < 0) | self._seen_host[lic, eic].to(torch.bool)
        if self._step_pins:
            blocked[list(self._step_pins)] = True
        # Pass 1: only >=2-tick-cold victims (never disturbs slots a
        # CONCURRENT in-flight graph might still read — the background
        # manager's constraint). Pass 2 (emergency): the synchronous callers
        # (force_promote / ensure_resident, runner thread, no forward in
        # flight) relax the coldness bound rather than leave a missing
        # expert UNRESTORED — a replay that keeps zeroed contributions is a
        # silent quality hit and a nondeterminism source, strictly worse
        # than evicting a warm-but-idle slot.
        passes = [blocked | ((self._tick - tk) < 2)]
        if emergency:
            passes.append(blocked)
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
            victims = torch.topk(kk, take, largest=False).indices.tolist()
            for s in victims:
                vli, vei, vt = self._owner[s]
                self.slot_table[vli, vei] = -1
                self._mirror[vli, vei] = -1
                self._n_evicted += 1
                self._win_evicted += 1
                if _TRACE >= 2:
                    logger.info(
                        "[%s] evict   L%-2d E%-3d  slot %-4d (cold %d ticks)",
                        self._tag, vli, vei, s, self._tick - vt)
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
        self.slot_table[li, ei] = slot
        self._mirror[li, ei] = slot
        self._owner[slot] = (li, ei, self._tick)
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
        replay's passes must never cannibalize each other's fetches."""
        with self._lock:
            self._step_pins.clear()

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
        with self._lock:
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
            if not pairs:
                return 0
            slots = self._take_slots_batch(len(pairs))   # non-emergency
            plan = [(p, s) for p, s in zip(pairs, slots)]
            if not plan:
                return 0
            rows = self._store.rows_for([p for p, _ in plan])
            for ((li, ei), slot), row in zip(plan, rows):
                self._owner[slot] = (li, ei, self._tick)
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

    def force_promote(self, layers=None, max_promote=None) -> int:
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
            seen_set = set()
            cand = []
            for li, ei in seen.tolist():
                seen_set.add((li, ei))
                # Refresh last_seen for CACHED owners routed this window
                # (mirrors _tick_once). force_promote zeroes `seen` after its
                # snapshot, so a manager tick racing this step would otherwise
                # see an empty window, find these slots "cold" (stale tick),
                # evict + rewrite one WHILE the imminent replay reads it —
                # measured as rare cross-request greedy nondeterminism. The
                # tick refresh keeps them coldness-protected for >=2 ticks.
                s = int(self._mirror[li, ei])
                if s >= 0:
                    la, ex, _ = self._owner[s]
                    self._owner[s] = (la, ex, self._tick)
                    self._step_pins.add(s)
                if layer_filter is not None and li not in layer_filter:
                    continue
                # NEED signal: this step was gate-flagged (2-bit low-confidence), so
                # every expert active in it gets a need bump -- INCLUDING ones already
                # FP4 (so repeat offenders accumulate need and resist eviction). The
                # true culprits are the experts consistently present across fires;
                # decay washes out the coincidental ones.
                self._need[li, ei] += 1.0
                if li in self._store and int(self._mirror[li, ei]) < 0:
                    cand.append((li, ei))
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
            slots = self._take_slots_batch(len(cand), emergency=True)
            plan = [((li, ei), slot) for (li, ei), slot in zip(cand, slots)]
            if not plan:
                return 0
            # one batched host read (pack-file store: mmap -> pinned stage;
            # pinned store: zero-copy views), THEN the H2D copies — all stage
            # rows stay valid until the single sync below.
            rows = self._store.rows_for([p for p, _ in plan])
            for ((li, ei), slot), row in zip(plan, rows):
                self._owner[slot] = (li, ei, self._tick)
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
            # QUALITY KPI: some of this step's missing experts got NO slot
            # (free list empty + every victim ineligible: seen-live or <2
            # ticks old). The mandatory replay then RE-ZEROES their
            # contributions — a silent quality drop even at MISS_TOL=0, and
            # (pool-content-dependent) a source of run-to-run greedy
            # nondeterminism. The fix is a bigger pool, not a knob.
            self._kpi_unfixed += len(cand) - len(plan)
            logger.warning_once(
                "moe_w2 [%s]: %d missing experts could not be promoted "
                "(pool too tight to evict) — replay keeps their zeroed "
                "contributions. Raise the pool GiB; occurrences counted "
                "in the KPI line.", self._tag, len(cand) - len(plan))
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
            cand = []
            for e in ids.cpu():
                e = int(e)
                s = int(self._mirror[layer_key, e])
                if s < 0:
                    cand.append((layer_key, e))
                else:
                    # tick-refresh cached hits (same rationale as
                    # force_promote: protect them from a racing manager
                    # eviction while this layer's eager GEMMs read them)
                    la, ex, _ = self._owner[s]
                    self._owner[s] = (la, ex, self._tick)
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
                self._owner[slot] = (li, ei, self._tick)
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
        if _KPI_EVERY <= 0 or self._kpi_steps < _KPI_EVERY:
            return
        cov_total = max(len(self._store), 1) * self.E
        unfixed = (f"; UNRESTORED experts: {self._kpi_unfixed} "
                   "(pool too tight — quality at risk)"
                   if self._kpi_unfixed else "")
        if self._kpi_2nd:
            unfixed += (f"; second-order replays: {self._kpi_2nd}")
        if self._kpi_prefetched:
            unfixed += (f"; draft-prefetched: {self._kpi_prefetched}")
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
        self._kpi_2nd = 0
        self._kpi_prefetched = 0

    def kpi_second_order(self, n: int, capped: bool = False) -> None:
        """Runner reports n EXTRA fixed-point replays this step (a corrected
        pass re-routed onto still-missing experts). Chronic second-order
        replays = coverage too low for the model's routing volatility.
        `capped`: the loop exited at its bound with misses STILL present —
        the step's logits kept zeroed contributions (hard determinism/
        quality breach; logged loudly)."""
        self._kpi_2nd += n
        if capped:
            self._kpi_unfixed += 1
            logger.warning(
                "moe_w2 [%s]: fixed-point replay hit its cap with misses "
                "remaining — this step kept zeroed expert contributions "
                "(nondeterministic output). Pool badly undersized for this "
                "workload.", self._tag)

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
        if os.getenv("VLLM_MOE_W2_PREFETCH", "0") == "1":
            # in-graph routing log for the draft-affinity prefetcher:
            # [n_layers, T_cap, K_cap] — T_cap covers small decode batches
            # (seqs x MTP verify tokens); bigger steps log a prefix only.
            _BASE_TIER.route_log = torch.zeros(
                n_layers, int(os.getenv("VLLM_MOE_W2_PREFETCH_TCAP", "8")),
                16, dtype=torch.int32, device=_BASE_TIER.dev)
            logger.info("moe_w2 BASE cache: draft-affinity PREFETCH armed "
                        "(route_log %s)", tuple(_BASE_TIER.route_log.shape))
        n_step = n_layers * 16      # decode working set (top-k<=16) headroom
        assert _BASE_TIER.n_slots >= max(2 * 256, n_step), (
            f"moe_w2 base cache: pool of {_BASE_TIER.n_slots} slots is smaller "
            f"than a step's worst-case working set; raise "
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
        _TIER = DeltaTier(
            n_layers, n_experts, dev or torch.device("cuda"),
            w13_bytes=W13_BYTES if w13_bytes is None else w13_bytes,
            w2_bytes=W2_BYTES if w2_bytes is None else w2_bytes,
            policy=policy, tag="fp4" if base_enabled() else "delta",
            host_pinned=not base_enabled())
        # Start the background manager as soon as the tier exists. It idles until
        # experts are actually routed (seen empty -> early return) and only
        # promotes layers whose host planes are already staged, so an early start
        # is safe. This fires correctly under PIPELINE PARALLELISM, where
        # layer_keys are LOCAL per rank and never reach NUM_LAYERS-1 -> the old
        # "start on the last layer built" trigger never ran and the tier sat
        # inactive (pool allocated but no promotions).
        _TIER.start()
    return _TIER
