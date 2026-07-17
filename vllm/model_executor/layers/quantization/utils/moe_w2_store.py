# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side expert stores for the 2-bit MoE tiers (moe_w2_delta.DeltaTier).

Three backends behind one tiny interface:

  - PinnedHostStore: today's behaviour — per-layer [E, slot_bytes] host
    tensors (pinned or pageable), rows handed to cudaMemcpyAsync directly.
    Default; byte-identical to the pre-store code path.
  - MmapPackStore (VLLM_MOE_W2_STORE_DIR=<dir>): rows live in a per-rank
    PACK FILE on disk; reads are buffered preads -> pinned stage -> H2D.
    The kernel page cache is the RAM tier (LRU for free), so host RAM holds
    only the hot part of the base instead of the whole 73-190 GiB store —
    and the pack doubles as a persistent quantization cache across boots (a
    layer already in the pack skips D2H staging entirely).
  - TieredPackStore (additionally VLLM_MOE_W2_BASE_RAM_GB=<GiB|auto>, base
    tier only): a PINNED arena of the N most-recently-used rows over the
    same pack. An arena hit is a zero-copy pinned view (H2D DMAs straight
    from it — the exact PinnedHostStore hot path, no syscall, no memcpy);
    a miss preadv's the row into the arena slot (the arena IS the bounce
    buffer), buffered by default so the page cache serves as an
    opportunistic L3 under the arena (VLLM_MOE_W2_TIER_DIRECT=1 for
    O_DIRECT misses). The arena itself can never be reclaimed under
    memory pressure — the hot fetch set stays RAM-fast even on hosts with
    zero spare page cache. Policy is recency (LRU): freq-pinning lost on
    live GLM traces (routing too flat — see GLM_RAMTIER_FINDINGS).

Pack layout (per tier tag, per TP rank):
    <dir>/<tag>.rank<r>of<w>.pack        raw rows, offset = (li*E+ei)*stride
    <dir>/<tag>.rank<r>of<w>.json        sidecar: shapes + layers written

`stride` is slot_bytes rounded up to 4 KiB so the SAME pack serves the
O_DIRECT reader without a repack (O_DIRECT needs 4K-aligned offset/length/
buffer; the pinned arena is page-aligned and stride-strided, so every row
satisfies all three). Rows of layers not listed in the sidecar are holes
(sparse file) and are never read.

Concurrency: every read/write caller already holds the owning DeltaTier's
lock (manager tick, force_promote, ensure_resident are serialized there),
so the shared pinned stage buffer / arena bookkeeping need no lock of
their own.
"""

import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ALIGN = 4096
_PACK_VERSION = 1


def _ckpt_id():
    """Identity of the checkpoint the pack's rows derive from — the SAME
    function the planes cache uses (model path + sha1 of the safetensors
    index), so both persistent caches agree on what "same model" means.
    A pack is raw expert rows at (layer*E + expert)*stride: NOTHING in its
    geometry distinguishes two checkpoints with equal expert dimensions,
    so without this field a shared STORE_DIR silently serves the other
    model's weights (correct-looking logits, wrong model — the worst
    failure mode). None when no vllm config is current (offline tools,
    unit tests): None only ever matches a pack also written without a
    config, never a production pack."""
    try:
        from vllm.model_executor.layers.quantization.utils.moe_w2_planes_cache import (  # noqa: E501
            _ckpt_id as planes_ckpt_id,
        )
        return planes_ckpt_id()
    except Exception:  # noqa: BLE001 - uninitialized config/offline use
        return None


def _rank_suffix() -> str:
    """Pack-file name suffix identifying this rank's shard. TP rank always;
    PP rank appended only under pipeline parallelism (PP ranks host disjoint
    layers but share TP rank numbers — same-name packs would race on the
    sidecar). Graceful fallback when torch.distributed is uninitialized
    (single-GPU, tests, offline tools)."""
    try:
        from vllm.distributed import (get_pp_group,
                                      get_tensor_model_parallel_rank,
                                      get_tensor_model_parallel_world_size)
        tp_rank = get_tensor_model_parallel_rank()
        tp_world = get_tensor_model_parallel_world_size()
        pp = get_pp_group()
        pp_rank, pp_world = pp.rank_in_group, pp.world_size
    except Exception:  # noqa: BLE001 - any uninitialized-state error
        tp_rank, tp_world, pp_rank, pp_world = 0, 1, 0, 1
    s = f"rank{tp_rank}of{tp_world}"
    if pp_world > 1:
        s += f".pp{pp_rank}of{pp_world}"
    return s


class PinnedHostStore:
    """Per-layer host tensors, exactly the pre-store `_host` dict."""

    resident = True

    def __init__(self, slot_bytes: int, pinned: bool = True):
        self.slot_bytes = slot_bytes
        self._pinned = pinned
        self._layers: dict[int, torch.Tensor] = {}

    def __contains__(self, layer_key: int) -> bool:
        return layer_key in self._layers

    def __len__(self) -> int:
        return len(self._layers)

    def add_layer(self, layer_key: int, parts) -> None:
        E = parts[0].shape[0]
        host = torch.empty(E, self.slot_bytes, dtype=torch.uint8,
                           pin_memory=self._pinned)
        off = 0
        for t in parts:
            host[:, off:off + t.shape[1]].copy_(t, non_blocking=False)
            off += t.shape[1]
        assert off == self.slot_bytes, (off, self.slot_bytes)
        self._layers[layer_key] = host

    def rows_for(self, pairs, scan: bool = False) -> list[torch.Tensor]:
        """Host rows for [(layer, expert), ...]; zero-copy pinned views.
        `scan` (a prefill-sized one-shot batch) only matters for the
        tiered backend's arena policy — ignored here."""
        return [self._layers[li][ei] for li, ei in pairs]

    def release(self) -> None:
        self._layers = {}


class MmapPackStore:
    """Rows in an on-disk pack file; reads staged through a pinned buffer.

    The store is append-only at load time and read-only afterwards. A layer
    present in the sidecar is trusted (shape-checked) and its staging is
    skipped on later boots — the persistent-quant-cache property.
    """

    resident = False

    def __init__(self, dir_: str, tag: str, n_layers: int, n_experts: int,
                 slot_bytes: int):
        self.slot_bytes = slot_bytes
        self.E = n_experts
        self.n_layers = n_layers
        self.stride = (slot_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        os.makedirs(dir_, exist_ok=True)
        base = f"{tag}.{_rank_suffix()}"
        self.path = os.path.join(dir_, base + ".pack")
        self._sidecar_path = os.path.join(dir_, base + ".json")
        self._meta = dict(version=_PACK_VERSION, tag=tag, E=n_experts,
                          n_layers=n_layers, slot_bytes=slot_bytes,
                          stride=self.stride, ckpt_id=_ckpt_id(), layers=[])
        # Identity + geometry gate for the persistent-quant-cache property.
        # ckpt_id (checkpoint path + safetensors-index sha1, the planes
        # cache's convention) is what actually ties the raw rows to A MODEL
        # — geometry alone cannot (two checkpoints with equal expert dims
        # sharing a STORE_DIR would silently swap weights). n_layers guards
        # the row addressing ((layer*E + e)*stride: a different layer count
        # re-maps every offset). A sidecar WITHOUT ckpt_id (pre-fix packs)
        # is treated as stale — one rebuild re-stamps it; operators can
        # instead pre-stamp known-good packs (tools/stamp_pack_identity.py).
        _CHECK = ("version", "E", "n_layers", "slot_bytes", "stride",
                  "ckpt_id")
        if os.path.exists(self._sidecar_path):
            try:
                with open(self._sidecar_path) as f:
                    old = json.load(f)
                match = all(old.get(k) == self._meta[k] for k in _CHECK)
                if match:
                    self._meta["layers"] = sorted(
                        int(li) for li in old.get("layers", []))
                else:
                    logger.warning(
                        "moe_w2 store: pack %s identity/shape mismatch "
                        "(have %s, want %s) — rebuilding", self.path,
                        {k: old.get(k) for k in _CHECK},
                        {k: self._meta[k] for k in _CHECK})
            except (OSError, ValueError, json.JSONDecodeError) as e:
                logger.warning("moe_w2 store: unreadable sidecar %s (%s) — "
                               "rebuilding", self._sidecar_path, e)
        self._present = set(self._meta["layers"])
        size = self.n_layers * self.E * self.stride
        flags = os.O_RDWR | os.O_CREAT
        self._fd = os.open(self.path, flags, 0o644)
        if os.fstat(self._fd).st_size < size:
            os.ftruncate(self._fd, size)   # sparse until layers are written
        # reusable pinned stage for reads (grown on demand; callers hold the
        # tier lock, and the tier syncs its H2D copies before the next call,
        # so reuse is safe).
        self._stage = torch.empty(0, slot_bytes, dtype=torch.uint8,
                                  pin_memory=True)
        # write-side staging reused across layers (pageable [E, stride])
        self._wbuf: torch.Tensor | None = None
        self._pool = ThreadPoolExecutor(
            max_workers=int(os.getenv("VLLM_MOE_W2_STORE_THREADS", "8")),
            thread_name_prefix="moe-w2-store")
        self._reads = 0
        self._read_bytes = 0
        self._read_s = 0.0
        if self._present:
            logger.info(
                "moe_w2 store[%s]: pack %s has %d/%d layers — staging for "
                "those layers will be SKIPPED (persistent quant cache)",
                tag, self.path, len(self._present), n_layers)

    # ---- staging (load time) ----------------------------------------

    def __contains__(self, layer_key: int) -> bool:
        return layer_key in self._present

    def __len__(self) -> int:
        return len(self._present)

    def add_layer(self, layer_key: int, parts) -> None:
        if layer_key in self._present:
            return          # already packed on a previous boot
        E = parts[0].shape[0]
        assert E == self.E, (E, self.E)
        if self._wbuf is None:
            self._wbuf = torch.zeros(self.E, self.stride, dtype=torch.uint8)
        off = 0
        for t in parts:
            self._wbuf[:, off:off + t.shape[1]].copy_(t, non_blocking=False)
            off += t.shape[1]
        assert off == self.slot_bytes, (off, self.slot_bytes)
        mv = memoryview(self._wbuf.numpy()).cast("B")
        base_off = layer_key * self.E * self.stride
        written = 0
        while written < len(mv):        # pwrite may be partial (>2 GiB rows)
            written += os.pwrite(self._fd, mv[written:written + (1 << 30)],
                                 base_off + written)
        os.fdatasync(self._fd)
        self._present.add(layer_key)
        self._meta["layers"] = sorted(self._present)
        tmp = self._sidecar_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._meta, f)
        os.replace(tmp, self._sidecar_path)
        if len(self._present) == self.n_layers:
            self._wbuf = None       # all layers packed; drop write staging

    # ---- reads (serve time) -----------------------------------------

    def rows_for(self, pairs, scan: bool = False) -> list[torch.Tensor]:
        """Pinned-stage rows for [(layer, expert), ...]. The returned views
        alias the shared stage buffer: consume (issue H2D + sync) before the
        next rows_for call — which every DeltaTier call site does.
        `scan` is the tiered backend's arena discipline — ignored here.

        Reads are buffered preads straight into the pinned stage: one
        syscall per row (GIL released, kernel readahead at full drive
        bandwidth) instead of mmap page-fault storms — measured 8x slower
        via mmap on the DS4 PoC, ~100 x 70 KB faults per 6.75 MiB row. An
        fadvise(WILLNEED) pass first lets the drive overlap the cold rows
        of a batch; warm rows are page-cache memcpys — the cache IS the
        RAM tier."""
        n = len(pairs)
        if self._stage.shape[0] < n:
            self._stage = torch.empty(max(n, 2 * self._stage.shape[0]),
                                      self.slot_bytes, dtype=torch.uint8,
                                      pin_memory=True)
        t0 = time.perf_counter()
        offs = [(li * self.E + ei) * self.stride for li, ei in pairs]
        for off in offs:
            try:
                os.posix_fadvise(self._fd, off, self.slot_bytes,
                                 os.POSIX_FADV_WILLNEED)
            except OSError:
                break       # advisory only
        stage_mv = memoryview(self._stage.numpy()).cast("B")

        def _read_one(i_off):
            i, off = i_off
            row = stage_mv[i * self.slot_bytes:(i + 1) * self.slot_bytes]
            done = 0
            while done < self.slot_bytes:
                got = os.preadv(self._fd, [row[done:]], off + done)
                if got <= 0:
                    raise IOError(f"moe_w2 pack short read @ {off + done} "
                                  f"({self.path})")
                done += got

        # preadv releases the GIL: a pool turns both page-cache memcpys and
        # cold NVMe reads into parallel work (single-threaded memcpy at
        # ~6 GB/s made a 400 MB replay fetch cost ~70 ms — measured).
        if n > 2:
            list(self._pool.map(_read_one, enumerate(offs)))
        else:
            for pair in enumerate(offs):
                _read_one(pair)
        self._reads += n
        self._read_bytes += n * self.slot_bytes
        self._read_s += time.perf_counter() - t0
        return [self._stage[i] for i in range(n)]

    def release(self) -> None:
        self._pool.shutdown(wait=False)
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._present = set()
        self._stage = torch.empty(0, self.slot_bytes, dtype=torch.uint8)
        self._wbuf = None

    def stats(self) -> dict:
        return dict(reads=self._reads, read_bytes=self._read_bytes,
                    read_s=self._read_s)


class TieredPackStore(MmapPackStore):
    """Pinned-arena RAM tier over the pack file, O_DIRECT NVMe underneath.

    The arena holds the `n_slots` most-recently-used rows in ONE pinned
    allocation ([n_slots, stride]; page-aligned base + 4K stride = every
    row O_DIRECT-legal). rows_for returns views INTO the arena:

      - hit: zero-copy pinned view, H2D DMAs straight from it — the exact
        PinnedHostStore hot path (no syscall, no memcpy — this is what
        recovers the pack backend's measured -9%);
      - miss: O_DIRECT preadv straight into the arena slot (the arena is
        its own bounce buffer), evicting the least-recently-used slot not
        referenced by the current batch. Host-only eviction is safe: the
        GPU reads its pool copy, never the arena.

    Miss reads are BUFFERED by default: the page cache then acts as an
    opportunistic L3 under the arena — on a RAM-rich host an arena miss is
    a page-cache memcpy, on a tight host the cache stays small and misses
    degrade gracefully to NVMe reads, while the pinned arena floor (the
    hot fetch set) can never be reclaimed either way. Measured on DS4
    1x5090 (base 11 GiB, arena 20 GiB, same-night A/B): pinned 33.0 tok/s,
    tiered-buffered 32.8 (PARITY — the pack backend without an arena sat
    at 25.5), while pure O_DIRECT misses on the box's Gen3-x4-linked drive
    (3.7 GB/s) cost -33%. VLLM_MOE_W2_TIER_DIRECT=1 forces O_DIRECT misses
    (no cache growth, fully deterministic latency = raw drive speed).

    SCAN RESISTANCE: a rows_for(..., scan=True) batch (ensure_resident,
    i.e. prefill layer working sets) may fill FREE arena slots but never
    evicts — a long-document prefill touching most experts once would
    otherwise wipe the decode hot set (measured on GLM needle runs: arena
    hit-rate halved). Scan misses beyond the free slots are served from
    the parent's buffered stage. Decode fetches (force_promote, manager
    _promote) insert/evict normally — the caller, not a batch-size
    heuristic, decides: on GLM the decode replay fetch is routinely
    100+ rows and a size threshold froze the arena (29->10 tok/s).
    VLLM_MOE_W2_TIER_SCAN=0 disables the discipline entirely.

    PREHEAT: the arena's key list (hot-first) is dumped to
    <pack>.heat.json every 1024 fetch calls (async, off the fetch path);
    on boot the previous hot set is read back into the arena
    (VLLM_MOE_W2_TIER_PREHEAT=0 to skip) so the first requests after a
    restart start from RAM instead of paying the NVMe warmup.

    View-lifetime contract (same as the parent's stage): consume the
    returned rows (issue H2D + sync) before the next rows_for call —
    every DeltaTier call site does, under the tier lock. Slots referenced
    by the CURRENT batch are never evicted within it; a batch larger than
    the whole arena overflows into the parent's buffered stage (correct,
    logged, expected only for absurdly small arenas).
    """

    def __init__(self, dir_: str, tag: str, n_layers: int, n_experts: int,
                 slot_bytes: int, ram_gb: float):
        super().__init__(dir_, tag, n_layers, n_experts, slot_bytes)
        self.n_arena = max(int(ram_gb * 2**30) // self.stride, 16)
        self._arena = torch.empty(self.n_arena, self.stride,
                                  dtype=torch.uint8, pin_memory=True)
        assert self._arena.data_ptr() % _ALIGN == 0, "pinned base unaligned?"
        self._arena_mv = memoryview(self._arena.numpy()).cast("B")
        self._pos: dict[tuple[int, int], int] = {}     # (li,ei) -> slot
        self._owner_pair: list = [None] * self.n_arena
        self._last = [0] * self.n_arena                # recency clock stamps
        self._clock = 0
        self._free = list(range(self.n_arena))
        # Miss-read mode: buffered (default; page cache = opportunistic L3)
        # or O_DIRECT (deterministic, bypasses the cache). The O_DIRECT fd
        # is separate; the parent's buffered fd keeps serving writes
        # (add_layer) and stage-overflow reads.
        self.direct = os.getenv("VLLM_MOE_W2_TIER_DIRECT", "0") == "1"
        self._dfd = (os.open(self.path, os.O_RDONLY | os.O_DIRECT)
                     if self.direct else -1)
        self.scan_enabled = os.getenv("VLLM_MOE_W2_TIER_SCAN", "1") == "1"
        # fetch metrics (read by DeltaTier._log_summary/_dump)
        self._hit_rows = 0
        self._miss_rows = 0
        self._miss_bytes = 0
        self._lat_hit_ms = deque(maxlen=2048)    # pure arena-hit calls
        self._lat_miss_ms = deque(maxlen=2048)   # calls with >=1 NVMe row
        self._calls = 0
        self._heat_path = self.path + ".heat.json"
        if os.getenv("VLLM_MOE_W2_TIER_PREHEAT", "1") == "1":
            self._preheat()

    # -- internals ------------------------------------------------------

    def _read_row(self, slot: int, off: int) -> None:
        """One row read from the pack into arena slot `slot` (thread-pool
        body). O_DIRECT mode reads the full stride (offset/length/buffer
        all 4K-aligned by construction); buffered mode reads just
        slot_bytes through the page cache."""
        row = self._arena_mv[slot * self.stride:(slot + 1) * self.stride]
        fd, want = ((self._dfd, self.stride) if self.direct
                    else (self._fd, self.slot_bytes))
        done = 0
        while done < want:
            got = os.preadv(fd, [row[done:want]], off + done)
            if got <= 0:
                raise IOError(f"moe_w2 pack short read @ {off + done} "
                              f"({self.path})")
            done += got

    def _evict_order(self, busy: set) -> list:
        """Slot ids coldest-first, skipping the current batch's slots."""
        order = sorted(range(self.n_arena), key=self._last.__getitem__)
        return [s for s in order if s not in busy]

    def _dump_heat(self, keys: list) -> None:
        """Persist the arena's hot set (async, pool thread). Best-effort:
        a missed dump only costs preheat freshness."""
        try:
            tmp = self._heat_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(dict(version=1, keys=keys), f)
            os.replace(tmp, self._heat_path)
        except OSError as e:
            logger.warning_once("moe_w2 tiered store: heat dump failed: %s",
                                e)

    def _preheat(self) -> None:
        """Refill the arena with the previous run's hot set (boot time,
        before serving — no locking needed). Any failure leaves the arena
        empty and serving proceeds with a cold arena."""
        try:
            with open(self._heat_path) as f:
                keys = [tuple(k) for k in json.load(f).get("keys", [])]
        except (OSError, ValueError, json.JSONDecodeError):
            return
        keys = [k for k in dict.fromkeys(keys)          # dedupe, keep order
                if k[0] in self._present and 0 <= k[1] < self.E]
        keys = keys[:self.n_arena]
        if not keys:
            return
        t0 = time.perf_counter()
        try:
            fills = [(i, (li * self.E + ei) * self.stride)
                     for i, (li, ei) in enumerate(keys)]
            list(self._pool.map(lambda p: self._read_row(p[0], p[1]), fills))
            for i, k in enumerate(keys):
                self._pos[k] = i
                self._owner_pair[i] = k
                self._last[i] = len(keys) - i      # heat order = recency
            self._free = list(range(len(keys), self.n_arena))
            self._clock = len(keys) + 1
            logger.info(
                "moe_w2 tiered store: arena PREHEATED — %d rows "
                "(%.1f GiB) from %s in %.1f s",
                len(keys), len(keys) * self.stride / 2**30,
                self._heat_path, time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001 - preheat must not kill boot
            logger.warning("moe_w2 tiered store: preheat failed (%s) — "
                           "starting cold", e)
            self._pos = {}
            self._owner_pair = [None] * self.n_arena
            self._last = [0] * self.n_arena
            self._free = list(range(self.n_arena))
            self._clock = 0

    # -- reads ----------------------------------------------------------

    def rows_for(self, pairs, scan: bool = False) -> list[torch.Tensor]:
        t0 = time.perf_counter()
        self._clock += 1
        self._calls += 1
        n = len(pairs)
        # Scan resistance: a prefill batch (caller-flagged) may consume
        # free slots but never evicts (see class docstring) — its overflow
        # reads go via the stage and leave the decode hot set alone.
        scan = scan and self.scan_enabled
        out: list = [None] * n
        busy: set = set()
        miss_idx: list[int] = []
        # pass 1: arena hits (and intra-batch duplicates via _pos updates)
        for i, (li, ei) in enumerate(pairs):
            s = self._pos.get((li, ei))
            if s is not None:
                out[i] = s
                self._last[s] = self._clock
                busy.add(s)
            else:
                miss_idx.append(i)
        # pass 2: place misses — free slots, then LRU eviction (non-scan)
        evict_order = None
        ev_i = 0
        overflow: list[int] = []
        placed: list[tuple[int, int, int]] = []   # (idx, slot, offset)
        for i in miss_idx:
            li, ei = pairs[i]
            s = self._pos.get((li, ei))
            if s is not None:          # duplicate earlier in this batch
                out[i] = s
                busy.add(s)
                continue
            if self._free:
                slot = self._free.pop()
            elif scan:
                overflow.append(i)         # scans never evict
                continue
            else:
                if evict_order is None:
                    evict_order = self._evict_order(busy)
                while ev_i < len(evict_order) \
                        and evict_order[ev_i] in busy:
                    ev_i += 1
                if ev_i >= len(evict_order):
                    overflow.append(i)     # batch > arena; stage fallback
                    continue
                slot = evict_order[ev_i]
                ev_i += 1
                old = self._owner_pair[slot]
                if old is not None:
                    del self._pos[old]
            self._pos[(li, ei)] = slot
            self._owner_pair[slot] = (li, ei)
            self._last[slot] = self._clock
            busy.add(slot)
            out[i] = slot
            placed.append((i, slot, (li * self.E + ei) * self.stride))
        # pass 3: parallel fills (the link saturates at low QD for
        # multi-MiB rows; the pool mainly overlaps syscall latency and,
        # in buffered mode, parallelizes page-cache memcpys)
        if placed:
            if not self.direct:
                for _, _, off in placed:
                    try:
                        os.posix_fadvise(self._fd, off, self.slot_bytes,
                                         os.POSIX_FADV_WILLNEED)
                    except OSError:
                        break       # advisory only
            if len(placed) > 2:
                list(self._pool.map(
                    lambda p: self._read_row(p[1], p[2]), placed))
            else:
                for p in placed:
                    self._read_row(p[1], p[2])
        # overflow rows (scan discipline, or arena smaller than one batch):
        # parent's buffered stage
        stage_rows: dict[int, torch.Tensor] = {}
        if overflow:
            if not scan:
                logger.warning(
                    "moe_w2 tiered store: batch of %d rows exceeds the "
                    "arena (%d slots) — %d rows served via buffered stage; "
                    "raise VLLM_MOE_W2_BASE_RAM_GB", n, self.n_arena,
                    len(overflow))
            srows = MmapPackStore.rows_for(
                self, [pairs[i] for i in overflow])
            stage_rows = dict(zip(overflow, srows))
        # metrics + result assembly
        n_miss = len(placed) + len(overflow)
        self._hit_rows += n - n_miss
        self._miss_rows += n_miss
        self._miss_bytes += n_miss * self.stride
        dt_ms = (time.perf_counter() - t0) * 1e3
        (self._lat_miss_ms if n_miss else self._lat_hit_ms).append(dt_ms)
        if placed and self._calls % 1024 == 0:
            keys = sorted(self._pos, key=lambda k: self._last[self._pos[k]],
                          reverse=True)
            self._pool.submit(self._dump_heat, [list(k) for k in keys])
        return [stage_rows[i] if out[i] is None
                else self._arena[out[i], :self.slot_bytes]
                for i in range(n)]

    def release(self) -> None:
        if self._dfd >= 0:
            try:
                os.close(self._dfd)
            except OSError:
                pass
        self._pos = {}
        self._owner_pair = []
        self._free = []
        self._arena_mv = None
        self._arena = torch.empty(0, dtype=torch.uint8)
        super().release()

    def stats(self) -> dict:
        def pct(d, q):
            if not d:
                return 0.0
            v = sorted(d)
            return v[min(int(len(v) * q), len(v) - 1)]
        st = super().stats()
        st.update(
            arena_slots=self.n_arena,
            arena_used=self.n_arena - len(self._free),
            hit_rows=self._hit_rows,
            miss_rows=self._miss_rows,
            miss_bytes=self._miss_bytes,
            hit_p50_ms=pct(self._lat_hit_ms, 0.50),
            hit_p99_ms=pct(self._lat_hit_ms, 0.99),
            miss_p50_ms=pct(self._lat_miss_ms, 0.50),
            miss_p99_ms=pct(self._lat_miss_ms, 0.99),
        )
        return st


def pack_has_layer(tag: str, layer_key: int, n_layers: int, n_experts: int,
                   slot_bytes: int) -> bool:
    """Sidecar-only presence probe: does the pack this config would serve
    from already hold `layer_key`? Used at WEIGHT-CREATE time (before any
    store exists) to decide the loader-level skip — a pack-resident layer's
    checkpoint experts never need to be read into host staging at all.
    Deliberately touches only the sidecar JSON (no fd, no arena, no pinned
    allocs) and never raises."""
    dir_ = os.getenv("VLLM_MOE_W2_STORE_DIR", "").strip()
    if not dir_:
        return False
    try:
        stride = (slot_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        sidecar = os.path.join(dir_, f"{tag}.{_rank_suffix()}.json")
        if not os.path.exists(sidecar):
            return False
        with open(sidecar) as f:
            meta = json.load(f)
        want = dict(version=_PACK_VERSION, E=n_experts,
                    slot_bytes=slot_bytes, stride=stride)
        if any(meta.get(k) != v for k, v in want.items()):
            return False
        return int(layer_key) in {int(li) for li in meta.get("layers", [])}
    except Exception:  # noqa: BLE001 - probe only, staging path still works
        return False


def make_store(tag: str, n_layers: int, n_experts: int, slot_bytes: int,
               pinned: bool):
    """Store factory: pack-file backends when VLLM_MOE_W2_STORE_DIR is set
    (plus a pinned arena for the BASE tier when VLLM_MOE_W2_BASE_RAM_GB
    is set), else the classic pinned/pageable host store. Env read at call
    time so tests can toggle backends without reimporting the module."""
    if os.getenv("VLLM_MOE_W2_BASE_NVME_RATIO", "").strip():
        logger.error(
            "VLLM_MOE_W2_BASE_NVME_RATIO (the RAM:NVMe interleaved-split "
            "experiment, moe_w2_nvme) is superseded by the pack store and "
            "IGNORED. Equivalent config: VLLM_MOE_W2_STORE_DIR=<dir> + "
            "VLLM_MOE_W2_BASE_RAM_GB=<pinned GiB> (the arena fraction is "
            "the RAM share; it also persists quantization across boots).")
    dir_ = os.getenv("VLLM_MOE_W2_STORE_DIR", "").strip()
    if not dir_:
        return PinnedHostStore(slot_bytes, pinned=pinned)
    ram_raw = os.getenv("VLLM_MOE_W2_BASE_RAM_GB", "").strip().lower()
    if tag == "base" and ram_raw not in ("", "0", "0.0"):
        stride = (slot_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        pack_gib = n_layers * n_experts * stride / 2**30
        ram_gb = 0.25 * pack_gib if ram_raw == "auto" else float(ram_raw)
        store = TieredPackStore(dir_, tag, n_layers, n_experts, slot_bytes,
                                ram_gb)
        logger.info(
            "moe_w2 store[%s]: TIERED backend %s — pinned arena %.1f GiB "
            "(%d slots, %.0f%% of the %.1f GiB pack) + %s NVMe misses",
            tag, store.path, store.n_arena * store.stride / 2**30,
            store.n_arena, 100.0 * store.n_arena / (n_layers * n_experts),
            pack_gib, "O_DIRECT" if store.direct else "buffered")
        if store.n_arena < 2 * n_experts:
            logger.warning(
                "moe_w2 store[%s]: arena of %d slots is smaller than one "
                "prefill layer's worst case (2*E=%d) — expect stage "
                "overflows; raise VLLM_MOE_W2_BASE_RAM_GB",
                tag, store.n_arena, 2 * n_experts)
        return store
    store = MmapPackStore(dir_, tag, n_layers, n_experts, slot_bytes)
    logger.info(
        "moe_w2 store[%s]: PACK-FILE backend %s (slot %.2f MiB, "
        "stride %d, %d layers x %d experts; host RAM tier = page cache)",
        tag, store.path, slot_bytes / 2**20, store.stride, n_layers,
        n_experts)
    return store
