# SPDX-License-Identifier: Apache-2.0
"""EXL3 trellis base tier for the moe_w2 stack (vLLM-Moet).

Serves routed-expert GEMMs from EXL3 packs (built by exl3-ab/
build_exl3_pack.py) via exllamav3's grouped multi-matrix kernel
`exl3_mgemm` (pointer-table expert indirection, in-kernel weighted
reduction). Designed to REPLACE the 2-bit scalar base tier while leaving
the FP4 need-pool, gate and miss machinery untouched: experts currently
served by the FP4 pool are masked out of the EXL3 call (negative indices
or compaction) and their contribution is added by the existing moe_w4
path — the composition is additive per token.

This module is deliberately vLLM-import-free so it can be parity-tested
standalone (tools/test_moe_w2_exl3.py) and imported by moe_w2_cubit.

Env:
  VLLM_MOE_W2_EXL3_PACK   pack directory (exl3-l{li:02d}.safetensors)
  VLLM_MOE_W2_EXL3_REPO   exllamav3 checkout providing the CUDA ext
                          (default /root/workspace/exllamav3)
  VLLM_MOE_W2_EXL3_W2K    down-proj K variant: 1|2|3 (default 2)
"""
from __future__ import annotations

import importlib
import os
import sys
import types

import torch
import torch.nn.functional as F

_EXT = None


def delta_slot_geom(dk13: int = 2, dk2: int = 2, hidden: int = 4096,
                    inter: int = 2048) -> dict:
    """Byte geometry of one Δ-pool slot (pack v3). Layout (u8, contiguous):
    w13 section [tr1|suh1|svh1|tr3|suh3|svh3] then w2 section
    [tr2|suh2|svh2]. Trellis bytes = (k/16)*(n/16)*(16*dk words)*2; sidecars
    are fp16 per-channel vectors. All offsets are 16-byte aligned for DS4
    dims. Returns dict(dk13, dk2, sec13, sec2, slot_bytes, offs={proj:
    (tr, suh, svh)}) with offsets relative to the SLOT start.

    Per-projection delta K ([P] 2026-08-02, the 1.67 track's winning
    allocation is base (2,2,1) + Δ (2,2,3)): the w13 streams and the w2
    stream may carry different trellis bitrates; dk13 == dk2 reproduces
    the legacy uniform layout byte-for-byte."""
    tr13 = (hidden // 16) * (inter // 16) * dk13 * 32
    tr2 = (inter // 16) * (hidden // 16) * dk2 * 32
    suh13, svh13 = hidden * 2, inter * 2
    suh2, svh2 = inter * 2, hidden * 2
    one13 = tr13 + suh13 + svh13
    sec13 = 2 * one13
    sec2 = tr2 + suh2 + svh2
    offs = {
        "g": (0, tr13, tr13 + suh13),
        "u": (one13, one13 + tr13, one13 + tr13 + suh13),
        "d": (sec13, sec13 + tr2, sec13 + tr2 + suh2),
    }
    return dict(dk13=dk13, dk2=dk2, sec13=sec13, sec2=sec2,
                slot_bytes=sec13 + sec2, offs=offs)


def delta_ptr_tables(pool_base: int, slot_bytes: int,
                     slot_row: torch.Tensor, geom: dict) -> dict:
    """Per-step Δ pointer tables into the pool: for each projection,
    int64 [E] tensors (trellis, suh, svh) = pool_base + slot*slot_bytes +
    section offset. slot -1 (not pooled) is clamped to slot 0 — those
    entries are never selected (negative-index skip + zero weights in
    forward_topk_dual) but must still point at mapped memory. Pure
    elementwise int64 arithmetic: capture-safe."""
    sp = slot_row.clamp_min(0).to(torch.int64) * slot_bytes + pool_base
    return {proj: (sp + otr, sp + osuh, sp + osvh)
            for proj, (otr, osuh, osvh) in geom["offs"].items()}


def build_m8_groups(topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                    cap: int = 6, rows: int = 8) -> list:
    """Partition one decode step's routed experts into the fixed-size
    groups the M=8-native decode wave serves (charter M8 §2bis: the m8
    kernels are FIXED-SHAPE na=6, so the UNION of the step's experts —
    |U| ∈ [k, T*k] — is split into ceil(|U|/cap) groups and the launcher
    runs once per group; every union expert is decoded exactly ONCE per
    step regardless of the partition, which is the whole M8 value).

    topk_ids [T, k] int, topk_weights [T, k] float — any device; the
    grouping is data-dependent HOST logic (this is an eager-only path),
    so both are copied to CPU here and the returned tensors live on CPU.

    Returns a list of per-group triples:
      experts     (cap,)      int64 — pack expert ids feeding the group's
                  slot pointer columns. Deterministic: the union is
                  ascending, chunked in order. The last group is padded
                  to `cap` with REPEATS of the group's first expert
                  (§2bis: real experts with all rows -1) — a repeated
                  in-group expert re-reads bytes that are already being
                  streamed for this group (L2-warm) instead of pulling a
                  foreign expert's 12.6 MB, and its rows are all -1 so
                  stage_in zeroes them and out skips them: exactly zero
                  contribution.
      row_map     (cap, rows) int32 — [expert-in-group][row] = token
                  index into x, or -1 (empty row). Rows of an expert are
                  its routed tokens in ascending token order, packed
                  from row 0 (T <= rows always holds: a token yields at
                  most ONE row per expert).
      row_weights (cap, rows) float16 — the routing weight of (token,
                  expert); exactly 0.0 where row_map == -1. A duplicate
                  expert inside one token's topk row (not produced by
                  real routing, but legal input) accumulates its weights
                  into the single (expert, token) row in fp32 before the
                  half cast.
    """
    T, k = topk_ids.shape
    assert T <= rows, f"M8 grouping serves T<={rows} tokens, got {T}"
    ids = topk_ids.detach().to("cpu", torch.int64)
    wts = topk_weights.detach().to("cpu", torch.float32)
    per: dict[int, dict[int, float]] = {}
    for t in range(T):
        for j in range(k):
            e = int(ids[t, j])
            tw = per.setdefault(e, {})
            tw[t] = tw.get(t, 0.0) + float(wts[t, j])
    union = sorted(per)
    groups = []
    for g0 in range(0, len(union), cap):
        chunk = union[g0:g0 + cap]
        experts = torch.tensor(chunk + [chunk[0]] * (cap - len(chunk)),
                               dtype=torch.int64)
        row_map = torch.full((cap, rows), -1, dtype=torch.int32)
        row_weights = torch.zeros(cap, rows, dtype=torch.float32)
        for s, e in enumerate(chunk):
            for r, (t, w) in enumerate(sorted(per[e].items())):
                row_map[s, r] = t
                row_weights[s, r] = w
        groups.append((experts, row_map, row_weights.to(torch.half)))
    return groups


def build_m8_groups_graph(topk_ids: torch.Tensor,
                          topk_weights: torch.Tensor,
                          cap: int = 6, rows: int = 8):
    """Capture-safe (fixed-shape, device-only) builder of the M8 groups —
    the in-graph counterpart of build_m8_groups (charter CS [E]
    2026-08-10, F0c: parity 20000/20000 bit-identical with the host
    builder, including (t,e) duplicates, token overlaps, partial and
    fully-empty groups).

    G is FIXED at T (union <= 6T, cap 6 => ceil/6 <= T): real groups come
    first, surplus groups are EMPTY (all row_map -1, experts = repeat of
    union[0] so slot pointers stay dereferenceable; the m8g canon guard
    exits those CTAs in the prologue). Rows are PREFIX-PACKED from row 0
    — the m8g guard's emptiness test reads row 0 only, so this packing
    is a hard contract, not a convention.

    Everything is sort/cumsum/scatter/where on the input device with
    static shapes — no host syncs, no data-dependent shapes: legal
    inside cudagraph capture. Duplicate (t,e) weights accumulate in
    fp64 -> fp32 -> half, matching the host builder's chain bit-exactly.

    Returns (experts [G, cap] i64, row_map [G, cap, rows] i32,
    row_weights [G, cap, rows] f16), all on topk_ids.device."""
    T, k = topk_ids.shape
    G = T
    S = G * cap
    N = T * k
    dev = topk_ids.device

    flat = topk_ids.reshape(N).to(torch.int64)
    wts = topk_weights.reshape(N).to(torch.float64)

    svals, sidx = torch.sort(flat, stable=True)
    first = torch.ones(N, dtype=torch.bool, device=dev)
    first[1:] = svals[1:] != svals[:-1]
    rank_sorted = torch.cumsum(first.to(torch.int64), 0) - 1
    U = rank_sorted[-1] + 1                              # device scalar
    rank = torch.empty(N, dtype=torch.int64, device=dev)
    rank[sidx] = rank_sorted

    exp_of_rank = torch.zeros(S, dtype=torch.int64, device=dev)
    exp_of_rank.scatter_(0, rank_sorted, svals)

    u_ix = torch.arange(S, device=dev)
    g_first = (u_ix // cap) * cap
    pad_started = exp_of_rank[torch.clamp(g_first, max=S - 1)]
    experts = torch.where(u_ix < U, exp_of_rank,
                          torch.where(g_first < U, pad_started,
                                      exp_of_rank[0]))

    tok = torch.arange(T, device=dev, dtype=torch.int64)
    tok_of_el = tok.repeat_interleave(k)
    cell = rank * rows + tok_of_el
    w_dense = torch.zeros(S * rows, dtype=torch.float64, device=dev)
    w_dense.scatter_add_(0, cell, wts)
    w_dense = w_dense.to(torch.float32).reshape(S, rows)
    p_dense = torch.zeros(S * rows, dtype=torch.int64, device=dev)
    p_dense.scatter_(0, cell, torch.ones_like(cell))
    present = p_dense.reshape(S, rows).to(torch.bool)

    row_idx = torch.cumsum(present.to(torch.int64), 1) - 1
    row_map = torch.full((S, rows), -1, dtype=torch.int64, device=dev)
    row_w = torch.zeros(S, rows, dtype=torch.float32, device=dev)
    tgt = torch.where(present, row_idx, torch.zeros_like(row_idx))
    src_t = torch.arange(rows, device=dev,
                         dtype=torch.int64).unsqueeze(0).expand(S, rows)
    row_map.scatter_(1, tgt, torch.where(present, src_t,
                                         torch.full_like(src_t, -1)))
    row_w.scatter_(1, tgt, torch.where(present, w_dense,
                                       torch.zeros_like(w_dense)))
    # scatter order among duplicate target 0 is undefined -> repair col 0
    any_present = present.any(1)
    first_tok = torch.argmax(present.to(torch.int8), 1)
    fw = w_dense.gather(1, first_tok.unsqueeze(1)).squeeze(1)
    row_map[:, 0] = torch.where(any_present, first_tok,
                                torch.full_like(first_tok, -1))
    row_w[:, 0] = torch.where(any_present, fw, torch.zeros_like(fw))

    return (experts.reshape(G, cap),
            row_map.reshape(G, cap, rows).to(torch.int32),
            row_w.reshape(G, cap, rows).to(torch.half))


def _load_ext():
    """Load exllamav3_ext without executing the heavy exllamav3 package
    __init__ (tokenizers/generators are irrelevant here)."""
    global _EXT
    if _EXT is not None:
        return _EXT
    repo = os.environ.get("VLLM_MOE_W2_EXL3_REPO",
                          "/root/workspace/exllamav3")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    for name, path in [("exllamav3", f"{repo}/exllamav3")]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            m.__package__ = name
            sys.modules[name] = m
    ext_mod = importlib.import_module("exllamav3.ext")
    _EXT = ext_mod.exllamav3_ext
    return _EXT


class Exl3BaseTier:
    """Resident EXL3 base tier: per-layer pointer tables over pack tensors.

    forward_topk() computes the full routed-expert block for a batch of
    tokens: y[t] = sum_j w[t,j] * expert_{ids[t,j]}(x[t]), skipping any
    (t, j) marked in fp4_mask (those are served by the FP4 tier outside).
    """

    def __init__(self, pack_dir: str, layers, device: str = "cuda:0",
                 w13_k: int = 2, w2_k: int | None = None,
                 num_experts: int = 256, hidden: int = 4096,
                 inter: int = 2048, act_limit: float = 10.0):
        from safetensors import safe_open
        self.dev = device
        self.E = num_experts
        self.H = hidden
        self.I = inter
        self.act_limit = act_limit
        self.w13_k = w13_k
        self.w2_k = w2_k or int(os.environ.get("VLLM_MOE_W2_EXL3_W2K", "2"))
        self.ext = _load_ext()
        self._store: dict[str, torch.Tensor] = {}
        self._tables: dict[int, dict] = {}
        # Resident zero output-hadamard (svh) vectors. Pointing a non-pooled
        # expert's Δ svh at these zeros its Δ contribution exactly (svh is a
        # post-decode multiply, hadamard_inner) — base-only fallback for the
        # decode wave without a kernel-side delta-enable. svh length = the
        # projection's OUTPUT width: up (g/u) -> I, down (d) -> H.
        self._zero_svh_up = torch.zeros(self.I, dtype=torch.half, device=device)
        self._zero_svh_dn = torch.zeros(self.H, dtype=torch.half, device=device)
        # Persistent decode-wave pointer tables (init_ptabs); None until the
        # serving fast path is wired (first eager forward, before capture).
        self._ptab: dict[int, torch.Tensor] | None = None
        # M8 decode-wave state (charter M8 §2bis): cached ext-capability
        # probe and the lazily allocated per-tier work buffers, reused
        # across groups AND layers — stage_in_m8 zeroes the -1 rows, so
        # they are never cleared between calls.
        self._wave_m8_avail: bool | None = None
        self._m8_bufs: tuple | None = None
        # capture-safe M8 (charter CS [E] 2026-08-10): guarded-cubin probe
        # + persistent fixed-shape buffers of the in-graph route (see
        # forward_topk_wave_m8_graph).
        self._wave_m8_guarded: bool | None = None
        self._m8_gbufs: dict | None = None

        for li in layers:
            path = os.path.join(pack_dir, f"exl3-l{li:02d}.safetensors")
            keep = {}
            with safe_open(path, "pt") as f:
                for ei in range(self.E):
                    for proj, kb in (("w1", self.w13_k), ("w3", self.w13_k),
                                     ("w2", self.w2_k)):
                        for part in ("trellis", "suh", "svh"):
                            k = f"e{ei}.{proj}.k{kb}.{part}"
                            keep[k] = f.get_tensor(k).to(device)
            # layer-prefixed keys: multi-layer tiers must not overwrite the
            # previous layer's tensors (the pointer tables would dangle)
            self._store.update({f"l{li}.{k}": v for k, v in keep.items()})

            def tbl(proj, kb, part):
                return torch.tensor(
                    [keep[f"e{e}.{proj}.k{kb}.{part}"].data_ptr()
                     for e in range(self.E)],
                    dtype=torch.int64, device=device)

            self._tables[li] = {
                "g": tuple(tbl("w1", self.w13_k, p)
                           for p in ("trellis", "suh", "svh")),
                "u": tuple(tbl("w3", self.w13_k, p)
                           for p in ("trellis", "suh", "svh")),
                "d": tuple(tbl("w2", self.w2_k, p)
                           for p in ("trellis", "suh", "svh")),
            }

    def total_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._store.values())

    def _mgemm(self, A, tabs, C, A_had, idx, wts, kb, mults=(0, 0),
               num_tokens=1):
        # mults = (mcg_mult, mul1_mult) uint32 codebook selectors: (0, 0) =
        # 3INST (the base pack); (0, 0x83DCD12D) = MUL1 (the delta pack).
        self.ext.exl3_mgemm(A, tabs[0], C, tabs[1], A_had, tabs[2], idx, wts,
                            kb, -1, mults[0], mults[1], -1, -1, 0, num_tokens)

    def _batched(self, li: int, x: torch.Tensor, topk_ids: torch.Tensor,
                 base_w2_wts: torch.Tensor,
                 dtabs: dict | None = None, dk13: int = 2, dk2: int = 2,
                 dmults: tuple = (0, 0x83DCD12D),
                 delta_w2_wts: torch.Tensor | None = None,
                 delta_gu_sel: torch.Tensor | None = None) -> torch.Tensor:
        """Shared batched core: ALL m tokens x k pairs in ONE mgemm per
        (projection x stream) — 6 calls per LAYER instead of 6 per token
        (the measured per-call cost is a fixed ~8-16 us barrier train, so
        the per-token loop was the dominant decode/prefill dispatch cost;
        see internal/R_EXL3_KERNEL_ANATOMY_2026-08-03.md).

        Slot layout is token-major ([m*k] = m groups of k); the kernel's
        num_tokens grouped reduction (v1.3.0) sums each group into row t.
        Pair selection is WEIGHT-side only (zeroed w2 reduction weights),
        never index-side -1 skips — same reliability rule the dual path
        established. Capture-safe: fixed shapes, no host syncs.

        base_w2_wts / delta_w2_wts: [m, k] float weights for the base /
        delta down-proj reduction (already masked by the caller).
        delta_gu_sel: [m, k] 0/1 gate for the Δ g/u contribution pre-
        activation (unified semantics); None with dtabs = ungated add
        (dual semantics)."""
        t = self._tables[li]
        m, k = topk_ids.shape
        bszm = m * k
        xh = x.half()
        ids = topk_ids.reshape(1, bszm).contiguous()
        # .contiguous() is load-bearing: at m == 1 the expand+reshape can
        # legally return a stride-0 VIEW (one real row aliased k times) and
        # the kernel indexes A + j*H assuming a dense [bszm, 1, H] layout —
        # slots j >= 1 would read out of bounds (NaN cascade; measured).
        A = xh.unsqueeze(1).expand(m, k, self.H) \
            .reshape(bszm, 1, self.H).contiguous()
        w_base = base_w2_wts.half().reshape(1, bszm).contiguous()

        Cg = torch.zeros(bszm, 1, self.I, dtype=torch.half, device=self.dev)
        Cu = torch.zeros(bszm, 1, self.I, dtype=torch.half, device=self.dev)
        Ah = torch.empty(bszm, 1, self.H, dtype=torch.half, device=self.dev)
        Ah2 = torch.empty(bszm, 1, self.I, dtype=torch.half, device=self.dev)
        self._mgemm(A, t["g"], Cg, Ah, ids, None, self.w13_k)
        self._mgemm(A, t["u"], Cu, Ah, ids, None, self.w13_k)

        if dtabs is not None:
            Cg2 = torch.zeros_like(Cg)
            Cu2 = torch.zeros_like(Cu)
            self._mgemm(A, dtabs["g"], Cg2, Ah, ids, None, dk13, dmults)
            self._mgemm(A, dtabs["u"], Cu2, Ah, ids, None, dk13, dmults)
            if delta_gu_sel is not None:
                sel = delta_gu_sel.half().reshape(bszm, 1, 1)
                g = Cg.float() + sel * Cg2.float()
                u = Cu.float() + sel * Cu2.float()
            else:
                g = Cg.float() + Cg2.float()
                u = Cu.float() + Cu2.float()
        else:
            g = Cg.float()
            u = Cu.float()
        h = (F.silu(g.clamp(max=self.act_limit))
             * u.clamp(min=-self.act_limit, max=self.act_limit)).half()

        Cd = torch.zeros(bszm, 1, self.H, dtype=torch.float, device=self.dev)
        self._mgemm(h, t["d"], Cd, Ah2, ids, w_base, self.w2_k,
                    num_tokens=m)
        y = Cd[:m, 0]
        if dtabs is not None:
            w_d = delta_w2_wts.half().reshape(1, bszm).contiguous()
            Cd2 = torch.zeros_like(Cd)
            self._mgemm(h, dtabs["d"], Cd2, Ah2, ids, w_d, dk2, dmults,
                        num_tokens=m)
            y = y + Cd2[:m, 0]
        return y

    @torch.inference_mode()
    def forward_topk(self, li: int, x: torch.Tensor, topk_ids: torch.Tensor,
                     topk_weights: torch.Tensor,
                     fp4_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [m, H] (any float dtype); topk_ids [m, k] long; topk_weights
        [m, k] float; fp4_mask [m, k] bool, True = served by FP4 tier.
        Returns y [m, H] float32 with masked contributions EXCLUDED
        (weight-side zeroing at fixed shapes — no compaction, no host
        syncs, batch-of-tokens in one mgemm per projection)."""
        wts = topk_weights
        if fp4_mask is not None:
            wts = wts.masked_fill(fp4_mask, 0.0)
        return self._batched(li, x, topk_ids, wts)

    @torch.inference_mode()
    def forward_topk_unified(self, li: int, x: torch.Tensor,
                             topk_ids: torch.Tensor,
                             topk_weights: torch.Tensor,
                             dmask: torch.Tensor, dtabs: dict,
                             dk13: int = 2, dk2: int = 2,
                             dmults: tuple = (0, 0x83DCD12D)) -> torch.Tensor:
        """P6-perf: ONE pass serving base AND Δ-pool together — 6 mgemm
        calls per token instead of forward_topk(masked) + forward_topk_dual
        = 9 (which also computed the base g/u twice for pooled pairs).

        Per token, all k pairs at fixed shapes (capture-safe):
          g = g_base + dmask*g_delta;  u likewise  (Δ zeroed pre-activation
          for non-pooled pairs — their slot-0 reads are finite garbage)
          h = silu(clamp(g)) * clamp(u)
          y = w2_base(h, weights=w_full) + w2_delta(h, weights=w_masked)
        Algebraically identical to the split path: pooled pairs get the
        full (Wb+Δ) block on the upgraded h; non-pooled pairs get the pure
        base block (delta w2 weight = 0). Batched: 6 mgemm per LAYER."""
        wts_d = topk_weights.masked_fill(~dmask, 0.0)
        return self._batched(li, x, topk_ids, topk_weights,
                             dtabs=dtabs, dk13=dk13, dk2=dk2, dmults=dmults,
                             delta_w2_wts=wts_d, delta_gu_sel=dmask)

    @torch.inference_mode()
    def forward_topk_dual(self, li: int, x: torch.Tensor,
                          topk_ids: torch.Tensor,
                          topk_weights: torch.Tensor,
                          dmask: torch.Tensor, dtabs: dict,
                          dk13: int = 2, dk2: int = 2,
                          dmults: tuple = (0, 0x83DCD12D)) -> torch.Tensor:
        """Δ-pool contribution: full expert block y = sum_j w[t,j] *
        expert_{base+Δ}(x[t]) for the (t, j) pairs where dmask is True
        (Δ-resident in the pool). Per projection the two trellis streams
        are summed PRE-activation (weight-level additivity: (Wb+Δ)x =
        Wb x + Δ x), then the SwiGLU clamp applies to the sum — matching
        the delta pack's block validation semantics exactly.

        CAPTURE-SAFE: fixed shapes, no compaction, no host syncs. Pair
        selection is weight-side only: EVERY pair computes base+Δ g/u/h
        (non-pooled experts read slot-0 Δ bytes via the clamped pointer
        tables — wrong-but-finite, never unmapped/NaN), and the down-proj
        reduction weights are zeroed for non-pooled pairs, so their
        contribution is exactly 0. (Index-side -1 skip is NOT used: the
        mgemm negative-skip is not reliable mid-array in this fan-out
        mode — measured, parity test stage 1.)

        dtabs: {"g"|"u"|"d": (trellis_ptrs, suh_ptrs, svh_ptrs)} int64 [E]
        tables pointing into the pool slots (built by the caller from
        slot_table + the Δ slot geometry). dmults = (mcg_mult, mul1_mult)
        codebook selectors for the delta stream (pack v3 quantizes with
        MUL1; the base pack is 3INST). Batched: 6 mgemm per LAYER.
        """
        wts_d = topk_weights.masked_fill(~dmask, 0.0)
        return self._batched(li, x, topk_ids, wts_d,
                             dtabs=dtabs, dk13=dk13, dk2=dk2, dmults=dmults,
                             delta_w2_wts=wts_d, delta_gu_sel=None)

    @torch.inference_mode()
    def forward_topk_wave(self, li: int, x: torch.Tensor,
                          topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                          dtabs: dict | None = None,
                          dmask: torch.Tensor | None = None) -> torch.Tensor:
        """Decode path (M=1) via exl3_decode_wave_dual: the whole MoE layer
        in five flat launches instead of six mgemm barrier-trains (track 3.0
        wave; sonda verdict 2026-08-05).

        x [1, H]; topk_ids [1, k] long; topk_weights [1, k] float.
        dmask [1, k] bool selects the Δ-pooled pairs (True = base+Δ). Non-
        pooled pairs fall back to base-only by pointing their Δ output-
        hadamard (svh) at a resident zero vector, which zeros the Δ g/u
        (pre-SwiGLU) and the Δ down contribution exactly — no kernel-side
        delta-enable needed. dmask=None serves every active expert base+Δ
        (== forward_topk_dual all-True). Returns y [1, H] float32 =
        sum_j w[0,j] * expert_{ids[0,j]}(x[0]) with per-pair base/base+Δ.

        The kernel hardcodes the pack-v3 serving pair: base (2,2,1) on cb0
        (3INST) with a K1 down, Δ (2,2,3) on cb2 (MUL1) with a K3 down — so
        the tier must be built w13_k=2, w2_k=1 and dtabs must carry dk13=2,
        dk2=3.

        dtabs=None = the SERVING fast path over the persistent ptab stack
        (init_ptabs): the six slot-major pointer arrays the kernel expects
        (ups [gb,ub,gd,ud] -> 4*na; downs [db,dd] -> 2*na) come out of ONE
        int64 index_select over the per-layer [18, E] table — the Δ rows
        (incl. the zero-svh mask for non-pooled experts) are maintained at
        pool-MUTATION time (refresh_ptabs via the delta tier's mutate hook),
        not per step. The glue anatomy [V] 2026-08-05 measured the per-step
        compose (13 int64 adds + 15 gathers + 6 cats + wheres per LAYER) at
        ~2.0 ms/step of ~1 µs launches; the fast path replaces it with one
        kernel per layer. Values are bit-identical: same pointer arithmetic,
        moved from per-step to per-mutation.

        dtabs given = legacy per-step compose from the Δ pool tables (dmask
        semantics as above) — kept for parity tests and non-serving callers.
        Both paths: capture-safe, no host syncs."""
        assert x.shape[0] == 1, "decode wave is M=1 only"
        assert self.w13_k == 2 and self.w2_k == 1, \
            "wave serves the base (2,2,1) pair (w13_k=2, w2_k=1)"
        ids = topk_ids[0].to(torch.long)
        na = int(ids.numel())
        if dtabs is None:
            # fast path: one gather over the persistent per-layer ptab.
            # Row-major [rows, na] flattens to exactly the slot-major
            # concatenations the kernel wants (see _PTAB_ROWS in init_ptabs).
            assert self._ptab is not None, \
                "forward_topk_wave fast path before init_ptabs()"
            cols = self._ptab[li].index_select(1, ids)   # [18, na]
            up_tr = cols[0:4].reshape(-1)
            up_suh = cols[4:8].reshape(-1)
            up_svh = cols[8:12].reshape(-1)
            dn_tr = cols[12:14].reshape(-1)
            dn_suh = cols[14:16].reshape(-1)
            dn_svh = cols[16:18].reshape(-1)
        else:
            t = self._tables[li]
            g, u, d = t["g"], t["u"], t["d"]
            dg, du, dd = dtabs["g"], dtabs["u"], dtabs["d"]

            # Δ output-hadamard tables, masked to base-only where not pooled
            dg_svh, du_svh, dd_svh = dg[2][ids], du[2][ids], dd[2][ids]
            if dmask is not None:
                m = dmask[0].to(torch.bool)
                zu = dg_svh.new_full((), self._zero_svh_up.data_ptr())
                zd = dd_svh.new_full((), self._zero_svh_dn.data_ptr())
                dg_svh = torch.where(m, dg_svh, zu)
                du_svh = torch.where(m, du_svh, zu)
                dd_svh = torch.where(m, dd_svh, zd)

            # ups slot-major: [base_gate, base_up, delta_gate, delta_up] x na
            up_tr = torch.cat([g[0][ids], u[0][ids], dg[0][ids], du[0][ids]])
            up_suh = torch.cat([g[1][ids], u[1][ids], dg[1][ids], du[1][ids]])
            up_svh = torch.cat([g[2][ids], u[2][ids], dg_svh, du_svh])
            # downs slot-major: [base_down, delta_down] x na
            dn_tr = torch.cat([d[0][ids], dd[0][ids]])
            dn_suh = torch.cat([d[1][ids], dd[1][ids]])
            dn_svh = torch.cat([d[2][ids], dd_svh])

        xh = x.half().reshape(1, self.H).contiguous()
        out = torch.zeros(1, self.H, dtype=torch.float, device=self.dev)
        w = topk_weights[0].to(torch.half).contiguous()
        a_had = torch.empty(4 * na, self.H, dtype=torch.half, device=self.dev)
        c_up = torch.empty(4 * na, self.I, dtype=torch.half, device=self.dev)
        d_in = torch.empty(2 * na, self.I, dtype=torch.half, device=self.dev)
        c_down = torch.empty(2 * na, self.H, dtype=torch.half, device=self.dev)
        self.ext.exl3_decode_wave_dual(
            xh, out, w,
            up_tr.contiguous(), up_suh.contiguous(), up_svh.contiguous(),
            dn_tr.contiguous(), dn_suh.contiguous(), dn_svh.contiguous(),
            a_had, c_up, d_in, c_down,
            self.act_limit, na)
        return out

    # ---- M=8-native decode wave (charter M8 F2 tor B, [K] 2026-08-09) ----

    def ext_wave_m8_available(self) -> bool:
        """True when the loaded ext exposes the M=8-native decode-wave
        launcher AND its loader resolved the m8 cubins (§2bis: lazy
        cuModuleLoad on first launch; exl3_wave_m8_available() -> bool).
        An older ext build without the symbol (AttributeError) or a probe
        that raises degrades to False — the dispatch then keeps the
        existing fallbacks (the AFRAG env-flag-with-degradation pattern).
        Cached: env + ext build cannot change within a process."""
        if self._wave_m8_avail is None:
            try:
                self._wave_m8_avail = bool(self.ext.exl3_wave_m8_available())
            except Exception:
                self._wave_m8_avail = False
        return self._wave_m8_avail

    def ext_wave_m8_guarded(self) -> bool:
        """True when the loader resolved the GUARDED m8 canon set
        (*_m8g.cubin, empty-group prologue guard, 4-ptr ABI) — the hard
        prerequisite of the capture-safe route: with fixed G=T groups
        per captured size, empty groups must cost ~4.5 us (guard exit),
        not ~100 us (full decode). Same degradation contract as
        ext_wave_m8_available (old ext / plain cubins -> False)."""
        if self._wave_m8_guarded is None:
            try:
                self._wave_m8_guarded = bool(self.ext.exl3_wave_m8_guarded())
            except Exception:
                self._wave_m8_guarded = False
        return self._wave_m8_guarded

    def _m8_workbufs(self) -> tuple:
        """Lazy per-tier m8 work buffers (a_had, c_up, d_in, c_down), the
        §2bis shapes: 24 up / 12 dn slots x 8 rows. Allocated once and
        reused across groups AND layers WITHOUT clearing — stage_in_m8
        zeroes the -1 rows itself (§2bis), so stale rows are never read."""
        if self._m8_bufs is None:
            self._m8_bufs = (
                torch.empty(24, 8, self.H, dtype=torch.half, device=self.dev),
                torch.empty(24, 8, self.I, dtype=torch.half, device=self.dev),
                torch.empty(12, 8, self.I, dtype=torch.half, device=self.dev),
                torch.empty(12, 8, self.H, dtype=torch.half, device=self.dev),
            )
        return self._m8_bufs

    @torch.inference_mode()
    def forward_topk_wave_m8(self, li: int, x: torch.Tensor,
                             topk_ids: torch.Tensor,
                             topk_weights: torch.Tensor):
        """Decode step for M ∈ [1, 8] tokens via exl3_decode_wave_dual_m8:
        the union of the step's routed experts is partitioned into fixed
        groups of 6 (build_m8_groups) and the m8 launcher runs once per
        group — each union expert's trellis is decoded ONCE (HMMA over 8
        token rows), vs the M·k expansion's per-(token, expert) decode.

        x [T, H]; topk_ids [T, k] int; topk_weights [T, k] float. Returns
        (out [T, H] float32, n_groups). out is allocated AND zeroed here,
        once per call (= once per layer per step): the ext contract
        (§2bis) is that the kernel only ever atomic-accumulates into out
        — accumulation ACROSS groups is how a token collects the sum
        over all its experts — so callers must not pre-zero or reuse it.

        Serving pair and pointer plumbing are exactly the M=1 wave's:
        pack-v3 base (2,2,1) cb0 + Δ (2,2,3) cb2 over the persistent
        ptab stack (init_ptabs must have run) — one index_select over
        the group's 6 expert columns yields the slot-major 24-up/12-dn
        pointer arrays (ups [gb,ub,gd,ud] x 6, downs [db,dd] x 6). The
        Δ zero-svh bake for non-pooled experts rides along unchanged.

        EAGER-ONLY: the grouping is data-dependent host logic (a D2H
        sync on topk); the dispatch never takes this route during
        cudagraph capture. Work buffers are per-tier and dirty by design
        (see _m8_workbufs)."""
        T = x.shape[0]
        assert 1 <= T <= 8, "decode wave m8 serves T in [1, 8]"
        assert self.w13_k == 2 and self.w2_k == 1, \
            "wave m8 serves the base (2,2,1) pair (w13_k=2, w2_k=1)"
        assert self._ptab is not None, \
            "forward_topk_wave_m8 before init_ptabs()"
        groups = build_m8_groups(topk_ids, topk_weights)
        xh = x.half().reshape(T, self.H).contiguous()
        out = torch.zeros(T, self.H, dtype=torch.float, device=self.dev)
        a_had, c_up, d_in, c_down = self._m8_workbufs()
        ptab = self._ptab[li]
        for experts, row_map, row_weights in groups:
            cols = ptab.index_select(1, experts.to(self.dev))   # [18, 6]
            self.ext.exl3_decode_wave_dual_m8(
                xh, out,
                row_map.to(self.dev).contiguous(),
                row_weights.to(self.dev).contiguous(),
                cols[0:4].reshape(-1).contiguous(),
                cols[4:8].reshape(-1).contiguous(),
                cols[8:12].reshape(-1).contiguous(),
                cols[12:14].reshape(-1).contiguous(),
                cols[14:16].reshape(-1).contiguous(),
                cols[16:18].reshape(-1).contiguous(),
                a_had, c_up, d_in, c_down,
                self.act_limit, 6)
        return out, len(groups)

    def _m8_graph_bufs(self) -> dict:
        """Persistent buffers of the capture-safe M8 route: the ext call
        args must live at STABLE addresses (kernel launch params are
        frozen into the captured graph), so the in-graph builder/gather
        WRITE into these each step and the ext reads them by address —
        the init_ptabs idiom extended to the per-step M8 state.
        cols layout (G, 18, 6) row-major: for group g the six slot-major
        pointer arrays the ext takes are CONTIGUOUS row-slices
        (cols[g,0:4] flat = 24 up_tr, [4:8] suh, [8:12] svh, [12:14]
        dn_tr, [14:16] dn_suh, [16:18] dn_svh) — zero per-call copies."""
        if self._m8_gbufs is None:
            self._m8_gbufs = {
                "rmap": torch.full((8, 6, 8), -1, dtype=torch.int32,
                                   device=self.dev),
                "rw": torch.zeros(8, 6, 8, dtype=torch.half,
                                  device=self.dev),
                "cols": torch.zeros(8, 18, 6, dtype=torch.int64,
                                    device=self.dev),
                "out": torch.zeros(8, self.H, dtype=torch.float,
                                   device=self.dev),
                "xh": torch.zeros(8, self.H, dtype=torch.half,
                                  device=self.dev),
            }
        return self._m8_gbufs

    @torch.inference_mode()
    def forward_topk_wave_m8_graph(self, li: int, x: torch.Tensor,
                                   topk_ids: torch.Tensor,
                                   topk_weights: torch.Tensor):
        """Capture-safe decode step for T in [2, 8] tokens on the
        M=8-native wave (charter CS [E] 2026-08-10): fixed G = T ext
        calls per step, groups built ON DEVICE (build_m8_groups_graph —
        no host syncs, static shapes), pointers gathered from the
        persistent ptab into persistent per-group buffers. Surplus
        groups are empty (all rows -1): the m8g canon guard exits them
        at ~4.5 us/canon, stage kernels write/skip zeros. Requires
        ext_wave_m8_guarded().

        Legal both inside cudagraph capture and eager (the eager route
        forward_topk_wave_m8 stays preferred outside graphs: its host
        builder emits only the real groups)."""
        T = x.shape[0]
        assert 2 <= T <= 8, "graph wave m8 serves T in [2, 8]"
        assert self.w13_k == 2 and self.w2_k == 1, \
            "wave m8 serves the base (2,2,1) pair (w13_k=2, w2_k=1)"
        assert self._ptab is not None, \
            "forward_topk_wave_m8_graph before init_ptabs()"
        bufs = self._m8_graph_bufs()
        a_had, c_up, d_in, c_down = self._m8_workbufs()
        ptab = self._ptab[li]

        experts, rmap, rw = build_m8_groups_graph(topk_ids, topk_weights)
        bufs["rmap"][:T].copy_(rmap)
        bufs["rw"][:T].copy_(rw)
        # (18, T*6) -> (T, 18, 6) into the persistent slot-major buffer
        cols_all = ptab.index_select(1, experts.reshape(-1))
        bufs["cols"][:T].copy_(cols_all.view(18, T, 6).permute(1, 0, 2))
        xh = bufs["xh"][:T]
        xh.copy_(x.reshape(T, self.H))
        out = bufs["out"][:T]
        out.zero_()

        for g in range(T):
            cg = bufs["cols"][g]
            self.ext.exl3_decode_wave_dual_m8(
                xh, out,
                bufs["rmap"][g],
                bufs["rw"][g],
                cg[0:4].reshape(-1),
                cg[4:8].reshape(-1),
                cg[8:12].reshape(-1),
                cg[12:14].reshape(-1),
                cg[14:16].reshape(-1),
                cg[16:18].reshape(-1),
                a_had, c_up, d_in, c_down,
                self.act_limit, 6)
        return out, T

    # ---- persistent decode-wave pointer tables (glue fix [V] 2026-08-05) --
    # Row layout of the per-layer [18, E] int64 table. Rows are grouped so a
    # single index_select(1, ids) -> [18, na] yields, via contiguous row-major
    # slices, exactly the six slot-major arrays exl3_decode_wave_dual takes:
    #   0-3   up_tr  = [g.tr,  u.tr,  dg.tr,  du.tr ]
    #   4-7   up_suh = [g.suh, u.suh, dg.suh, du.suh]
    #   8-11  up_svh = [g.svh, u.svh, dg.svh*, du.svh*]
    #   12-13 dn_tr  = [d.tr,  dd.tr ]
    #   14-15 dn_suh = [d.suh, dd.suh]
    #   16-17 dn_svh = [d.svh, dd.svh*]
    # (*) Δ svh rows carry the baked base-only mask: a non-pooled expert's
    # entry points at the resident zero vector instead of the pool — the same
    # semantics the per-step dmask/torch.where used to apply, moved to
    # pool-mutation time.

    def init_ptabs(self, ftier, geom: dict, pairs) -> None:
        """Build the persistent ptab stack and hook refreshes into the delta
        tier. `pairs` = [(layer_key, li), ...] mapping the delta tier's
        slot_table rows (layer_key) to this tier's pack layers (li). Base
        rows are static (resident pack tensors); Δ rows are (re)computed by
        refresh_ptabs, called once now and then from ftier's mutate hook —
        pool mutations only ever happen in eager host code (gate promote,
        prefill ensure, prefetch), never inside a capture/replay, so the
        captured gather always reads current pointers by address."""
        assert not torch.cuda.is_current_stream_capturing(), \
            "init_ptabs during capture (dummy-run should have initialized)"
        self._ptab_pairs = sorted(pairs)
        self._ptab_keys = torch.tensor([lk for lk, _ in self._ptab_pairs],
                                       dtype=torch.long, device=self.dev)
        L = len(self._ptab_pairs)
        stack = torch.zeros(L, 18, self.E, dtype=torch.int64, device=self.dev)
        for ix, (_lk, li) in enumerate(self._ptab_pairs):
            t = self._tables[li]
            for row, src in ((0, t["g"][0]), (1, t["u"][0]),
                             (4, t["g"][1]), (5, t["u"][1]),
                             (8, t["g"][2]), (9, t["u"][2]),
                             (12, t["d"][0]), (14, t["d"][1]),
                             (16, t["d"][2])):
                stack[ix, row] = src
        self._ptab_stack = stack
        self._ptab = {li: stack[ix]
                      for ix, (_lk, li) in enumerate(self._ptab_pairs)}
        self._ptab_pool_base = ftier.pool.data_ptr()
        self._ptab_slot_bytes = int(ftier.slot_bytes)
        self._ptab_offs = geom["offs"]
        self._ptab_zu = torch.tensor(self._zero_svh_up.data_ptr(),
                                     dtype=torch.int64, device=self.dev)
        self._ptab_zd = torch.tensor(self._zero_svh_dn.data_ptr(),
                                     dtype=torch.int64, device=self.dev)
        self.refresh_ptabs(ftier)
        ftier.mutate_hooks.append(lambda: self.refresh_ptabs(ftier))

    @torch.inference_mode()
    def refresh_ptabs(self, ftier) -> None:
        """Recompute the Δ rows of the ptab stack from the CURRENT slot_table
        — vectorized over all layers (~14 small kernels total per mutation
        event vs 13 int64 kernels x 43 layers x every step before). Same
        pointer arithmetic as delta_ptr_tables + the dmask zero-svh bake.
        inference_mode: the stack is an inference tensor (created inside the
        runner's inference context) and mutation hooks may fire from host
        paths outside it (manager tick) — the decorator keeps the in-place
        row writes legal from any caller."""
        st = ftier.slot_table.index_select(0, self._ptab_keys)   # [L, E] i32
        pooled = st >= 0
        sp = (st.clamp_min(0).to(torch.int64) * self._ptab_slot_bytes
              + self._ptab_pool_base)
        offs = self._ptab_offs
        P = self._ptab_stack
        (g_tr, g_suh, g_svh) = offs["g"]
        (u_tr, u_suh, u_svh) = offs["u"]
        (d_tr, d_suh, d_svh) = offs["d"]
        P[:, 2] = sp + g_tr
        P[:, 3] = sp + u_tr
        P[:, 6] = sp + g_suh
        P[:, 7] = sp + u_suh
        P[:, 10] = torch.where(pooled, sp + g_svh, self._ptab_zu)
        P[:, 11] = torch.where(pooled, sp + u_svh, self._ptab_zu)
        P[:, 13] = sp + d_tr
        P[:, 15] = sp + d_suh
        P[:, 17] = torch.where(pooled, sp + d_svh, self._ptab_zd)
