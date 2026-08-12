#!/usr/bin/env python3
"""Unit tests for moe_w2_exl3.build_m8_groups — the expert-group builder
of the M=8-native decode-wave route (charter M8 F2 tor B, §2bis).

Pure CPU, no GPU, no CUDA ext, no vLLM imports: the module is loaded
standalone via importlib (its import graph is os/sys/types/torch only;
_load_ext only runs from Exl3BaseTier.__init__, never at import).

Run: python3 tools/test_moe_w2_exl3_m8_maps.py

Covers (brief F2):
  (a) full overlap     8 tokens x same 6 experts -> 1 group, rows 0-7
  (b) zero overlap     union 48 -> 8 groups, 1 row per expert
  (c) partial overlap  union 15 -> 3 groups, tail padded with -1 rows
  (d) T = 2..8         random routing, structural + invariants
  (e) weights at the right (expert, row)     hand-built expectation
  (f) -1 exactly where no token              hand-built + structural
  (g) determinism      two builds bit-identical
  (h) scatter invariant: sum(weight x token) over all groups == the sum
      taken directly from topk (random per-expert vectors, fp32)
  plus: duplicate (t, e) accumulation, non-default cap, T > rows raises.
"""
import importlib.util
import math
import os
import sys

import torch

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "moe_w2_exl3",
    f"{WT}/vllm/model_executor/layers/quantization/utils/moe_w2_exl3.py")
X3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X3)

E = 256
FAILS = []
NCHECKS = [0]


def check(name, ok, detail=""):
    NCHECKS[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def ref_weight_matrix(ids, wts):
    """[T, E] half: dedupe-accumulated routing weights straight from topk
    (fp32 accumulate -> half cast, mirroring the builder's contract)."""
    T, k = ids.shape
    W = torch.zeros(T, E, dtype=torch.float32)
    for t in range(T):
        for j in range(k):
            W[t, int(ids[t, j])] += float(wts[t, j])
    return W.to(torch.half)


def group_weight_matrix(groups, T):
    """[T, E] half reconstructed from the groups' scatter maps. Returns
    (W, n_dup) where n_dup counts (token, expert) cells written twice —
    must be 0 (one row per routed pair across ALL groups)."""
    W = torch.zeros(T, E, dtype=torch.half)
    seen, dup = set(), 0
    for experts, rm, rw in groups:
        for s in range(experts.numel()):
            for r in range(rm.shape[1]):
                t = int(rm[s, r])
                if t < 0:
                    continue
                key = (t, int(experts[s]))
                dup += key in seen
                seen.add(key)
                W[t, int(experts[s])] = rw[s, r]
    return W, dup


def structural_problems(groups, T, cap=6, rows=8):
    """Shape/dtype/packing/partition invariants; returns list of issues."""
    probs = []
    real_seq = []          # real experts in group-concatenation order
    for gi, (experts, rm, rw) in enumerate(groups):
        if tuple(experts.shape) != (cap,) or experts.dtype != torch.int64:
            probs.append(f"g{gi}: experts shape/dtype")
        if tuple(rm.shape) != (cap, rows) or rm.dtype != torch.int32:
            probs.append(f"g{gi}: row_map shape/dtype")
        if tuple(rw.shape) != (cap, rows) or rw.dtype != torch.float16:
            probs.append(f"g{gi}: row_weights shape/dtype")
        if not ((rm >= -1) & (rm < T)).all():
            probs.append(f"g{gi}: row_map out of range")
        if not (rw[rm == -1] == 0).all():
            probs.append(f"g{gi}: nonzero weight on -1 row")
        n_real = 0
        for s in range(cap):
            row = rm[s].tolist()
            valid = [v for v in row if v >= 0]
            if row[:len(valid)] != valid:
                probs.append(f"g{gi}s{s}: rows not prefix-packed")
            if sorted(set(valid)) != valid:
                probs.append(f"g{gi}s{s}: rows not ascending-unique")
            if valid:
                real_seq.append(int(experts[s]))
                n_real += 1
                if n_real != s + 1:
                    probs.append(f"g{gi}s{s}: real slot after a pad slot")
            elif int(experts[s]) != int(experts[0]):
                probs.append(f"g{gi}s{s}: pad is not group's first expert")
        if gi < len(groups) - 1 and n_real != cap:
            probs.append(f"g{gi}: pad before the last group")
    if len(real_seq) != len(set(real_seq)):
        probs.append("an expert appears in two groups")
    if real_seq != sorted(real_seq):
        probs.append("union not ascending across groups")
    return probs


def scatter_sum(groups, T, F32):
    """y[t] = sum over groups/slots/rows of weight * f(expert) — the
    literal semantics of the kernel's per-(token, expert) out scatter,
    accumulated ACROSS groups like the atomic out contract (§2bis)."""
    y = torch.zeros(T, F32.shape[1], dtype=torch.float32)
    for experts, rm, rw in groups:
        for s in range(experts.numel()):
            for r in range(rm.shape[1]):
                t = int(rm[s, r])
                if t >= 0:
                    y[t] += float(rw[s, r]) * F32[int(experts[s])]
    return y


def topk_sum(ids, wts, F32):
    """y[t] = sum_j half(w[t,j]) * f(ids[t,j]) directly from topk (the
    per-pair half weight matches the builder's cast; fp32 accumulate)."""
    T, k = ids.shape
    Wh = ref_weight_matrix(ids, wts).float()
    return Wh @ F32


def run_case(name, ids, wts, cap=6, rows=8):
    """Structural + partition-count + weight-matrix + scatter invariants
    common to every case; returns the groups for extra assertions."""
    T = ids.shape[0]
    groups = X3.build_m8_groups(ids, wts, cap=cap, rows=rows)
    union = sorted({int(e) for e in ids.flatten()})
    check(f"{name}: group count", len(groups) == math.ceil(len(union) / cap),
          f"groups={len(groups)} union={len(union)} cap={cap}")
    probs = structural_problems(groups, T, cap=cap, rows=rows)
    check(f"{name}: structural", not probs, "; ".join(probs[:4]))
    Wg, dup = group_weight_matrix(groups, T)
    check(f"{name}: one row per (token, expert)", dup == 0, f"dup={dup}")
    Wr = ref_weight_matrix(ids, wts)
    check(f"{name}: weight matrix == topk (exact)", torch.equal(Wg, Wr))
    F32 = torch.randn(E, 16, dtype=torch.float32,
                      generator=torch.Generator().manual_seed(7))
    ys, yt = scatter_sum(groups, T, F32), topk_sum(ids, wts, F32)
    check(f"{name}: scatter sum == topk sum",
          torch.allclose(ys, yt, rtol=1e-5, atol=1e-5),
          f"maxdiff={float((ys - yt).abs().max()):.2e}")
    return groups


def main():
    g = torch.Generator().manual_seed(20260810)

    # (a) full overlap: 8 tokens x the same 6 experts (unsorted on input)
    ids = torch.tensor([[11, 3, 200, 7, 99, 42]] * 8, dtype=torch.int64)
    wts = torch.rand(8, 6, generator=g)
    groups = run_case("(a) full overlap T=8", ids, wts)
    ex, rm, _ = groups[0]
    check("(a) single group, experts sorted",
          len(groups) == 1 and ex.tolist() == [3, 7, 11, 42, 99, 200])
    check("(a) every expert rows 0-7",
          all(rm[s].tolist() == list(range(8)) for s in range(6)))

    # (b) zero overlap: 8 x 6 all-distinct experts -> union 48 -> 8 groups
    ids = torch.arange(48, dtype=torch.int64).reshape(8, 6) * 5  # spread ids
    wts = torch.rand(8, 6, generator=g)
    groups = run_case("(b) zero overlap T=8", ids, wts)
    check("(b) 8 groups, 1 row per expert",
          len(groups) == 8 and all((g_[1] >= 0).sum() == 6 for g_ in groups))

    # (c) partial overlap: union 15 -> 3 groups, last padded (3 pad slots)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5],
                        [0, 1, 2, 6, 7, 8],
                        [0, 9, 10, 11, 12, 13],
                        [0, 1, 5, 9, 14, 13]], dtype=torch.int64)
    wts = torch.rand(4, 6, generator=g)
    groups = run_case("(c) partial overlap T=4", ids, wts)
    ex, rm, rw = groups[-1]
    check("(c) 3 groups, last has 3 pad slots",
          len(groups) == 3 and [int(rm[s].max()) for s in range(3, 6)]
          == [-1, -1, -1])
    check("(c) pad slots repeat first expert, zero weights",
          ex[3:].tolist() == [int(ex[0])] * 3 and (rw[3:] == 0).all())
    check("(c) shared expert 0 has rows = tokens 0-3",
          groups[0][1][0].tolist() == [0, 1, 2, 3, -1, -1, -1, -1])

    # (d) T = 2..8, random distinct-per-token routing (k=6, E=256)
    for T in range(2, 9):
        ids = torch.stack([torch.randperm(E, generator=g)[:6]
                           for _ in range(T)])
        wts = torch.rand(T, 6, generator=g)
        run_case(f"(d) random T={T}", ids, wts)

    # (e)+(f) hand-built: exact row_map / row_weights placement
    ids = torch.tensor([[5, 9], [9, 5], [7, 5]], dtype=torch.int64)
    wts = torch.tensor([[.1, .2], [.3, .4], [.5, .6]])
    groups = run_case("(e) hand T=3 k=2", ids, wts)
    ex, rm, rw = groups[0]
    check("(e) union [5, 7, 9] + 3 pads of 5",
          ex.tolist() == [5, 7, 9, 5, 5, 5])
    check("(e) row_map exact",
          rm.tolist() == [[0, 1, 2, -1, -1, -1, -1, -1],
                          [2, -1, -1, -1, -1, -1, -1, -1],
                          [0, 1, -1, -1, -1, -1, -1, -1],
                          [-1] * 8, [-1] * 8, [-1] * 8])
    want = torch.zeros(6, 8, dtype=torch.half)
    want[0, :3] = torch.tensor([.1, .4, .6]).half()
    want[1, 0] = torch.tensor(.5).half()
    want[2, :2] = torch.tensor([.2, .3]).half()
    check("(e) row_weights exact at (expert, row)", torch.equal(rw, want))
    check("(f) -1 exactly where no token",
          ((rm == -1) == (want == 0)).all())

    # (g) determinism: two independent builds are bit-identical
    ids = torch.stack([torch.randperm(E, generator=g)[:6] for _ in range(5)])
    wts = torch.rand(5, 6, generator=g)
    g1 = X3.build_m8_groups(ids.clone(), wts.clone())
    g2 = X3.build_m8_groups(ids.clone(), wts.clone())
    check("(g) determinism", len(g1) == len(g2) and all(
        torch.equal(a[i], b[i]) for a, b in zip(g1, g2) for i in range(3)))

    # duplicate (t, e) inside one topk row accumulates into ONE row
    ids = torch.tensor([[4, 4, 8], [8, 2, 2]], dtype=torch.int64)
    wts = torch.tensor([[.1, .2, .3], [.4, .5, .6]])
    groups = run_case("dup (t,e) accumulate", ids, wts)
    ex, rm, rw = groups[0]
    check("dup: single row, fp32-accumulated weight",
          ex.tolist()[:3] == [2, 4, 8]
          and rm[1].tolist() == [0, -1, -1, -1, -1, -1, -1, -1]
          and rw[1, 0] == torch.tensor(.3, dtype=torch.float32).half()
          and rw[0, 0] == torch.tensor(1.1, dtype=torch.float32).half())

    # non-default cap partitions the same union into more groups
    ids = torch.arange(10, dtype=torch.int64).reshape(2, 5)
    wts = torch.rand(2, 5, generator=g)
    groups = run_case("cap=4", ids, wts, cap=4)
    check("cap=4: shapes follow cap",
          all(tuple(gr[0].shape) == (4,) and tuple(gr[1].shape) == (4, 8)
              for gr in groups))

    # T > rows must refuse (a token would need a 9th row)
    try:
        X3.build_m8_groups(torch.zeros(9, 6, dtype=torch.int64),
                           torch.rand(9, 6, generator=g))
        check("T=9 raises", False)
    except AssertionError:
        check("T=9 raises", True)

    print(f"\n{NCHECKS[0]} checks, {len(FAILS)} failed"
          + (f": {FAILS}" if FAILS else " — ALL GREEN"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
