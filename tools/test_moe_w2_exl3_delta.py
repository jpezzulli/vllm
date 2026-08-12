#!/usr/bin/env python3
"""Parity test for the Δ-pool dispatch (P6, [M] 2026-07-30): Exl3BaseTier.
forward_topk_dual over a synthetic pool built from the REAL delta pack.

Self-contained (the legacy test_moe_w2_exl3.py depends on the lost exl3-ab
scripts): checkpoint FP4 dequant, real router, base pack, delta pack, and
the exllamav3 ext via internal/exl3-sr-poc/exl3_import (mounted at /poc).

Run inside the PoC-style container:
  python3 /wt/tools/test_moe_w2_exl3_delta.py

Stages:
  1. dual-stream kernel exactness: forward_topk_dual vs torch reference
     (reconstruct + Hadamards) on base+Δ weights, pooled pairs only.
  2. additive split: forward_topk(base, masked) + dual(Δ-pairs) equals the
     mixed reference block.
  3. no pooled pair -> dual contributes exactly zero (negative-skip safety).
  4. quality ladder: err(hybrid 2/6 Δ) < err(base-only); err(all-Δ) matches
     the pack's block validation band (~0.06-0.11 vs FP4).
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/poc")
from exl3_import import load_exl3  # noqa: E402
from sr_poc import dequant_fp4, load_ckpt_tensor  # noqa: E402

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location(
    "moe_w2_exl3",
    f"{WT}/vllm/model_executor/layers/quantization/utils/moe_w2_exl3.py")
X3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X3)

Q, EXT = load_exl3()

BASEPACK = "/basepack"
DELTAPACK = "/delta-pack"
DEV = "cuda:0"
LI = 21
LIMIT = 10.0
M = 32
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def rel(a, b):
    return ((a - b).square().mean().sqrt()
            / b.square().mean().sqrt().clamp_min(1e-12)).item()


def real_routing(x):
    gw = load_ckpt_tensor(f"layers.{LI}.ffn.gate.weight").to(DEV).float()
    gb = load_ckpt_tensor(f"layers.{LI}.ffn.gate.bias").to(DEV).float()
    scores = F.softplus(x @ gw.T).sqrt()
    idx = (scores + gb).topk(6, dim=-1)[1]
    w = scores.gather(1, idx)
    w = w / w.sum(dim=-1, keepdim=True) * 1.5
    return idx.long(), w.float()


def fp4_expert(ei):
    out = []
    for p in ("w1", "w3", "w2"):
        w = load_ckpt_tensor(f"layers.{LI}.ffn.experts.{ei}.{p}.weight")
        s = load_ckpt_tensor(f"layers.{LI}.ffn.experts.{ei}.{p}.scale")
        out.append(dequant_fp4(w, s))          # [out, in] f32
    return tuple(out)


def dereg(w_inner, suh, svh):
    w = Q.preapply_had_l(w_inner.float(), Q.had_k)
    w = w * suh.float().unsqueeze(1)
    w = Q.preapply_had_r(w, Q.had_n)
    return w * svh.float().unsqueeze(0)


def pack_expert(tier, ei):
    def rec(proj, kb, in_f, out_f):
        tr = tier._store[f"l{LI}.e{ei}.{proj}.k{kb}.trellis"]
        suh = tier._store[f"l{LI}.e{ei}.{proj}.k{kb}.suh"]
        svh = tier._store[f"l{LI}.e{ei}.{proj}.k{kb}.svh"]
        w = torch.empty(in_f, out_f, dtype=torch.half, device=DEV)
        tier.ext.reconstruct(w, tr, kb, False, False)
        return dereg(w, suh, svh).T.contiguous()          # [out, in]
    return (rec("w1", tier.w13_k, 4096, 2048),
            rec("w3", tier.w13_k, 4096, 2048),
            rec("w2", tier.w2_k, 2048, 4096))


def delta_expert_from_row(row, geom):
    def part(proj, kdim, ndim):
        dk = geom["dk13"] if proj in ("g", "u") else geom["dk2"]
        otr, osuh, osvh = geom["offs"][proj]
        trb = (kdim // 16) * (ndim // 16) * dk * 32
        tr = row[otr:otr + trb].view(torch.int16) \
            .view(kdim // 16, ndim // 16, 16 * dk).contiguous()
        suh = row[osuh:osuh + kdim * 2].view(torch.half)
        svh = row[osvh:osvh + ndim * 2].view(torch.half)
        w = torch.empty(kdim, ndim, dtype=torch.half, device=DEV)
        EXT.reconstruct(w, tr, dk, False, True)   # pack v3 delta = MUL1
        return dereg(w, suh, svh).T.contiguous()          # [out, in]
    return (part("g", 4096, 2048), part("u", 4096, 2048),
            part("d", 2048, 4096))


def block_torch(x, ids, wts, weight_fn, mask=None):
    y = torch.zeros(x.shape[0], 4096, dtype=torch.float, device=DEV)
    cache = {}
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            if mask is not None and not bool(mask[t, j]):
                continue
            ei = int(ids[t, j])
            if ei not in cache:
                cache[ei] = weight_fn(ei)
            w1, w3, w2 = cache[ei]
            g = (x[t] @ w1.T).clamp(max=LIMIT)
            u = (x[t] @ w3.T).clamp(min=-LIMIT, max=LIMIT)
            y[t] += wts[t, j] * ((F.silu(g) * u) @ w2.T)
    return y


def main():
    torch.manual_seed(29)
    torch.set_grad_enabled(False)
    x = torch.load(f"/capture/x_layer{LI:02d}.pt",
                   map_location="cpu", weights_only=False)[-M:].to(DEV).float()
    ids, wts = real_routing(x)
    used = ids.unique()
    print(f"real routing: {M} tokens -> {used.numel()} distinct experts")

    with open(f"{DELTAPACK}/exl3-delta-l{LI:02d}.manifest.json") as f:
        man = json.load(f)
    dk13 = int(man.get("kd13", man["kd"]))
    dk2 = int(man.get("kd2", man["kd"]))
    base_w2k = int(man.get("base_w2k", 2))
    tier = X3.Exl3BaseTier(BASEPACK, layers=[LI], device=DEV,
                           w2_k=base_w2k)
    geom = X3.delta_slot_geom(dk13, dk2)
    print(f"tier {tier.total_bytes()/2**30:.2f} GiB; Δ slot "
          f"{geom['slot_bytes']/2**20:.2f} MiB (dk13={dk13} dk2={dk2} "
          f"base_w2k={base_w2k})")

    # synthetic pool: every ROUTED expert gets a slot (worst case coverage)
    from safetensors import safe_open
    slot_row = torch.full((256,), -1, dtype=torch.int32, device=DEV)
    rows = []
    with safe_open(f"{DELTAPACK}/exl3-delta-l{LI:02d}.safetensors",
                   "pt") as f:
        for si, e in enumerate(used.tolist()):
            parts = []
            for proj in ("w1", "w3"):
                for part in ("trellis", "suh", "svh"):
                    parts.append(f.get_tensor(f"e{e}.{proj}.dk{dk13}.{part}")
                                 .contiguous().view(torch.uint8).view(-1))
            for part in ("trellis", "suh", "svh"):
                parts.append(f.get_tensor(f"e{e}.w2.dk{dk2}.{part}")
                             .contiguous().view(torch.uint8).view(-1))
            rows.append(torch.cat(parts).to(DEV))
            slot_row[e] = si
    pool = torch.stack(rows)
    assert pool.shape[1] == geom["slot_bytes"]
    tabs = X3.delta_ptr_tables(pool.data_ptr(), geom["slot_bytes"],
                               slot_row, geom)

    def bd_expert(ei):
        b = pack_expert(tier, ei)
        d = delta_expert_from_row(pool[int(slot_row[ei])], geom)
        return tuple(bb + dd for bb, dd in zip(b, d))

    # 1. dual-stream exactness on pooled pairs (2/6 per token)
    dmask = torch.zeros(M, 6, dtype=torch.bool, device=DEV)
    for t in range(M):
        dmask[t, torch.randperm(6)[:2]] = True
    y_dual = tier.forward_topk_dual(LI, x, ids, wts, dmask, tabs, dk13=dk13, dk2=dk2)
    y_ref = block_torch(x, ids, wts, bd_expert, mask=dmask)
    e1 = rel(y_dual, y_ref)
    check("dual-stream exactness vs torch base+Δ reference", e1 < 0.01,
          f"rel-err {e1:.4f}")

    # 2. additive split: base(masked) + dual == mixed reference
    y_base = tier.forward_topk(LI, x, ids, wts, fp4_mask=dmask)
    y_mix_ref = (block_torch(x, ids, wts,
                             lambda e: pack_expert(tier, e), mask=~dmask)
                 + y_ref)
    e2 = rel(y_base + y_dual, y_mix_ref)
    check("hybrid split additivity (base + Δ-pairs)", e2 < 0.01,
          f"rel-err {e2:.4f}")

    # 3. no pooled pair -> exact zero
    none = torch.zeros(M, 6, dtype=torch.bool, device=DEV)
    y0 = tier.forward_topk_dual(LI, x, ids, wts, none, tabs, dk13=dk13, dk2=dk2)
    check("no-Δ token contributes exactly zero",
          float(y0.abs().max()) == 0.0)

    # 3b. unified 6-call pass == split path (base masked + dual) and ref
    y_uni = tier.forward_topk_unified(LI, x, ids, wts, dmask, tabs, dk13=dk13, dk2=dk2)
    e_uni_split = rel(y_uni, y_base + y_dual)
    e_uni_ref = rel(y_uni, y_mix_ref)
    check("unified pass == split path", e_uni_split < 0.002,
          f"rel-err {e_uni_split:.5f}")
    check("unified pass vs torch mixed reference", e_uni_ref < 0.01,
          f"rel-err {e_uni_ref:.4f}")
    y_uni0 = tier.forward_topk_unified(LI, x, ids, wts, none, tabs, dk13=dk13, dk2=dk2)
    e_uni0 = rel(y_uni0, tier.forward_topk(LI, x, ids, wts))
    check("unified with empty pool == pure base", e_uni0 < 0.002,
          f"rel-err {e_uni0:.5f}")

    # 4. quality ladder vs FP4 exact
    y_fp4 = block_torch(x, ids, wts, fp4_expert,
                        mask=torch.ones_like(dmask))
    e_base = rel(tier.forward_topk(LI, x, ids, wts), y_fp4)
    e_hyb = rel(y_base + y_dual, y_fp4)
    all_m = torch.ones_like(dmask)
    y_all = tier.forward_topk_dual(LI, x, ids, wts, all_m, tabs, dk13=dk13, dk2=dk2)
    e_all = rel(y_all, y_fp4)
    check("hybrid (2/6 Δ) improves on pure base", e_hyb < e_base,
          f"{e_base:.4f} -> {e_hyb:.4f}")
    check("all-Δ error in pack validation band", 0.02 < e_all < 0.13,
          f"rel-err {e_all:.4f} (block validations 0.023-0.107)")

    # 5. decode-wave path (M=1): the whole layer in five flat launches, every
    #    active expert served base+Δ == forward_topk_dual with all-True dmask
    #    (the serving decode route for integration 2.3/2.5, [B] 2026-08-05).
    x1 = x[-1:]
    ids1, wts1 = ids[-1:], wts[-1:]
    allm1 = torch.ones(1, 6, dtype=torch.bool, device=DEV)
    y_wave = tier.forward_topk_wave(LI, x1, ids1, wts1, tabs)
    y_dual1 = tier.forward_topk_dual(LI, x1, ids1, wts1, allm1, tabs,
                                     dk13=dk13, dk2=dk2)
    y_ref1 = block_torch(x1, ids1, wts1, bd_expert, mask=allm1)
    check("decode-wave output finite", bool(torch.isfinite(y_wave).all()))
    e_wd = rel(y_wave, y_dual1)
    e_wr = rel(y_wave, y_ref1)
    check("decode-wave vs mgemm-dual (all-Δ, M=1)", e_wd < 0.006,
          f"rel-err {e_wd:.5f}")
    check("decode-wave vs torch base+Δ reference (M=1)", e_wr < 0.01,
          f"rel-err {e_wr:.4f}")

    # 6. decode-wave with a PARTIAL pool (masked Δ): pooled pairs base+Δ,
    #    non-pooled base-only == forward_topk_unified on the same dmask (the
    #    real serving case — only a subset of a token's experts are pooled).
    dmw = torch.zeros(1, 6, dtype=torch.bool, device=DEV)
    dmw[0, torch.randperm(6)[:2]] = True
    y_wm = tier.forward_topk_wave(LI, x1, ids1, wts1, tabs, dmask=dmw)
    y_um = tier.forward_topk_unified(LI, x1, ids1, wts1, dmw, tabs,
                                     dk13=dk13, dk2=dk2)
    y_rm = (block_torch(x1, ids1, wts1, bd_expert, mask=dmw)
            + block_torch(x1, ids1, wts1,
                          lambda e: pack_expert(tier, e), mask=~dmw))
    check("decode-wave (masked 2/6) finite", bool(torch.isfinite(y_wm).all()))
    e_wmu = rel(y_wm, y_um)
    e_wmr = rel(y_wm, y_rm)
    check("decode-wave (masked) vs mgemm-unified", e_wmu < 0.006,
          f"rel-err {e_wmu:.5f}")
    check("decode-wave (masked) vs torch mixed ref", e_wmr < 0.01,
          f"rel-err {e_wmr:.4f}")
    none1 = torch.zeros(1, 6, dtype=torch.bool, device=DEV)
    y_w0 = tier.forward_topk_wave(LI, x1, ids1, wts1, tabs, dmask=none1)
    e_w0 = rel(y_w0, tier.forward_topk(LI, x1, ids1, wts1))
    check("decode-wave (empty pool) == pure base", e_w0 < 0.006,
          f"rel-err {e_w0:.5f}")

    # 7. decode-wave FAST path (persistent ptab stack, glue fix [V] 08-05):
    #    dtabs=None serves off init_ptabs' [18, E] table — base rows static,
    #    Δ rows + the baked zero-svh mask refreshed at pool-MUTATION time via
    #    the delta tier's mutate hook. Same pointer arithmetic as the legacy
    #    per-step compose -> repeat-band equality (wave's fp32 atomics leave
    #    run-to-run noise ~1e-7; 1e-5 bar is decisive vs the 1e-3 kernel band).
    class _FtierShim:  # minimal DeltaTier surface init_ptabs touches
        pass
    shim = _FtierShim()
    shim.pool = pool
    shim.slot_bytes = geom["slot_bytes"]
    shim.slot_table = torch.full((LI + 1, 256), -1, dtype=torch.int32,
                                 device=DEV)
    shim.slot_table[LI] = slot_row
    shim.mutate_hooks = []
    tier.init_ptabs(shim, geom, [(LI, LI)])
    check("ptab init registered a mutate hook", len(shim.mutate_hooks) == 1)
    y_fast = tier.forward_topk_wave(LI, x1, ids1, wts1)
    y_leg = tier.forward_topk_wave(LI, x1, ids1, wts1, tabs,
                                   dmask=(slot_row[ids1] >= 0))
    e_f1 = rel(y_fast, y_leg)
    check("wave FAST (ptab, full pool) == legacy compose", e_f1 < 1e-5,
          f"rel-err {e_f1:.2e}")
    # partial pool: evict half the routed experts, refresh via the hook —
    # the fast path must flip them to base-only exactly like a live dmask
    ev = used[::2]
    shim.slot_table[LI, ev] = -1
    for h in shim.mutate_hooks:
        h()
    dmp = shim.slot_table[LI][ids1] >= 0
    y_fast2 = tier.forward_topk_wave(LI, x1, ids1, wts1)
    y_leg2 = tier.forward_topk_wave(LI, x1, ids1, wts1, tabs, dmask=dmp)
    e_f2 = rel(y_fast2, y_leg2)
    check("wave FAST (ptab, partial pool via hook) == legacy compose",
          e_f2 < 1e-5, f"rel-err {e_f2:.2e}")
    y_rm2 = (block_torch(x1, ids1, wts1, bd_expert, mask=dmp)
             + block_torch(x1, ids1, wts1,
                           lambda e: pack_expert(tier, e), mask=~dmp))
    e_f3 = rel(y_fast2, y_rm2)
    check("wave FAST (partial) vs torch mixed ref", e_f3 < 0.01,
          f"rel-err {e_f3:.4f}")
    # empty pool: everything base-only == pure base forward
    shim.slot_table[LI] = -1
    for h in shim.mutate_hooks:
        h()
    y_fast0 = tier.forward_topk_wave(LI, x1, ids1, wts1)
    e_f0 = rel(y_fast0, tier.forward_topk(LI, x1, ids1, wts1))
    check("wave FAST (empty pool) == pure base", e_f0 < 0.006,
          f"rel-err {e_f0:.5f}")

    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
