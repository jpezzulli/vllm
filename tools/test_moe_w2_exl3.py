#!/usr/bin/env python3
"""Parity/dispatch-semantics test for moe_w2_exl3.Exl3BaseTier (host-side,
no vLLM imports; run: CUDA_VISIBLE_DEVICES=<g> python3 tools/test_moe_w2_exl3.py).

Uses REAL inputs end to end: captured hidden states, REAL router (gate
weight+bias from the checkpoint, sqrtsoftplus + noaux_tc selection,
route_scale 1.5), the built EXL3 pack, and FP4-dequant reference weights.

Stages:
  1. no-mask: EXL3 block vs pack-reconstructed manual composition
     (kernel exactness) and vs FP4 reference (quality band).
  2. partial FP4 mask: additive split — exl3(masked) + fp4(torch on the
     masked experts) must equal the mixed-composition reference.
  3. all-masked token contributes zero from the EXL3 side.
"""
import os
import sys
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB_DIR = "/root/workspace/exl3-ab"
PACK = "/root/workspace/moet-serve/exl3-packs-ds4"
CAP = f"{AB_DIR}/capture"
DEV = "cuda:0"
LI = 21
LIMIT = 10.0
M = 32

sys.path.insert(0, AB_DIR)
import exl3_ab as AB  # noqa: E402
from bootstrap_exl3 import load_exl3_quantize  # noqa: E402

Q = load_exl3_quantize()

spec = importlib.util.spec_from_file_location(
    "moe_w2_exl3",
    f"{WT}/vllm/model_executor/layers/quantization/utils/moe_w2_exl3.py")
X3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X3)

# moe_w2_planes is import-clean (os + torch only) -> load it standalone too,
# to exercise the v2 FP4 pool-slot inverse packers + dequant that
# moe_w2_cubit._exl3_fp4_apply uses.
_spec_pl = importlib.util.spec_from_file_location(
    "moe_w2_planes",
    f"{WT}/vllm/model_executor/layers/quantization/utils/moe_w2_planes.py")
PL = importlib.util.module_from_spec(_spec_pl)
_spec_pl.loader.exec_module(PL)

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def real_routing(x):
    """sqrtsoftplus + noaux_tc bias selection, route_scale 1.5 (model.py Gate)."""
    gw = torch.from_numpy(np.asarray(AB.load_raw(f"layers.{LI}.ffn.gate.weight"))
                          .view(np.uint16)).view(torch.bfloat16) \
        .view(256, 4096).to(DEV).float()
    gb = torch.from_numpy(np.asarray(AB.load_raw(f"layers.{LI}.ffn.gate.bias"))
                          .view(np.float32).copy()).view(256).to(DEV)
    scores = F.softplus(x @ gw.T).sqrt()
    idx = (scores + gb).topk(6, dim=-1)[1]
    w = scores.gather(1, idx)
    w = w / w.sum(dim=-1, keepdim=True) * 1.5
    return idx.long(), w.float()


def fp4_expert(ei):
    return tuple(torch.from_numpy(AB.dequant_codes(
        AB.unpack(AB.load_raw(f"layers.{LI}.ffn.experts.{ei}.{p}.weight")),
        AB.load_raw(f"layers.{LI}.ffn.experts.{ei}.{p}.scale"))).to(DEV)
        for p in ("w1", "w3", "w2"))


def block_torch(x, ids, wts, weight_fn, mask=None):
    """Reference block: weight_fn(ei) -> (w1, w3, w2) fp32 [out,in]."""
    y = torch.zeros(x.shape[0], 4096, dtype=torch.float, device=DEV)
    cache = {}
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            if mask is not None and bool(mask[t, j]):
                continue
            ei = int(ids[t, j])
            if ei not in cache:
                cache[ei] = weight_fn(ei)
            w1, w3, w2 = cache[ei]
            g = (x[t] @ w1.T).clamp(max=LIMIT)
            u = (x[t] @ w3.T).clamp(min=-LIMIT, max=LIMIT)
            y[t] += wts[t, j] * ((F.silu(g) * u) @ w2.T)
    return y


def pack_expert(tier, ei):
    def rec(proj, kb, in_f, out_f):
        tr = tier._store[f"l{LI}.e{ei}.{proj}.k{kb}.trellis"]
        suh = tier._store[f"l{LI}.e{ei}.{proj}.k{kb}.suh"]
        svh = tier._store[f"l{LI}.e{ei}.{proj}.k{kb}.svh"]
        w = torch.empty(in_f, out_f, dtype=torch.half, device=DEV)
        tier.ext.reconstruct(w, tr, kb, False, False)
        w = Q.preapply_had_l(w.float(), Q.had_k)
        w *= suh.float().unsqueeze(1)
        w = Q.preapply_had_r(w, Q.had_n)
        w *= svh.float().unsqueeze(0)
        return w.T.contiguous()  # -> [out, in]
    return (rec("w1", tier.w13_k, 4096, 2048),
            rec("w3", tier.w13_k, 4096, 2048),
            rec("w2", tier.w2_k, 2048, 4096))


def rel(a, b):
    return ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()


def main():
    torch.manual_seed(29)
    x = torch.load(f"{CAP}/x_layer{LI:02d}.pt")[-M:].to(DEV).float()
    ids, wts = real_routing(x)
    used = ids.unique().numel()
    print(f"real routing: {M} tokens -> {used} distinct experts")

    tier = X3.Exl3BaseTier(PACK, layers=[LI], device=DEV)
    print(f"tier resident: {tier.total_bytes()/2**30:.2f} GiB (layer {LI})")

    # 1. no mask
    y_x3 = tier.forward_topk(LI, x, ids, wts)
    y_man = block_torch(x, ids, wts, lambda e: pack_expert(tier, e))
    y_fp4 = block_torch(x, ids, wts, fp4_expert)
    e_exact = rel(y_x3, y_man)
    e_qual = rel(y_x3, y_fp4)
    check("kernel exactness vs pack reconstruction", e_exact < 0.01,
          f"rel-err {e_exact:.4f}")
    check("quality vs FP4 reference in expected band", 0.10 < e_qual < 0.30,
          f"rel-err {e_qual:.4f}")

    # 2. partial mask: 2 random slots per token served by "FP4 tier"
    mask = torch.zeros(M, 6, dtype=torch.bool, device=DEV)
    for t in range(M):
        mask[t, torch.randperm(6)[:2]] = True
    y_base = tier.forward_topk(LI, x, ids, wts, fp4_mask=mask)
    y_fp4part = block_torch(x, ids, wts, fp4_expert, mask=~mask)
    y_hybrid = y_base + y_fp4part
    y_refmix = (block_torch(x, ids, wts, lambda e: pack_expert(tier, e), mask=mask)
                + y_fp4part)
    e_split = rel(y_hybrid, y_refmix)
    check("masked split additivity", e_split < 0.01, f"rel-err {e_split:.4f}")
    e_hq = rel(y_hybrid, y_fp4)
    check("hybrid (2/6 FP4) improves on pure base", e_hq < e_qual,
          f"{e_qual:.4f} -> {e_hq:.4f}")

    # 3. fully masked token
    mask_all = torch.zeros(M, 6, dtype=torch.bool, device=DEV)
    mask_all[0, :] = True
    y0 = tier.forward_topk(LI, x, ids, wts, fp4_mask=mask_all)
    check("fully-masked token yields zero", float(y0[0].abs().max()) == 0.0)

    # 4. v2 FP4 pool-slot inverse packers + dequant (moe_w2_cubit
    # ._exl3_fp4_apply primitives). Decisive + self-contained: exact
    # fragment-major round-trip and dequant == direct e2m1 * 2^(e8m0-127).
    torch.manual_seed(4)
    for (Nn, Kk) in ((4096, 4096), (4096, 2048)):
        nib = torch.randint(0, 16, (Nn, Kk), dtype=torch.uint8, device=DEV)
        sc = torch.randint(100, 156, (Nn, Kk // 32), dtype=torch.uint8,
                           device=DEV)
        plane = PL.pack_fp4_fragment_major(nib)
        scpl = PL.pack_scales(sc)
        nib_rt = PL.unpack_fp4_fragment_major(plane, Nn, Kk)
        sc_rt = PL.unpack_scales(scpl, Nn, Kk)
        check(f"fragment-major nibble round-trip [{Nn}x{Kk}]",
              bool((nib_rt == nib).all()))
        check(f"scale round-trip [{Nn}x{Kk}]", bool((sc_rt == sc).all()))
        w_slot = PL.dequant_fp4_expert(plane, scpl, Nn, Kk)
        vals = PL._E2M1_VALS.to(DEV)[nib.long()]
        scale = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=1)
        e_form = rel(w_slot, (vals * scale).float())
        check(f"slot dequant == direct formula [{Nn}x{Kk}]", e_form < 1e-6,
              f"rel-err {e_form:.2e}")

    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
