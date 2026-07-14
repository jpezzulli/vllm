#!/usr/bin/env python3
"""GPU unit test for VLLM_MOE_W2_PREFILL_FP4 (prefill consumes the FP4 tier).

Builds synthetic experts through the production plane builders, promotes
half of them into a DeltaTier, then runs _moe_w2_forward at PREFILL size
(T=333 > 96) and checks against a mixed f32 reference:
  - resident experts -> FP4 (non-split) / true-e2m1 (split) dequant,
  - non-resident     -> 2-bit sign-sym dequant,
with the same a32 activation-quant numerics on both GEMM inputs.

Cases: non-split w4, split w4q (both with AFRAG on), flag-off bit-exactness
vs the legacy 2-bit-only prefill, and empty-pool no-op.

Run inside the vllm image on one GPU:
  VLLM_MOE_W2_CUBIT_DIR=/cubit-share python3 tools_prefill_fp4_test.py
"""
import os
import sys

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit  # noqa: E402
from vllm.model_executor.layers.quantization.utils import moe_w2_delta  # noqa: E402
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (  # noqa: E402
    mxfp4_to_codes, mxfp4_to_nibbles, pack_fp4_fragment_major,
    pack_fragment_major, pack_quintal_fragment_major, pack_scales,
    quintal_dequant,
)

assert moe_w2_cubit._ensure_ready(), "cubins not found"
assert moe_w2_cubit._PREFILL_FP4, "flag expected default-on"
dev = torch.device("cuda")
torch.manual_seed(23)

E, H, I = 32, 4096, 2048
T, TOPK = 333, 6                    # T > 96 -> prefill tier (mc4/afrag)
LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=dev)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2, device=dev)
E2M1[8:] *= -1

w13_pack = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
w2_pack = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device=dev)

st = dict(N13=2 * I, K13=H, N2=H, K2=I, E=E)
st["planes13"] = torch.stack([pack_fragment_major(mxfp4_to_codes(w13_pack[e])) for e in range(E)])
st["sc13"] = torch.stack([pack_scales(s13[e]) for e in range(E)])
st["planes2"] = torch.stack([pack_fragment_major(mxfp4_to_codes(w2_pack[e])) for e in range(E)])
st["sc2"] = torch.stack([pack_scales(s2[e]) for e in range(E)])
moe_w2_cubit._LAYERS[0] = st

x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).to(torch.int32)
topk_w = torch.rand(T, TOPK, device=dev) * 0.5


def dequant_w2(pack, sc):
    codes = mxfp4_to_codes(pack)
    return LEVELS[codes.long()] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


def dequant_fp4(pack, sc):
    nib = mxfp4_to_nibbles(pack)
    return E2M1[nib.long()] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


def dequant_split(pack, sc):
    nib = mxfp4_to_nibbles(pack)
    return (quintal_dequant(nib)
            * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1))


def reference(promoted, dq_hot):
    a_deq = moe_w2_cubit.a32_dequant_ref(x, gemm=1)
    ref = torch.zeros(T, H, device=dev)
    for t in range(T):
        for j in range(TOPK):
            e = int(topk_ids[t, j])
            dq = dq_hot if e in promoted else dequant_w2
            c13 = a_deq[t] @ dq(w13_pack[e], s13[e]).T
            act = torch.nn.functional.silu(c13[:I]) * c13[I:]
            act_deq = moe_w2_cubit.a32_dequant_ref(
                act.to(torch.bfloat16).unsqueeze(0), gemm=2)
            ref[t] += float(topk_w[t, j]) * (act_deq[0] @ dq(w2_pack[e], s2[e]).T)
    return ref


def check(tag, got, ref):
    rel = (got.float() - ref).abs().max().item() / ref.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.flatten(), dim=0).item()
    okc = rel < 0.06 and cos > 0.999
    print(f"{tag}: max_rel={rel:.3e} cos={cos:.6f} -> {'PASS' if okc else 'FAIL'}")
    return okc


ok = True
promoted = list(range(0, E, 2))

# ---- baseline: legacy 2-bit-only prefill (no tier at all)
moe_w2_delta._TIER = None
legacy = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
ok &= check("prefill 2-bit only (no tier)", legacy, reference(set(), None))

# ---- case 1: NON-SPLIT w4 tier consumed by prefill
os.environ["VLLM_MOE_W2_DELTA_GB"] = "1"
tier = moe_w2_delta.DeltaTier(1, E, dev,
                              w13_bytes=2 * I * H // 2, w2_bytes=H * I // 2)
moe_w2_delta._TIER = tier
fp13 = torch.stack([pack_fp4_fragment_major(mxfp4_to_nibbles(w13_pack[e]))
                    for e in range(E)])
fp2 = torch.stack([pack_fp4_fragment_major(mxfp4_to_nibbles(w2_pack[e]))
                   for e in range(E)])
tier.add_layer_host_planes(0, fp13, fp2)
for e in promoted:
    tier._promote(0, e, tier._take_slot(set()))
torch.cuda.synchronize()
got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
ok &= check(f"prefill w4 mixed ({len(promoted)}/{E} resident)",
            got, reference(set(promoted), dequant_fp4))

# ---- flag OFF must be bit-exact vs legacy (tier present but ignored)
moe_w2_cubit._PREFILL_FP4 = False
off = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
bit = torch.equal(off, legacy)
print(f"flag off vs legacy: {'bit-exact PASS' if bit else 'MISMATCH FAIL'}")
ok &= bit
moe_w2_cubit._PREFILL_FP4 = True

# ---- case 2: SPLIT (quintal w4q) tier consumed by prefill
moe_w2_delta._SPLIT = True
assert moe_w2_delta.split_enabled()
tier_s = moe_w2_delta.DeltaTier(1, E, dev,
                                w13_bytes=2 * I * H * 5 // 16,
                                w2_bytes=H * I * 5 // 16)
moe_w2_delta._TIER = tier_s
rf13 = torch.stack([pack_quintal_fragment_major(mxfp4_to_nibbles(w13_pack[e]))
                    for e in range(E)])
rf2 = torch.stack([pack_quintal_fragment_major(mxfp4_to_nibbles(w2_pack[e]))
                   for e in range(E)])
tier_s.add_layer_host_planes(0, rf13, rf2)
for e in promoted:
    tier_s._promote(0, e, tier_s._take_slot(set()))
torch.cuda.synchronize()
got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
ok &= check(f"prefill w4q SPLIT mixed ({len(promoted)}/{E} resident)",
            got, reference(set(promoted), dequant_split))

# ---- empty pool: tier exists, nothing resident -> must equal legacy
tier_e = moe_w2_delta.DeltaTier(1, E, dev,
                                w13_bytes=2 * I * H * 5 // 16,
                                w2_bytes=H * I * 5 // 16)
moe_w2_delta._TIER = tier_e
tier_e.add_layer_host_planes(0, rf13, rf2)
torch.cuda.synchronize()
empty = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
bit = torch.equal(empty, legacy)
print(f"empty pool vs legacy: {'bit-exact PASS' if bit else 'MISMATCH FAIL'}")
ok &= bit
moe_w2_delta._SPLIT = False

# ---- AFRAG off cross-check (row-major w2 prefill + w4q subs)
os.environ["VLLM_MOE_W2_AFRAG"] = "0"
moe_w2_cubit._afrag_ok = False
moe_w2_delta._SPLIT = True
moe_w2_delta._TIER = tier_s
got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
ok &= check("prefill w4q SPLIT, AFRAG off",
            got, reference(set(promoted), dequant_split))

# ---- ENSURE mode: guarantee from an EMPTY pool — every routed expert is
# fetched synchronously per layer, so the output must equal the FULL
# quintal reference (nothing left on 2-bit) despite zero initial residency.
tier_g = moe_w2_delta.DeltaTier(1, E, dev,
                                w13_bytes=2 * I * H * 5 // 16,
                                w2_bytes=H * I * 5 // 16)
moe_w2_delta._TIER = tier_g
tier_g.add_layer_host_planes(0, rf13, rf2)
torch.cuda.synchronize()
moe_w2_cubit._PREFILL_FP4_ENSURE = True
got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
ok &= check("prefill w4q ENSURE (empty pool -> full quintal)",
            got, reference(set(range(E)), dequant_split))
n_res = int((tier_g.slot_table[0] >= 0).sum())
routed = int(topk_ids.unique().numel())
print(f"ensure promoted {n_res} slots for {routed} routed experts",
      "PASS" if n_res == routed else "FAIL")
ok &= n_res == routed
moe_w2_cubit._PREFILL_FP4_ENSURE = False
moe_w2_delta._SPLIT = False

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
