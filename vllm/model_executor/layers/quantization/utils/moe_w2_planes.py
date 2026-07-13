# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""2-bit tensor-sym expert planes for the 1-GPU DeepSeek-V4 plan.

Load-time GPU quantizer + fragment-major plane packer for the cubit
`moe_w2` decode kernel. The quantization is the QUANT_PROBE-validated
K=4 sign-symmetric codebook {-4, -1, 1, 4} (acceptance 2.73 vs 2.68
baseline): every mxfp4 e2m1 value maps to the nearest level with
odd-symmetric tie-breaking (zeros map sign-preservingly to +-1).

Mapping (e2m1 nibble -> 2-bit code), code order {0:-4, 1:-1, 2:+1, 3:+4}:
  +vals [0, .5, 1, 1.5, 2, 3, 4, 6] -> [+1 x5, +4 x3] -> codes [2,2,2,2,2,3,3,3]
  -vals (nibble | 8)                -> [-1 x5, -4 x3] -> codes [1,1,1,1,1,0,0,0]
Scales: the checkpoint's block-32 UE8M0 bytes are kept VERBATIM (the
kernel feeds them straight into QMMA.SF per k32).

Plane layout (fragment-major, per expert weight matrix [N, K]):
  for each 16-row block nb (N/16), for each k64 block kb (K/64),
  for each lane (g, t) in (8, 4):
    8 bytes = codes for the lane's QMMA fragment chunks, in order:
      [t0 k32a lo, t0 k32a hi, t0 k32b lo, t0 k32b hi,
       t1 k32a lo, t1 k32a hi, t1 k32b lo, t1 k32b hi]
    where t0 row = nb*16 + g, t1 row = nb*16 + g + 8,
          k32a = kb*64, k32b = kb*64 + 32,
          lo = weights [k + 4t .. 4t+3], hi = [k + 16 + 4t .. +3],
          each 4-weight chunk packs little-endian: code(k+4t) in bits 0-1.
  => plane bytes = N/16 * K/64 * 32 lanes * 8 = N*K/4.
"""

import torch

# e2m1 nibble -> 2-bit code (tensor-sym {-4,-1,1,4}), validated against
# tools/repack_expert_bits.py in tools/test_moe_w2_planes.py.
# The classification is proximity: mags {0,.5,1,1.5,2} -> +-1 (small),
# {3,4,6} -> +-4 (big). Re-classifying mag 2 into the big class (the
# "variant A" bijection candidate for the split-FP4 zero loss) was
# MEASURED AND REJECTED 2026-07-13: base rel-RMS +35% (DS4) / +29% (GLM),
# cold-tier needle 32k FAIL with degeneration loops — see
# internal/SPLIT_FP4_ZERO_LOSS_NEXT_SESSION.md.
_NIBBLE_TO_CODE = torch.tensor(
    [2, 2, 2, 2, 2, 3, 3, 3,   # +0,.5,1,1.5,2,3,4,6
     1, 1, 1, 1, 1, 0, 0, 0],  # -0,-.5,-1,-1.5,-2,-3,-4,-6
    dtype=torch.uint8)

# 2-bit code -> e2m1 nibble of the reconstructed level (for golden tests)
_CODE_TO_NIBBLE = torch.tensor([0xE, 0xA, 0x2, 0x6], dtype=torch.uint8)

# --- split-FP4 refinement, LEGACY 2-bit (moe_w4s_mm) ------------------------
# SUPERSEDED by the radix-5 quintal planes below (moe_w4q_mm): the 2-bit
# refinement has 4 states for the small class's 5 magnitudes, so mag 0
# MERGES into 0.5 — every zero weight serves as +-0.5 x 2^(scale-127).
# Chosen on GLM packs (3.6% zeros); on DS4 zeros are 11.6% of elements and
# the merge measurably decays GSM8K with FP4-pool coverage. Kept only for
# the historical kernel's tests; serving dispatches moe_w4q_mm.
_MAG_TO_REF = torch.tensor([0, 0, 1, 2, 3, 0, 1, 2], dtype=torch.uint8)
# ref -> reconstructed |value|, per class (golden tests / references)
_REF_TO_VAL_SMALL = torch.tensor([0.5, 1.0, 1.5, 2.0])
_REF_TO_VAL_BIG = torch.tensor([3.0, 4.0, 6.0, 6.0])


def nibbles_to_refinement(nib: torch.Tensor) -> torch.Tensor:
    """e2m1 nibbles (u8, 0..15) -> 2-bit refinement codes for the LEGACY
    moe_w4s_mm (mag-0 merge included; superseded by the quintal planes)."""
    return _MAG_TO_REF.to(nib.device)[(nib & 7).long()]


def split_fp4_dequant(nib: torch.Tensor) -> torch.Tensor:
    """Values the LEGACY 2-bit SPLIT decode reconstructs from e2m1 nibbles
    (mag 0 -> 0.5 merge included) — the reference for moe_w4s golden tests.
    The quintal path (moe_w4q_mm) reconstructs TRUE e2m1 — its reference
    is quintal_dequant below."""
    dev = nib.device
    mag = (nib & 7).long()
    code = _NIBBLE_TO_CODE.to(dev)[nib.long()]
    ref = _MAG_TO_REF.to(dev)[mag].long()
    big = (code == 0) | (code == 3)
    val = torch.where(big, _REF_TO_VAL_BIG.to(dev)[ref],
                      _REF_TO_VAL_SMALL.to(dev)[ref])
    return torch.where(code <= 1, -val, val)


# --- split-FP4 refinement, RADIX-5 quintal planes (moe_w4q_mm) --------------
# BIT-EXACT split at 2.5 bits/elem: the decoder reconstructs the e2m1
# magnitude INDEX — the base code narrows it to the small ({0,.5,1,1.5,2})
# or big ({3,4,6}) class, so a base-5 digit per element covers both, and
# 4 digits pack into one 10-bit word per QMMA quad (5^4 = 625 <= 1024).
# Kernel decode: magic /5 ((x*0x334)>>12, exact for x<1024, brute-forced
# in gen_moe_w4q.py) -> selector = digit + 5*is_big(base) = mag idx ->
# one PRMT over the FULL e2m1 magnitude pool {0,.5,1,1.5 | 2,3,4,6}; sign
# from the base code. All 16 nibbles reconstruct exactly (zeros included).
# Slot = 5/8 of the non-split nibble plane -> 1.6x experts/GiB (1.7x over
# the base cache, where non-split slots also carry scale sections).
_QUINTAL_POW = torch.tensor([1, 5, 25, 125], dtype=torch.int64)


def nibbles_to_quintal_digits(nib: torch.Tensor) -> torch.Tensor:
    """e2m1 nibbles (u8, 0..15) -> base-5 digits (i64, 0..4): the e2m1
    magnitude index within the element's base-code class (small: mag idx
    0..4; big: mag idx - 5 in 0..2)."""
    mag = (nib & 7).long()
    code = _NIBBLE_TO_CODE.to(nib.device)[nib.long()]
    big = (code == 0) | (code == 3)
    return torch.where(big, mag - 5, mag)


def pack_quintal_fragment_major(nib: torch.Tensor) -> torch.Tensor:
    """[N, K] u8 e2m1 nibbles -> quintal FP4 plane [N*K*5/16] u8.

    moe_w4q_mm layout, per (nb, kb64) block (320 B): a 32-lane x 8 B "P8"
    section (record bits 0..64) then a 32-lane x 2 B "P2" section (bits
    64..80). A lane's 80-bit record = 8 words x 10 bits, little-endian;
    word w = base-5 pack (d0 + 5 d1 + 25 d2 + 125 d3) of the 4 elements
    of the w2-layout plane byte w (same [nb,kb,g,t,tile,k32,half,k4]
    permutation as pack_fragment_major)."""
    N, K = nib.shape
    assert N % 16 == 0 and K % 64 == 0
    d = nibbles_to_quintal_digits(nib)
    d = d.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    d = d.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 8, 4)
    words = (d * _QUINTAL_POW.to(d.device)).sum(-1)          # [lanes, 8]
    # 8 x 10-bit words -> 5 x u16 shorts of the 80-bit LE stream
    shorts = torch.zeros(words.shape[0], 5, dtype=torch.int64,
                         device=d.device)
    for w in range(8):
        b = 10 * w
        s, off = b // 16, b % 16
        shorts[:, s] |= words[:, w] << off
        if off > 6:
            shorts[:, s + 1] |= words[:, w] >> (16 - off)
    shorts &= 0xFFFF
    by = torch.stack([shorts & 0xFF, shorts >> 8], dim=-1).view(-1, 10)
    by = by.view(-1, 32, 10).to(torch.uint8)                 # [blk, lane, 10]
    return torch.cat([by[:, :, :8].reshape(-1, 256),
                      by[:, :, 8:].reshape(-1, 64)], dim=1).flatten()


def quintal_fp4_plane_bytes(n: int, k: int) -> int:
    """Slot bytes of one [n, k] quintal plane (2.5 bits/elem)."""
    assert (n * k) % 16 == 0
    return n * k * 5 // 16


def quintal_dequant(nib: torch.Tensor) -> torch.Tensor:
    """Values the QUINTAL decode reconstructs from e2m1 nibbles — true
    e2m1, merge-free (the golden reference for moe_w4q tests; equality
    with the E2M1 table IS the bit-exactness criterion)."""
    dev = nib.device
    digit = nibbles_to_quintal_digits(nib)
    code = _NIBBLE_TO_CODE.to(dev)[nib.long()]
    big = (code == 0) | (code == 3)
    mag_idx = torch.where(big, digit + 5, digit)             # decode side
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                        device=dev)
    val = mags[mag_idx]
    return torch.where(code <= 1, -val, val)


# 2-bit code -> e4m3 byte (the kernel's PRMT LUT): -4,-1,1,4
PRMT_LUT_WORD = 0x4838B8C8


def mxfp4_to_codes(w_packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] u8 packed e2m1 pairs -> [..., K] u8 2-bit codes (0..3).

    Nibble order: low nibble = even k (matches mxfp4 packing).
    """
    lut = _NIBBLE_TO_CODE.to(w_packed.device)
    lo = lut[(w_packed & 0xF).long()]
    hi = lut[(w_packed >> 4).long()]
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def pack_fragment_major(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] u8 codes (0..3) -> fragment-major plane [N*K/4] u8."""
    N, K = codes.shape
    assert N % 16 == 0 and K % 64 == 0
    c = codes.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    # dims: nb, tile(g|g+8), g, kb, k32(a|b), half(lo|hi), t, k4
    #   row = nb*16 + tile*8 + g ; k = kb*64 + k32*32 + half*16 + t*4 + k4
    # target order: [nb, kb, g, t, tile, k32, half, k4]
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    # pack 4 codes (k4) little-endian into one byte
    c = c.view(-1, 4).to(torch.int32)
    packed = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6))
    return packed.to(torch.uint8).flatten()


def quantize_expert(w_packed: torch.Tensor) -> torch.Tensor:
    """mxfp4 [N, K/2] u8 -> fragment-major 2-bit plane [N*K/4] u8 (GPU)."""
    return pack_fragment_major(mxfp4_to_codes(w_packed))


def mxfp4_to_nibbles(w_packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] u8 packed e2m1 pairs -> [..., K] u8 raw nibbles (0..15)."""
    lo = w_packed & 0xF
    hi = w_packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def pack_fp4_fragment_major(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] u8 e2m1 nibbles -> fragment-major FP4 plane [N*K/2] u8.

    moe_w4_mm layout: per (nb, kb64, lane) 16 bytes = 4 words in order
    [t0 k32a, t0 k32b, t1 k32a, t1 k32b]; word nibbles 0-3 = lo quad
    (k = 4t+j), 4-7 = hi quad (k = 16+4t+j), little-endian.
    """
    N, K = codes.shape
    assert N % 16 == 0 and K % 64 == 0
    c = codes.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    # [nb, tile, g, kb, k32, half, t, j] -> [nb, kb, g, t, tile, k32, half, j]
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    c = c.view(-1, 2).to(torch.int16)
    return (c[:, 0] | (c[:, 1] << 4)).to(torch.uint8).flatten()


# e2m1 magnitude grid and the midpoints between adjacent magnitudes.
# Bucketizing |u| against the midpoints (right=False: first midpoint >= |u|)
# reproduces tools/repack_expert_bits.py's nearest-with-lo-tie-break snap,
# e.g. |u| == 2.5 -> magnitude 2 (-> code +-1), matching the GLM-5.2 sweep.
_E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                         dtype=torch.float64)


def _f64_to_codes_scales(
    w: torch.Tensor,
    want_nibbles: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Dequantized f64 weights [N, K] -> (2-bit codes, UE8M0 scale bytes).

    The sweep-validated requant pipeline (internal/glm52-sweep/sweep.py):
    per-block-32 UE8M0 scale along K -> e2m1 snap -> tensor-sym {-4,-1,1,4}
    via _NIBBLE_TO_CODE. Load-time-only f64 math so midpoint comparisons and
    tie-breaks match the numpy prototype the sweep validated exactly.

    ZERO-SIGN BALANCING (Kimi-K2.7-NVFP4 finding): the tensor-sym codebook
    has no zero level, so exact zeros map to +-1 by their SIGN BIT. That is
    unbiased only while +0/-0 are balanced (GLM/DS4 checkpoints are). The
    modelopt INT4->BF16->NVFP4 Kimi-K2.7 export writes ALL zeros as +0
    (~13.3% of expert mass) -> sign-preserving mapping would inject a +0.134
    unit-space bias per tensor, 3x the asym-codebook bias that degenerates
    GLM (tools/sweep_nvfp4_codebook.py reproduces both numbers). When a
    tensor's exact zeros are one-signed (>95%), assign them +-1 ALTERNATING
    by k-position parity instead: net bias ~0 (block-local cancellation),
    identical L2 (|err| = 1 unit either way), deterministic. Balanced-zero
    checkpoints keep the validated sign-preserving map bit-exactly.
    VLLM_MOE_W2_ZERO_MODE={auto,sign,alt} overrides (default auto).
    """
    assert w.dtype == torch.float64
    N, K = w.shape
    assert K % 32 == 0
    wb = w.view(N, K // 32, 32)
    amax = wb.abs().amax(dim=2)
    # UE8M0: power-of-2 scale mapping block amax onto e2m1 max (6.0). All-zero
    # blocks get the minimum exponent so the (zero -> +-1 code) dequant stays
    # ~2^-127 instead of poisoning the block with +-1.0.
    # exponent clamped to e8m0's [-127, 127] (byte 255 = NaN is never emitted)
    exp = torch.where(amax > 0,
                      torch.round(torch.log2(amax / 6.0 + 1e-30)),
                      torch.full_like(amax, -127.0)).clamp_(-127.0, 127.0)
    scale_bytes = (exp + 127.0).to(torch.uint8)
    u = wb / torch.exp2(exp).unsqueeze(2)     # exact: power-of-2 division
    mag = torch.bucketize(u.abs().reshape(N, K),
                          _E2M1_MID.to(w.device)).to(torch.uint8)
    neg = torch.signbit(u).reshape(N, K)

    import os
    zero_mode = os.getenv("VLLM_MOE_W2_ZERO_MODE", "auto")
    if zero_mode != "sign":
        zero = (u == 0.0).reshape(N, K)
        nz = int(zero.sum())
        if nz:
            nneg = int(neg[zero].sum())
            one_signed = min(nneg, nz - nneg) < 0.05 * nz
            if zero_mode == "alt" or (zero_mode == "auto" and one_signed):
                # k-position parity: deterministic, block-local ~balance
                parity = (torch.arange(K, device=w.device, dtype=torch.uint8)
                          & 1).view(1, K).expand(N, K)
                neg = torch.where(zero, parity.bool(), neg)

    nibbles = mag | (neg.to(torch.uint8) << 3)
    codes = _NIBBLE_TO_CODE.to(w.device)[nibbles.long()]
    return codes, scale_bytes, (nibbles if want_nibbles else None)


def fp8_block_to_codes_scales(
    w_fp8: torch.Tensor,
    s_block: torch.Tensor,
    block: int = 128,
    want_nibbles: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FP8 block-quant checkpoint expert -> (2-bit codes, UE8M0 scale bytes).

    GLM-5.2 / DS4-FP8 checkpoints carry float8_e4m3fn weights with f32
    block-128x128 scales instead of the mxfp4 codes the mxfp4 loader feeds
    the plane packers. Dequantize to f64 and re-quantize with the
    sweep-validated pipeline (_f64_to_codes_scales).

    Returns (codes [N, K] u8 0..3, scale_bytes [N, K/32] u8 e8m0,
    nibbles [N, K] u8 e2m1 | None). `nibbles` (the FP4 "baseline" of the
    sweep) feeds the optional delta tier's FP4 planes.
    """
    N, K = w_fp8.shape
    w = w_fp8.double()
    sb = s_block.double()
    s = sb.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return _f64_to_codes_scales(w * s, want_nibbles)


# e2m1 nibble -> value (f64), for NVFP4 dequant: +[0,.5,1,1.5,2,3,4,6], then
# the same magnitudes negated (nibble bit 3 = sign).
_E2M1_VALS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float64)


def nvfp4_to_codes_scales(
    w_packed: torch.Tensor,
    s_block: torch.Tensor,
    s2: torch.Tensor,
    group: int = 16,
    want_nibbles: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """NVFP4 (modelopt) checkpoint expert -> (2-bit codes, UE8M0 scale bytes).

    nvidia/GLM-5.2-NVFP4-style tensors: `weight` [N, K/2] u8 packed e2m1
    pairs (low nibble = even k, same packing as mxfp4), `weight_scale`
    [N, K/16] float8_e4m3fn block-16 scales, `weight_scale_2` per-tensor f32
    (scalar, or [N] when the fused w13 carries distinct w1/w3 scale_2 —
    pass it expanded per row). True weight = e2m1 * e4m3_scale * scale_2.

    Dequantize to f64 (exact: all three factors are exactly representable)
    and re-quantize with the sweep-validated pipeline (_f64_to_codes_scales).
    The returned UE8M0 block-32 scales absorb scale_2, so the serving path
    needs no extra per-tensor factor.
    """
    N, K2 = w_packed.shape
    K = K2 * 2
    assert s_block.shape == (N, K // group), (s_block.shape, N, K, group)
    nib = mxfp4_to_nibbles(w_packed)                     # [N, K] u8
    w = _E2M1_VALS.to(w_packed.device)[nib.long()]      # f64
    s = s_block.double().repeat_interleave(group, dim=1)
    w = w * s
    s2 = s2.double().to(w.device)
    if s2.dim() == 0 or s2.numel() == 1:
        w = w * s2.reshape(())
    else:
        assert s2.shape == (N,), s2.shape
        w = w * s2.view(N, 1)
    return _f64_to_codes_scales(w, want_nibbles)


def pack_scales(scales: torch.Tensor) -> torch.Tensor:
    """[N, K/32] u8 e8m0 -> kernel scale plane [N*K/32] u8.

    Layout: sbyte[nb, ks, r] at (nb*(K/32) + ks)*16 + r  (r = row in the
    16-row block); kernel lane (g,t) reads r=g (tile0) / r=8+g (tile1).
    """
    N, KS = scales.shape
    assert N % 16 == 0
    return scales.view(N // 16, 16, KS).transpose(1, 2).contiguous().flatten()


def reference_dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[N, K] codes + [N, K/32] e8m0 scale bytes -> f32 weights (golden ref)."""
    levels = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=codes.device)
    vals = levels[codes.long()]
    s = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)
    return vals * s
