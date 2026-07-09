// KV-NVFP4 for sparse-MLA SM120 (GLM / DSv3.2-family) - prologue expansion.
// Packed storage layout V1, 352 B/token.
//
// Storage per token (gmem, flat addressing idx*352):
//   [0:256)   512 x E2M1 (2/byte, even dim in the low nibble)
//   [256:288) 32 x E4M3 block scales (block 16), dequant = e4m3 * 2^-6
//   [288:352) 64 x FP8 E4M3 rope, scale 1.0
//
// Kernel-side: IO lands the packed 288 B at the TAIL of the existing 528 B
// smem slot (offset 240) and the rope 64 B at the tail of the 128 B rope
// slot (offset 64). Before QK the math warps expand in-place to the
// GLM_NSA-compatible layout:
//   [0:512)   E4M3 (values rescaled to a common tile-128 scale)
//   [512:528) 4 x FP32 tile-128 scale (max of the block scales in the tile)
// so the ENTIRE existing FP8-MMA pipeline runs unchanged. Requant noise
// measured on real GLM latents: rel-RMS 0.0949 -> 0.0967 (the same error
// path validated end-to-end with fake-quant + fp8_ds_mla writes).
//
// Rope: E4M3 -> BF16 in-place dequant (exact, no extra noise).

#pragma once

#include <cuda_fp16.h>

// Global namespace, matching the other common/ headers (q_rope etc.);
// also visible from decode_dsv3_2_kernel.cuh's namespace.

// E2M1 magnitudes {0,.5,1,1.5,2,3,4,6} as E4M3 bytes (exact representation).
constexpr uint32_t NVFP4_LUT_A = 0x3C383000u;  // kody 0..3
constexpr uint32_t NVFP4_LUT_B = 0x4C484440u;  // kody 4..7
constexpr float NVFP4_GLOBAL_SCALE = 0.015625f;  // 2^-6, LAYOUT §2

// Offsets within the 352 B gmem blob and the smem slots.
constexpr int NVFP4_GMEM_SCALE_OFF = 256;
constexpr int NVFP4_GMEM_ROPE_OFF = 288;
constexpr int NVFP4_PACKED_BYTES = 288;      // nibble + skale (bulk 1)
constexpr int NVFP4_SMEM_LANDING_OFF = 240;  // 528 - 288
constexpr int NVFP4_ROPE_GMEM_BYTES = 64;    // bulk 2
constexpr int NVFP4_ROPE_LANDING_OFF = 64;   // 128 - 64

// 4 bytes = 8 E2M1 nibbles -> e4_lo (even dims), e4_hi (odd dims), byte
// order = pair order. Verified bit-exact against the torch reference.
__device__ __forceinline__ void nvfp4_e2m1x8_to_e4m3(uint32_t w, uint32_t& e4_lo,
                                                     uint32_t& e4_hi) {
  const uint32_t lo = w & 0x0F0F0F0Fu;
  const uint32_t hi = (w >> 4) & 0x0F0F0F0Fu;
#pragma unroll
  for (int k = 0; k < 2; k++) {
    const uint32_t s = k ? hi : lo;
    const uint32_t mag = s & 0x07070707u;
    const uint32_t t = mag | (mag >> 4);
    const uint32_t sel = __byte_perm(t, 0u, 0x4420u);
    uint32_t e4 = __byte_perm(NVFP4_LUT_A, NVFP4_LUT_B, sel);
    e4 |= (s & 0x08080808u) << 4;
    (k ? e4_hi : e4_lo) = e4;
  }
}

__device__ __forceinline__ __half2 nvfp4_fp8x2_to_h2(uint32_t two_bytes) {
  uint32_t f16x2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"((uint16_t)two_bytes));
  return *reinterpret_cast<__half2*>(&f16x2);
}

__device__ __forceinline__ float nvfp4_fp8_to_f32(uint8_t b) {
  return __half2float(__low2half(nvfp4_fp8x2_to_h2((uint32_t)b)));
}

// 4 E4M3 bytes (mag+sign) * ratio -> 4 E4M3 bytes (satfinite RN).
__device__ __forceinline__ uint32_t nvfp4_requant4(uint32_t e4, __half2 ratio2) {
  const __half2 a = __hmul2(nvfp4_fp8x2_to_h2(e4 & 0xFFFFu), ratio2);
  const __half2 b = __hmul2(nvfp4_fp8x2_to_h2(e4 >> 16), ratio2);
  uint16_t pa, pb;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(pa) : "r"(*(const uint32_t*)&a));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(pb) : "r"(*(const uint32_t*)&b));
  return (uint32_t)pa | ((uint32_t)pb << 16);
}

// 4 E4M3 bytes -> 2x u32 bf16x2 (rope dequant, exact).
__device__ __forceinline__ void nvfp4_fp8x4_to_bf16x4(uint32_t e4, uint32_t& b01,
                                                      uint32_t& b23) {
  const __half2 h01 = nvfp4_fp8x2_to_h2(e4 & 0xFFFFu);
  const __half2 h23 = nvfp4_fp8x2_to_h2(e4 >> 16);
  const float f0 = __low2float(h01), f1 = __high2float(h01);
  const float f2 = __low2float(h23), f3 = __high2float(h23);
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b01) : "f"(f1), "f"(f0));
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b23) : "f"(f3), "f"(f2));
}

// Prefill rope B-operand prefetch from GMEM: 64 x E4M3 instead of bf16.
// Returns a struct compatible with q_rope.cuh::KVRopePrefetch (templated
// because this header may be included before q_rope.cuh). Lane mapping is
// identical to prefetch_kv_rope: b[ks][0] = elems (16ks+2tid, +1),
// b[ks][1] = (+8).
template <typename KVRopePrefetchT, int N_ROPE_CHUNKS_T = 4>
__device__ __forceinline__ KVRopePrefetchT nvfp4_prefetch_kv_rope_t(
    const uint8_t* rope_packed, int lane) {
  const int tid = lane & 3;
  KVRopePrefetchT pf;
#pragma unroll
  for (int ks = 0; ks < N_ROPE_CHUNKS_T; ks++) {
    const int e0 = ks * 16 + tid * 2;
    const uint16_t p0 = *reinterpret_cast<const uint16_t*>(rope_packed + e0);
    const uint16_t p1 = *reinterpret_cast<const uint16_t*>(rope_packed + e0 + 8);
    const __half2 h0 = nvfp4_fp8x2_to_h2((uint32_t)p0);
    const __half2 h1 = nvfp4_fp8x2_to_h2((uint32_t)p1);
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
        : "=r"(pf.b[ks][0])
        : "f"(__high2float(h0)), "f"(__low2float(h0)));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
        : "=r"(pf.b[ks][1])
        : "f"(__high2float(h1)), "f"(__low2float(h1)));
  }
  return pf;
}

// Expand one tile of BI entries, in place.
//   kv_smem:   BI slots of smem_stride(=528) B; packed data lives at
//              [NVFP4_SMEM_LANDING_OFF, 528) of each slot.
//   rope_smem: BI slots of 128 B (bf16[64]); packed at [64,128); may be
//              nullptr (prefill reads rope from gmem).
// Called by THREADS threads (tid = 0..THREADS-1); BarSync must be a barrier
// covering EXACTLY those threads. Post-bulk visibility is provided by the
// mbarrier wait; callers need one more barrier before other warps read the
// expanded data (in practice: the same math warps).
template <int THREADS, int BI_T, typename BarSync>
__device__ __forceinline__ void nvfp4_expand_tile(uint8_t* kv_smem, int smem_stride,
                                                  uint8_t* rope_smem, int tid,
                                                  BarSync bar) {
  // Work: BI*16 latent segments (segment = 32 dims = 16 B packed -> 32 B
  // expanded) + BI*16 rope groups (group = 4 elems).
  constexpr int SEGS = BI_T * 16;
  constexpr int SEGS_PER_PHASE = SEGS / 2;
  constexpr int SEG_PER_THREAD = SEGS_PER_PHASE / THREADS;  // 2 dla BI=64,T=256
  static_assert(SEGS_PER_PHASE % THREADS == 0);

#pragma unroll
  for (int phase = 0; phase < 2; phase++) {
    // --- read/decode phase into registers ---
    uint32_t out[SEG_PER_THREAD][8];
    float tile_sc[SEG_PER_THREAD];
    uint32_t rope_out[SEG_PER_THREAD][2];
#pragma unroll
    for (int i = 0; i < SEG_PER_THREAD; i++) {
      const int seg = phase * SEGS_PER_PHASE + i * THREADS + tid;
      const int cand = seg >> 4;
      const int s = seg & 15;  // which 32-dim segment within the token
      uint8_t* slot = kv_smem + (size_t)cand * smem_stride;
      const uint8_t* packed = slot + NVFP4_SMEM_LANDING_OFF;

      // segment block scales (blocks 2s, 2s+1) and tile scale (max of 8)
      const uint8_t* scales = packed + 256;  // = slot + 496
      const int t128 = s >> 2;
      float tmax = 0.f;
#pragma unroll
      for (int b = 0; b < 8; b++)
        tmax = fmaxf(tmax, nvfp4_fp8_to_f32(scales[t128 * 8 + b]));
      const float sc0 = nvfp4_fp8_to_f32(scales[2 * s]);
      const float sc1 = nvfp4_fp8_to_f32(scales[2 * s + 1]);
      const float inv_t = (tmax > 0.f) ? (1.0f / tmax) : 0.f;
      const __half2 r0 = __half2half2(__float2half(sc0 * inv_t));
      const __half2 r1 = __half2half2(__float2half(sc1 * inv_t));
      tile_sc[i] = tmax * NVFP4_GLOBAL_SCALE;

      // 16 B of packed nibbles -> 32 B of e4m3 (rescaled to the tile scale)
      const uint4 pk = *reinterpret_cast<const uint4*>(packed + s * 16);
      const uint32_t ws[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint32_t e4l, e4h;
        nvfp4_e2m1x8_to_e4m3(ws[q], e4l, e4h);
        const __half2 rq = (q < 2) ? r0 : r1;  // q=0,1 -> block 2s; q=2,3 -> 2s+1
        e4l = nvfp4_requant4(e4l, rq);
        e4h = nvfp4_requant4(e4h, rq);
        // interleave back: pair byte j holds dim2j(lo) and dim2j+1(hi);
        // e4l = even dims (4B), e4h = odd; the output wants
        // [d0,d1,d2,d3][d4..] - interleave via byte_perm.
        out[i][2 * q] = __byte_perm(e4l, e4h, 0x5140u);      // d0 d1 d2 d3
        out[i][2 * q + 1] = __byte_perm(e4l, e4h, 0x7362u);  // d4 d5 d6 d7
      }

      // rope: 4-elem group #s of token cand (16 groups cover 64 elems)
      if (rope_smem != nullptr) {
        const uint8_t* rslot = rope_smem + (size_t)cand * 128;
        const uint32_t rw =
            *reinterpret_cast<const uint32_t*>(rslot + NVFP4_ROPE_LANDING_OFF + 4 * s);
        nvfp4_fp8x4_to_bf16x4(rw, rope_out[i][0], rope_out[i][1]);
      }
    }
    bar();
    // --- write phase ---
#pragma unroll
    for (int i = 0; i < SEG_PER_THREAD; i++) {
      const int seg = phase * SEGS_PER_PHASE + i * THREADS + tid;
      const int cand = seg >> 4;
      const int s = seg & 15;
      uint8_t* slot = kv_smem + (size_t)cand * smem_stride;
      uint2* dst = reinterpret_cast<uint2*>(slot + 32 * s);
#pragma unroll
      for (int q = 0; q < 4; q++) dst[q] = make_uint2(out[i][2 * q], out[i][2 * q + 1]);
      if ((s & 3) == 0) {
        reinterpret_cast<float*>(slot + 512)[s >> 2] = tile_sc[i];
      }
      if (rope_smem != nullptr) {
        uint2* rdst = reinterpret_cast<uint2*>(rope_smem + (size_t)cand * 128 + 8 * s);
        *rdst = make_uint2(rope_out[i][0], rope_out[i][1]);
      }
    }
    bar();
  }
}
