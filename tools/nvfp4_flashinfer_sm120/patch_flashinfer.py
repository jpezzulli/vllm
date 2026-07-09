"""Patch FlashInfer's JIT sources with KV-NVFP4: ModelType::GLM_NSA_NVFP4.

Packed gmem layout V1 (352 B/token): the decode and prefill kernels land the
packed bulk copies at the tail of the existing smem slots and expand them
in-place to the GLM_NSA layout (528 B: e4m3 + 4x fp32 tile-128 scales)
before first use. The rest of the pipeline (FP8 MMA, softmax, XV, the
W-residual for arbitrary scales) is unchanged.

Idempotent; hard anchors (assert). Run inside the serving docker image
against the installed flashinfer package.
"""
import pathlib
import sys

FI = pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer")
INC = FI / "data/include/flashinfer/attention/sparse_mla_sm120"
CSRC = FI / "data/csrc"
MARK = "GLM_NSA_NVFP4"


def patch(path: pathlib.Path, old: str, new: str, count: int = 1):
    s = path.read_text()
    if new in s:
        return
    assert old in s, f"ANCHOR NOT FOUND in {path}:\n{old[:200]}"
    assert s.count(old) == count, f"anchor x{s.count(old)} != {count} in {path}"
    path.write_text(s.replace(old, new))
    print(f"patched: {path.name}: {old[:60]!r}...")


def append_once(path: pathlib.Path, text: str):
    s = path.read_text()
    if MARK in s and text.strip()[:40] in s:
        return
    path.write_text(s + text)
    print(f"appended: {path.name}")


# ---------------------------------------------------------------- 0. naglowek
expand = (pathlib.Path(__file__).parent / "nvfp4_expand.cuh").read_text()
(INC / "common/nvfp4_expand.cuh").write_text(expand)
print("installed: common/nvfp4_expand.cuh")

# ------------------------------------------------------------- 1. model_type.h
patch(
    INC / "model/model_type.h",
    "enum class ModelType { DSV3_2, DSV4, GLM_NSA };",
    "enum class ModelType { DSV3_2, DSV4, GLM_NSA, GLM_NSA_NVFP4 };",
)

# -------------------------------------------------------- 2. kv_cache_traits
# pola domyslne w DSV3_2 (dziedziczone przez GLM_NSA) i DSV4
patch(
    INC / "model/kv_cache_traits.cuh",
    """  // Q nope stride (padded for ldmatrix alignment + bank conflict avoidance)
  static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;  // 528
  // Unused for DSV3_2 prefill; declared so SmemLayout<DSV3_2, BF16> compiles.
  static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;  // 520""",
    """  // NVFP4: defaults - no landing offset, bf16 rope from gmem
  static constexpr bool IS_NVFP4 = false;
  static constexpr int KV_SMEM_LANDING_OFF = 0;
  static constexpr int ROPE_GMEM_BYTES = D_ROPE * (int)sizeof(bf16);  // 128
  static constexpr int ROPE_SMEM_LANDING_OFF = 0;

  // Q nope stride (padded for ldmatrix alignment + bank conflict avoidance)
  static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;  // 528
  // Unused for DSV3_2 prefill; declared so SmemLayout<DSV3_2, BF16> compiles.
  static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;  // 520""",
)
patch(
    INC / "model/kv_cache_traits.cuh",
    """  // Q nope stride
  static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;      // 464
  static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;  // 456 bf16 (912 B)""",
    """  // NVFP4: defaults
  static constexpr bool IS_NVFP4 = false;
  static constexpr int KV_SMEM_LANDING_OFF = 0;
  static constexpr int ROPE_GMEM_BYTES = D_ROPE * (int)sizeof(bf16);  // 128
  static constexpr int ROPE_SMEM_LANDING_OFF = 0;

  // Q nope stride
  static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;      // 464
  static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;  // 456 bf16 (912 B)""",
)
# specjalizacja NVFP4: gmem 352 B, smem po ekspansji identyczne z GLM_NSA
patch(
    INC / "model/kv_cache_traits.cuh",
    "template <>\nstruct KVCacheTraits<ModelType::GLM_NSA> : KVCacheTraits<ModelType::DSV3_2> {\n  static constexpr ScaleFormat SCALE_FORMAT = ScaleFormat::ARBITRARY_FP32;\n};",
    """template <>
struct KVCacheTraits<ModelType::GLM_NSA> : KVCacheTraits<ModelType::DSV3_2> {
  static constexpr ScaleFormat SCALE_FORMAT = ScaleFormat::ARBITRARY_FP32;
};

// NVFP4: packed storage, 352 B/token (layout V1):
//   [0:256) 512xE2M1, [256:288) 32xE4M3 block-16 scales, [288:352) 64xFP8 rope.
// Smem after the prologue expansion equals the GLM_NSA layout (528 B +
// fp32 tile scales), so the whole MMA pipeline inherits GLM_NSA behavior.
template <>
struct KVCacheTraits<ModelType::GLM_NSA_NVFP4> : KVCacheTraits<ModelType::GLM_NSA> {
  static constexpr bool IS_NVFP4 = true;
  static constexpr int KV_GMEM_STRIDE = 352;
  static constexpr int KV_SCALE_GMEM_OFFSET = 256;
  static constexpr int KV_ROPE_GMEM_OFFSET = 288;
  static constexpr int KV_SMEM_COPY_BYTES = 288;          // bulk 1 (nibbles+scales)
  static constexpr int KV_SMEM_LANDING_OFF = 528 - 288;   // 240
  static constexpr int ROPE_GMEM_BYTES = 64;              // bulk 2 (fp8 rope)
  static constexpr int ROPE_SMEM_LANDING_OFF = 128 - 64;  // 64
};""",
)

# --------------------------------------------------- 3. kv_cache_io (prefill)
patch(
    INC / "common/kv_cache_io.cuh",
    """    if constexpr (USE_L2_HINT)
      cp_async_bulk_g2s_l2hint(dst + bi * SMEM_STRIDE, src, COPY_BYTES, mbar, cache_policy);
    else
      cp_async_bulk_g2s(dst + bi * SMEM_STRIDE, src, COPY_BYTES, mbar);""",
    """    if constexpr (USE_L2_HINT)
      cp_async_bulk_g2s_l2hint(dst + bi * SMEM_STRIDE + KV::KV_SMEM_LANDING_OFF, src, COPY_BYTES,
                               mbar, cache_policy);
    else
      cp_async_bulk_g2s(dst + bi * SMEM_STRIDE + KV::KV_SMEM_LANDING_OFF, src, COPY_BYTES, mbar);""",
)

# ------------------------------------------------------ 4. decode dsv3_2 cuh
DEC = INC / "decode_dsv3_2_kernel.cuh"
patch(
    DEC,
    '#include "model/scale_convert.cuh"',
    '#include "common/nvfp4_expand.cuh"\n#include "model/scale_convert.cuh"',
)
patch(
    DEC,
    """  constexpr uint32_t V2_BULK_NOPESC_BYTES = (uint32_t)KV_SMEM_STRIDE;         // 528
  constexpr uint32_t V2_BULK_ROPE_BYTES = (uint32_t)D_ROPE_C * sizeof(bf16);  // 128""",
    """  constexpr uint32_t V2_BULK_NOPESC_BYTES = (uint32_t)KV::KV_SMEM_COPY_BYTES;  // 528 / 288
  constexpr uint32_t V2_BULK_ROPE_BYTES = (uint32_t)KV::ROPE_GMEM_BYTES;       // 128 / 64""",
)
patch(
    DEC,
    """      // Bulk 1: NoPE + INLINE scales (528 B) → sm_kv_fp8 slot.
      cp_async_bulk_g2s(kv_fp8_dst + (size_t)entry_idx * KV_SMEM_STRIDE, data_base,
                        V2_BULK_NOPESC_BYTES, sm.mbar_full(buf));
      // Bulk 2: RoPE (128 B) → sm_kv_rope slot.
      cp_async_bulk_g2s(kv_rope_dst + (size_t)entry_idx * D_ROPE_C, data_base + KV_ROPE_OFFSET,
                        V2_BULK_ROPE_BYTES, sm.mbar_full(buf));""",
    """      // Bulk 1: NoPE(+scales) -> tail of the sm_kv_fp8 slot (NVFP4: landing 240).
      cp_async_bulk_g2s(
          kv_fp8_dst + (size_t)entry_idx * KV_SMEM_STRIDE + KV::KV_SMEM_LANDING_OFF, data_base,
          V2_BULK_NOPESC_BYTES, sm.mbar_full(buf));
      // Bulk 2: RoPE -> tail of the sm_kv_rope slot (NVFP4: landing 64).
      cp_async_bulk_g2s(reinterpret_cast<uint8_t*>(kv_rope_dst) +
                            (size_t)entry_idx * D_ROPE_C * sizeof(bf16) +
                            KV::ROPE_SMEM_LANDING_OFF,
                        data_base + KV_ROPE_OFFSET, V2_BULK_ROPE_BYTES, sm.mbar_full(buf));""",
)
patch(
    DEC,
    """    uint8_t* sm_kv_fp8 = sm.kv_fp8(buf);
    bf16* sm_kv_rope = sm.kv_rope(buf);

    // ── Stage 2 QK ────────────────────────────────────────────""",
    """    uint8_t* sm_kv_fp8 = sm.kv_fp8(buf);
    bf16* sm_kv_rope = sm.kv_rope(buf);

    // NVFP4: in-place expansion of the packed tile before first use.
    if constexpr (KV::IS_NVFP4) {
      nvfp4_expand_tile<DSV3_2_MATH_THREADS, DSV3_2_BI>(
          sm_kv_fp8, KV_SMEM_STRIDE, reinterpret_cast<uint8_t*>(sm_kv_rope), threadIdx.x,
          [] { bar_sync_t<3, DSV3_2_MATH_THREADS>(); });
    }

    // ── Stage 2 QK ────────────────────────────────────────────""",
)

# ------------------------------------------------- 5. decode dsv3_2 dispatch
DECCU = CSRC / "sparse_mla_sm120_decode_dsv3_2.cu"
patch(
    DECCU,
    """    if (mt == ModelType::DSV3_2) {                 \\
      DSV3_2_DISPATCH_MT(ModelType::DSV3_2, H, K)  \\
    } else if (mt == ModelType::GLM_NSA) {         \\
      DSV3_2_DISPATCH_MT(ModelType::GLM_NSA, H, K) \\
    }                                              \\""",
    """    if (mt == ModelType::DSV3_2) {                             \\
      DSV3_2_DISPATCH_MT(ModelType::DSV3_2, H, K)              \\
    } else if (mt == ModelType::GLM_NSA) {                     \\
      DSV3_2_DISPATCH_MT(ModelType::GLM_NSA, H, K)             \\
    } else if (mt == ModelType::GLM_NSA_NVFP4) {               \\
      DSV3_2_DISPATCH_MT(ModelType::GLM_NSA_NVFP4, H, K)       \\
    }                                                          \\""",
)

# ------------------------------------------------------- 6. jit binding (FFI)
BIND = CSRC / "sparse_mla_sm120_jit_binding.cu"
patch(
    BIND,
    """  const auto mt = static_cast<ModelType>(model_type);
  TVM_FFI_ICHECK(mt == ModelType::DSV3_2 || mt == ModelType::GLM_NSA)
      << "decode-dsv3_2 expects model_type DSV3_2 or GLM_NSA; got " << model_type;

  constexpr int BPT_DSV3_2 = 656;
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, BPT_DSV3_2, "kv_cache");""",
    """  const auto mt = static_cast<ModelType>(model_type);
  TVM_FFI_ICHECK(mt == ModelType::DSV3_2 || mt == ModelType::GLM_NSA ||
                 mt == ModelType::GLM_NSA_NVFP4)
      << "decode-dsv3_2 expects model_type DSV3_2/GLM_NSA/GLM_NSA_NVFP4; got " << model_type;

  const int bpt_v32 = (mt == ModelType::GLM_NSA_NVFP4) ? 352 : 656;
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, bpt_v32, "kv_cache");""",
)

# --------------------------------------------------- 7. orchestrator (prefill)
ORCH = CSRC / "sparse_mla_sm120.cu"
patch(
    ORCH,
    """    const auto mt = static_cast<ModelType>(model_type);
    TVM_FFI_ICHECK(mt == ModelType::DSV3_2 || mt == ModelType::GLM_NSA)
        << "d_qk=576 supports model_type auto, DSV3_2, or GLM_NSA; got " << model_type;
    return mt;""",
    """    const auto mt = static_cast<ModelType>(model_type);
    TVM_FFI_ICHECK(mt == ModelType::DSV3_2 || mt == ModelType::GLM_NSA ||
                   mt == ModelType::GLM_NSA_NVFP4)
        << "d_qk=576 supports model_type auto/DSV3_2/GLM_NSA/GLM_NSA_NVFP4; got " << model_type;
    return mt;""",
)
patch(
    ORCH,
    """inline int bytes_per_token(ModelType mt) {
  switch (mt) {
    case ModelType::DSV3_2:
    case ModelType::GLM_NSA:
      return 656;""",
    """inline int bytes_per_token(ModelType mt) {
  switch (mt) {
    case ModelType::DSV3_2:
    case ModelType::GLM_NSA:
      return 656;
    case ModelType::GLM_NSA_NVFP4:
      return 352;""",
)

# ------------------------------------------------------ 8. prefill dispatch
PRE = CSRC / "sparse_mla_sm120_prefill.cu"
patch(
    PRE,
    """    case ModelType::GLM_NSA:
      return dispatch_v32<ModelType::GLM_NSA>(num_heads, topk, Q, KV_cache, indices, attn_sink,
                                              output, out_lse, sm_scale, num_tokens,
                                              stride_kv_block, topk_length, stream);""",
    """    case ModelType::GLM_NSA:
      return dispatch_v32<ModelType::GLM_NSA>(num_heads, topk, Q, KV_cache, indices, attn_sink,
                                              output, out_lse, sm_scale, num_tokens,
                                              stride_kv_block, topk_length, stream);
    case ModelType::GLM_NSA_NVFP4:
      return dispatch_v32<ModelType::GLM_NSA_NVFP4>(num_heads, topk, Q, KV_cache, indices,
                                                    attn_sink, output, out_lse, sm_scale,
                                                    num_tokens, stride_kv_block, topk_length,
                                                    stream);""",
)

# ---------------------------------------------------- 9. prefill kernel cuh
PREK = INC / "prefill_kernel.cuh"
patch(
    PREK,
    '#include "common/online_softmax.cuh"',
    '#include "common/nvfp4_expand.cuh"\n#include "common/online_softmax.cuh"',
)
# SG: ekspansja na poczatku iteracji (kafel juz zaladowany przez mbar wait)
patch(
    PREK,
    """    for (int ti = 0; ti < actual_ni; ti++) {
      uint8_t* kv_smem = sm.kv_bufs[ti & 1];
      const int32_t* ib = idx_base + ti * BI;
      const int qk_nb = mwarp * ENTRIES_PER_WARP;
      uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;""",
    """    for (int ti = 0; ti < actual_ni; ti++) {
      uint8_t* kv_smem = sm.kv_bufs[ti & 1];
      // NVFP4: in-place expansion before the tile is first used.
      if constexpr (KV::IS_NVFP4) {
        nvfp4_expand_tile<MATH_THREADS, BI>(kv_smem, KV::KV_SMEM_STRIDE, nullptr, threadIdx.x,
                                            [] { bar_sync_t<2, MATH_THREADS>(); });
      }
      const int32_t* ib = idx_base + ti * BI;
      const int qk_nb = mwarp * ENTRIES_PER_WARP;
      uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;""",
)
# MG: jw.
patch(
    PREK,
    """    for (int ti = 0; ti < loop_bound; ti++) {
      uint8_t* kv_smem = sm.kv_buf(ti & 1);
      const int qk_nb = mwarp * ENTRIES_PER_WARP;
      uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;""",
    """    for (int ti = 0; ti < loop_bound; ti++) {
      uint8_t* kv_smem = sm.kv_buf(ti & 1);
      // NVFP4: in-place expansion before the tile is first used.
      if constexpr (KV::IS_NVFP4) {
        nvfp4_expand_tile<MATH_THREADS, BI>(kv_smem, KV::KV_SMEM_STRIDE, nullptr, threadIdx.x,
                                            [] { bar_sync_t<2, MATH_THREADS>(); });
      }
      const int qk_nb = mwarp * ENTRIES_PER_WARP;
      uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;""",
)
# rope w prefillu czytane z GMEM: dekod e4m3->bf16 przy prefetchu (SG + MG)
patch(
    PREK,
    """      KVRopePrefetch rope_pf = prefetch_kv_rope(
          reinterpret_cast<const bf16*>(entry_base[gid] + KV::KV_ROPE_GMEM_OFFSET), lane);""",
    """      KVRopePrefetch rope_pf;
      if constexpr (KV::IS_NVFP4) {
        rope_pf = nvfp4_prefetch_kv_rope_t<KVRopePrefetch>(
            entry_base[gid] + KV::KV_ROPE_GMEM_OFFSET, lane);
      } else {
        rope_pf = prefetch_kv_rope(
            reinterpret_cast<const bf16*>(entry_base[gid] + KV::KV_ROPE_GMEM_OFFSET), lane);
      }""",
)
patch(
    PREK,
    """      KVRopePrefetch rope_pf = prefetch_kv_rope(
          reinterpret_cast<const bf16*>(entry_base_gid + KV::KV_ROPE_GMEM_OFFSET), lane);""",
    """      KVRopePrefetch rope_pf;
      if constexpr (KV::IS_NVFP4) {
        rope_pf = nvfp4_prefetch_kv_rope_t<KVRopePrefetch>(
            entry_base_gid + KV::KV_ROPE_GMEM_OFFSET, lane);
      } else {
        rope_pf = prefetch_kv_rope(
            reinterpret_cast<const bf16*>(entry_base_gid + KV::KV_ROPE_GMEM_OFFSET), lane);
      }""",
)

# --------------------------------------------------------- 10. python (JIT py)
PY = FI / "mla/_sparse_mla_sm120.py"
patch(
    PY,
    '_KV_SCALE_FORMATS = frozenset({"auto", "pow2_fp32", "arbitrary_fp32"})',
    '_KV_SCALE_FORMATS = frozenset({"auto", "pow2_fp32", "arbitrary_fp32", "nvfp4_b16"})',
)
patch(
    PY,
    "_MODEL_TYPE_GLM_NSA = 2",
    "_MODEL_TYPE_GLM_NSA = 2\n_MODEL_TYPE_GLM_NSA_NVFP4 = 3",
)
patch(
    PY,
    """    if d_qk == 576:
        if fmt == "arbitrary_fp32":
            return _MODEL_TYPE_GLM_NSA
        return _MODEL_TYPE_DSV3_2""",
    """    if d_qk == 576:
        if fmt == "nvfp4_b16":
            return _MODEL_TYPE_GLM_NSA_NVFP4
        if fmt == "arbitrary_fp32":
            return _MODEL_TYPE_GLM_NSA
        return _MODEL_TYPE_DSV3_2""",
)
patch(
    PY,
    """def _bytes_per_token_for_model_type(model_type: int) -> int:
    if model_type in (_MODEL_TYPE_DSV3_2, _MODEL_TYPE_GLM_NSA):
        return _BPT_DSV3_2""",
    """def _bytes_per_token_for_model_type(model_type: int) -> int:
    if model_type == _MODEL_TYPE_GLM_NSA_NVFP4:
        return 352
    if model_type in (_MODEL_TYPE_DSV3_2, _MODEL_TYPE_GLM_NSA):
        return _BPT_DSV3_2""",
)
patch(
    PY,
    """        if model_type in (
            _MODEL_TYPE_DSV3_2,
            _MODEL_TYPE_GLM_NSA,
        ) and _decode_dsv3_2_dispatchable(num_tokens, num_heads, topk, d_qk, kv_pbs):""",
    """        if model_type in (
            _MODEL_TYPE_DSV3_2,
            _MODEL_TYPE_GLM_NSA,
            _MODEL_TYPE_GLM_NSA_NVFP4,
        ) and _decode_dsv3_2_dispatchable(num_tokens, num_heads, topk, d_qk, kv_pbs):""",
)

# --------------------------------------------- 11. python (_core.py, wrapper)
CORE = FI / "mla/_core.py"
patch(
    CORE,
    """    if kv_cache.ndim == 3:
        if kv_cache.size(-1) != 656:
            raise ValueError(
                "SM120 sparse MLA v32/GLM expects packed kv_cache last dim 656, "
                f"got {tuple(kv_cache.shape)}"
            )
        return kv_cache
    if kv_cache.ndim == 4:
        if kv_cache.size(1) != 1 or kv_cache.size(-1) != 656:""",
    """    if kv_cache.ndim == 3:
        if kv_cache.size(-1) not in (656, 352):
            raise ValueError(
                "SM120 sparse MLA v32/GLM expects packed kv_cache last dim 656, "
                f"got {tuple(kv_cache.shape)}"
            )
        return kv_cache
    if kv_cache.ndim == 4:
        if kv_cache.size(1) != 1 or kv_cache.size(-1) not in (656, 352):""",
)

# syntax check
import ast

ast.parse(PY.read_text())
ast.parse(CORE.read_text())
print("SYNTAX-OK python")
print("PATCH-FLASHINFER-DONE")
