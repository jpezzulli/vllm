# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cubit hand-written SASS skinny w8a8 FP8 GEMM (SM120, experimental, opt-in).

Replaces `w8a8_triton_block_scaled_mm` for decode-shaped dense GEMMs
(M <= 4 tokens) with a hand-scheduled SASS kernel (`skinny_fp8_mm`, ABI 2):
per-CTA 16 output columns, in-CTA split-K, QMMA.SF with in-flight UE8M0
block scaling, bf16 [M, N] epilogue.

ABI 2 (fragment-major weights, direct activations):
  - B: fp8e4m3 [N, K] repacked ONCE per layer to fragment-major (per 64-byte
    k-block, lane t's 16 bytes contiguous) -> one LDG.E.128 per tile per k64,
    79% DRAM roofline at K=4096 (was 68% with narrow loads). The repacked
    copy is cached next to the layer's weight (dual-residency).
  - A: fp8e4m3 [M, K] row-major read DIRECTLY (no act-pack kernel, no
    qn/sfa staging buffers): at M <= 4 the QMMA A-fragment is mostly pad
    zeros; the kernel loads the few real bytes under a g<M predicate.
  - As: f32 [M, K/128] POW2 scales read directly; the kernel extracts the
    e8m0 exponent in-flight (activations must be quantized with
    use_ue8m0=True -- the linear kernel class does).
  - Bs: [N/128, K/128] power-of-two scales converted once to E8M0 bytes.
  - N % 16 == 0, K in the per-K cubin set, M <= 4.
  - Launches via the CUDA driver API on the current torch stream:
    CUDA-graph capturable (same discipline as cubit_sparse_mla).

Any unsupported shape or setup failure -> caller falls back to Triton.
"""

import ctypes
import functools
import json
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_M = 4
_KERN = b"skinny_fp8_mm"
_DIR = os.getenv("VLLM_W8A8_SKINNY_CUBIT_DIR", "/cubit-share")
# Dual-residency control: fragment-major copies are cached ONLY for these K
# (comma list). K=4096 shapes carry ~73% of the measured win for ~half the
# extra VRAM (~1.7 GiB at TP2); K=1024 shapes are only 1.06-1.24x and fall
# back to Triton when not listed. "all" = every supported K.
_REPACK_KS = os.getenv("VLLM_W8A8_SKINNY_REPACK_KS", "2048,4096,7168")

_cu = None
_state = "uninit"           # uninit | ready | unavailable
_fns: dict[int, tuple[ctypes.c_void_p, int]] = {}   # K -> (fn, nthreads)
_B_CACHE: dict = {}         # id(weight) -> fragment-major fp8 byte tensor
_BS_CACHE: dict = {}        # id(weight_scale) -> e8m0 byte tensor


def _driver():
    global _cu
    if _cu is None:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p]
        cu.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_void_p, ctypes.c_char_p]
        _cu = cu
    return _cu


def _ck(r, what):
    if r:
        raise RuntimeError(f"skinny_fp8_cubit: CUDA error {r} in {what}")


def _ensure_ready() -> bool:
    """Load every ABI-2 FOLD-mode per-K cubin from the manifest. Eager-only.

    fold = the kernel consumes the EXACT f32 activation/weight scales
    (folded per kb128 in-kernel); no ue8m0 requant anywhere -> numerics
    match the Triton path and MTP acceptance is unaffected.
    """
    global _state
    if _state == "ready":
        return True
    if _state == "unavailable":
        return False
    try:
        man_path = os.path.join(_DIR, "skinny_fp8_manifest.json")
        with open(man_path) as f:
            manifest = json.load(f)
        cu = _driver()
        for k_str, meta in manifest.items():
            if int(meta.get("abi", 1)) != 2 or meta.get("scale") != "fold":
                continue
            path = os.path.join(_DIR, f"skinny_fp8_mm_k{k_str}.cubin")
            if not os.path.isfile(path):
                continue
            mod = ctypes.c_void_p()
            _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                "cuModuleLoad")
            fn = ctypes.c_void_p()
            _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, _KERN),
                "cuModuleGetFunction")
            _fns[int(k_str)] = (fn, int(meta["nwarp"]) * 32)
        if not _fns:
            raise FileNotFoundError(f"no ABI-2 fold-mode cubins in {_DIR}")
        _state = "ready"
        logger.info("skinny_fp8_cubit: loaded fold-mode cubins for K=%s",
                    sorted(_fns.keys()))
        return True
    except Exception as e:  # noqa: BLE001 - any failure means Triton fallback
        logger.warning("skinny_fp8_cubit unavailable (%s); Triton fallback", e)
        _state = "unavailable"
        return False


def supported_k(k: int) -> bool:
    return _state == "ready" and k in _fns


def repack_allowed(k: int) -> bool:
    ks = _REPACK_KS.strip()
    if ks.lower() == "all":
        return True
    return str(k) in {x.strip() for x in ks.split(",")}


def weight_fragment_major(weight: torch.Tensor) -> torch.Tensor | None:
    """[N, K] fp8 row-major -> fragment-major copy (cached per layer).

    Per 64-byte k-block: lane t's 16 bytes (4B x {lo0, hi0, lo1, hi1})
    land contiguously at offset 16t -> the kernel's LDG.E.128.
    Returns None when K is not in the repack allow-list (VRAM control).
    """
    key = id(weight)
    cached = _B_CACHE.get(key)
    if cached is not None:
        return cached
    n, k = weight.shape
    if not repack_allowed(k):
        return None
    w8 = weight.view(torch.uint8)
    packed = (w8.view(n, k // 64, 4, 4, 4).permute(0, 1, 3, 2, 4)
              .contiguous().view(n, k))
    _B_CACHE[key] = packed
    return packed


def weight_scale_f32(weight_scale: torch.Tensor) -> torch.Tensor:
    """[N/128, K/128] checkpoint scales -> contiguous f32 (cached).

    Fold mode consumes the scales EXACTLY as the Triton path does; no
    power-of-two requant anywhere.
    """
    key = id(weight_scale)
    cached = _BS_CACHE.get(key)
    if cached is not None:
        return cached
    ws = weight_scale
    if ws.dtype != torch.float32:
        ws = ws.float()
    ws = ws.contiguous()
    _BS_CACHE[key] = ws
    return ws


_DEBUG = os.getenv("VLLM_W8A8_SKINNY_DEBUG", "0") == "1"
_hit_shapes: set = set()
_miss_shapes: set = set()


def skinny_mm(
    A: torch.Tensor,         # [M, K] fp8e4m3 row-major
    As: torch.Tensor,        # [M, K/128] f32 scales (ARBITRARY, fold mode)
    B_packed: torch.Tensor,  # [N, K] fragment-major (weight_fragment_major)
    Bs_bytes: torch.Tensor,  # [N/128, K/128] f32 scales (weight_scale_f32)
) -> torch.Tensor | None:
    """Returns C [M, N] bf16, or None when the shape is unsupported."""
    M, K = A.shape
    N = B_packed.shape[0]
    if M > _MAX_M or N % 16 or not supported_k(K):
        if _DEBUG:
            key = (M, N, K, "shape")
            if key not in _miss_shapes:
                _miss_shapes.add(key)
                logger.info("skinny MISS (shape) M=%d N=%d K=%d", M, N, K)
        return None
    if As.dtype != torch.float32 or not As.is_contiguous():
        if _DEBUG:
            key = (M, N, K, str(As.dtype), As.is_contiguous())
            if key not in _miss_shapes:
                _miss_shapes.add(key)
                logger.info("skinny MISS (As %s contig=%s) M=%d N=%d K=%d",
                            As.dtype, As.is_contiguous(), M, N, K)
        return None
    if _DEBUG:
        key = (M, N, K)
        if key not in _hit_shapes:
            _hit_shapes.add(key)
            logger.info("skinny HIT M=%d N=%d K=%d", M, N, K)
    fn, nthr = _fns[K]

    C = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    cu = _driver()
    stream = ctypes.c_void_p(
        torch.cuda.current_stream(A.device).cuda_stream)
    args = [ctypes.c_uint64(A.data_ptr()),
            ctypes.c_uint64(As.data_ptr()),
            ctypes.c_uint64(B_packed.data_ptr()),
            ctypes.c_uint64(Bs_bytes.data_ptr()),
            ctypes.c_uint64(C.data_ptr()),
            ctypes.c_uint32(K),
            ctypes.c_uint32(K // 64),
            ctypes.c_uint32(N * 2),
            ctypes.c_uint32(K // 128),
            ctypes.c_uint32(M)]
    argv = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args])
    _ck(cu.cuLaunchKernel(fn, N // 16, 1, 1, nthr, 1, 1, 0, stream, argv,
                          None), "launch")
    return C


@functools.cache
def enabled() -> bool:
    if os.getenv("VLLM_W8A8_SKINNY_CUBIT", "0") != "1":
        return False
    from vllm.platforms import current_platform
    cap = current_platform.get_device_capability()
    major = getattr(cap, "major", None) or (cap[0] if cap else None)
    if major != 12:
        return False
    # cuModuleLoad needs a live CUDA context (kernel selection can run before
    # the first tensor op on this device)
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    return _ensure_ready()
