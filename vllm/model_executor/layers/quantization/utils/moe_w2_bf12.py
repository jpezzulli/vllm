# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf12 — lossless 12-bit residency for the BF16 dense remainder (Phase 0).

Opt-in via VLLM_MOE_W2_BF12=1. After weight load, every LinearBase whose
quant method is UnquantizedLinearMethod and whose weight is a large BF16
matrix (attention projections, shared experts, dense MLP, DSA indexer
projections, router gates — everything the checkpoint's exclude_modules
keep at BF16) is re-packed into a lossless 12-bit container and the BF16
parameter is freed:

  BF16 word = HI byte (sign+e7..e1) : LO byte (e0+m6..m0). Measured on the
  real GLM-5.2/Kimi-K2.7 checkpoints, HI takes 30-36 distinct values and
  the top-7 exponent classes x sign cover 99.9%+ of weights, so:

  nibble plane  u8[n/2]  code = (sign<<3) | ecls, 2 codes/byte
  LO plane      u8[n]    raw low bytes (mantissa plane is ~uniform)
  escapes       flat scatter list (byte position, true HI byte), ~3e-4
  LUT16         u8[16]   code -> HI byte (sign folded in; 7/15 = escape)

  12.05 bits/weight, bit-exact by construction and re-verified against
  the live tensor at every boot (VLLM_MOE_W2_BF12_VERIFY=0 opts out).
  Full-checkpoint sweeps: GLM-5.2 53.19->40.04 GiB, Kimi-K2.7
  22.71->17.10 GiB, roundtrip PASS on every tensor (internal/bf12).

Phase 0 forward = decode into a shared scratch (fixed address, allocated
before any cudagraph capture; decode is 4 strided byte-copies + one LUT
gather + one escape scatter, all stream-ordered torch ops -> capture- and
compile-safe) followed by the stock unquantized GEMM on the scratch view.
Outputs are bit-identical to the uncompressed path (same GEMM, identical
bytes). This trades a little decode bandwidth for VRAM; the read-in-place
SASS kernel is Phase 1.

Not converted here: embed_tokens/lm_head (not LinearBase; and the scratch
for a vocab-sharded lm_head would exceed its own saving — Phase 1 reads
planes in place instead) and the MTP drafter's routed experts (FusedMoE,
not LinearBase; Phase 1 target through the moe-desc kernel family).
"""

import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def enabled() -> bool:
    return os.getenv("VLLM_MOE_W2_BF12", "0") == "1"


def _min_numel() -> int:
    # Default 1M weights (2 MiB BF16): below that the plane/scratch
    # bookkeeping isn't worth the launch overhead.
    return int(os.getenv("VLLM_MOE_W2_BF12_MIN_NUMEL", str(1 << 20)))


def _verify() -> bool:
    return os.getenv("VLLM_MOE_W2_BF12_VERIFY", "1") == "1"


def _excludes() -> list[str]:
    raw = os.getenv("VLLM_MOE_W2_BF12_EXCLUDE", "")
    return [p for p in raw.split(",") if p]


def _paranoia() -> int:
    """Diagnostic modes (no VRAM saving, syncs — debugging only):
    1 = keep BF16 originals, bit-check the decode every forward, GEMM on
        the originals (isolates decode correctness);
    2 = same checks, but GEMM on the scratch AND a reference GEMM on the
        original, logging any output mismatch per layer (isolates the
        GEMM-on-scratch effect)."""
    return int(os.getenv("VLLM_MOE_W2_BF12_PARANOIA", "0"))


def _private_scratch() -> bool:
    """Diagnostic: dedicated decode buffer per layer (no sharing, no VRAM
    saving) — isolates shared-scratch ordering effects."""
    return os.getenv("VLLM_MOE_W2_BF12_PRIVATE", "0") == "1"


def _force_sync() -> bool:
    """Diagnostic: torch.cuda.synchronize() after every bf12 linear."""
    return os.getenv("VLLM_MOE_W2_BF12_SYNC", "0") == "1"


_LAYER_NAMES: dict[int, str] = {}
_LAYER_REFS: dict[int, torch.Tensor] = {}
_LAYER_PRIVATE: dict[int, torch.Tensor] = {}
_MISMATCH_LOGGED: set[int] = set()


# Decode scratch, sized to the largest converted layer, allocated during
# conversion (before profiling/capture -> fixed addresses for cudagraphs).
# TWO buffers per device, not one: the fused-MoE runner executes the
# shared-experts MLP on the process-wide aux stream OVERLAPPED with the
# router gate / routed experts on the main stream
# (SharedExperts._run_in_aux_stream), so a single shared buffer would be
# written concurrently by two streams (measured: degraded outputs on GLM).
# Decode+GEMM run back-to-back on ONE stream inside this op, so keying the
# buffer by "am I on the aux stream" restores plain stream-ordered
# correctness with zero cross-stream syncs, in eager and under capture
# alike (capture records the real stream objects, ids preserved).
_SCRATCH: dict[tuple, torch.Tensor] = {}
_AUX_STREAM_ID: int | None = None


def _aux_stream_id() -> int | None:
    global _AUX_STREAM_ID
    if _AUX_STREAM_ID is None:
        try:
            from vllm.utils.torch_utils import aux_stream
            aux = aux_stream()
            _AUX_STREAM_ID = -1 if aux is None else aux.stream_id
        except Exception:  # noqa: BLE001 - no aux overlap on this platform
            _AUX_STREAM_ID = -1
    return _AUX_STREAM_ID


def _scratch_key(dev: torch.device) -> tuple:
    on_aux = torch.cuda.current_stream(dev).stream_id == _aux_stream_id()
    return (dev.index, "aux" if on_aux else "main")


def _scratch_u8(dev: torch.device, nbytes: int) -> torch.Tensor:
    buf = _SCRATCH.get(_scratch_key(dev))
    if buf is None or buf.numel() < nbytes:
        raise RuntimeError(
            f"bf12 scratch missing/small for {_scratch_key(dev)}: "
            f"need {nbytes} B — layer converted after scratch sizing?")
    return buf[:nbytes]


def encode_bf12(w: torch.Tensor):
    """BF16 [N, K] (CUDA or CPU) -> (nib, lo, esc_bytepos, esc_hi, lut16).

    Lossless: decode_bf12() reproduces w bit-for-bit. numel must be even.
    """
    assert w.dtype == torch.bfloat16 and w.numel() % 2 == 0
    w_u8 = w.reshape(-1).view(torch.uint8)      # little-endian byte pairs
    hi = w_u8[1::2].contiguous()
    lo = w_u8[0::2].contiguous()

    ecls = (hi & 0x7F).to(torch.long)
    top7 = torch.bincount(ecls, minlength=128).argsort(descending=True)[:7]
    lut7 = top7.to(torch.uint8)

    # class index per weight; 7 = escape
    inv = torch.full((128,), 7, dtype=torch.uint8, device=w.device)
    inv[top7] = torch.arange(7, dtype=torch.uint8, device=w.device)
    cls = inv[ecls]
    codes = ((hi >> 7) << 3) | cls

    nib = codes[0::2] | (codes[1::2] << 4)

    esc_idx = torch.nonzero(cls == 7, as_tuple=False).reshape(-1)
    esc_bytepos = esc_idx * 2 + 1          # HI byte offset in LE u8 view
    esc_hi = hi[esc_idx]

    # LUT16: code (incl. sign bit) -> HI byte; escape slots hold a
    # placeholder that the scatter always overwrites.
    lut16 = torch.zeros(16, dtype=torch.uint8, device=w.device)
    lut16[:7] = lut7
    lut16[8:15] = lut7 | 0x80
    return nib, lo, esc_bytepos, esc_hi, lut16


def decode_bf12(layer, out_u8: torch.Tensor) -> None:
    """Decode a converted layer's planes into out_u8 (u8[2n], LE bf16)."""
    _decode_into(layer.bf12_nib, layer.bf12_lo, layer.bf12_esc_bytepos,
                 layer.bf12_esc_hi, layer.bf12_lut16, out_u8)


def _decode_into(nib, lo, esc_bytepos, esc_hi, lut16,
                 out_u8: torch.Tensor) -> None:
    out_u8[0::2] = lo
    out_u8[1::4] = lut16[(nib & 0xF).to(torch.long)]
    out_u8[3::4] = lut16[(nib >> 4).to(torch.long)]
    out_u8[esc_bytepos] = esc_hi


# The decode+GEMM pair is one opaque custom op. The shared scratch is
# resolved INSIDE the op (not passed as an argument): each call fully
# rewrites the region it reads, so the op is pure from the compiler's
# point of view — nothing to functionalize, clone or reorder against.
# Kernels still record in stream order under cudagraph capture, and the
# scratch address is fixed (allocated before any capture).
def _bf12_linear(
    x: torch.Tensor,
    nib: torch.Tensor,
    lo: torch.Tensor,
    esc_bytepos: torch.Tensor,
    esc_hi: torch.Tensor,
    lut16: torch.Tensor,
    n: int,
    k: int,
    layer_id: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    priv = _LAYER_PRIVATE.get(layer_id)
    out_u8 = priv if priv is not None else _scratch_u8(x.device, n * k * 2)
    _decode_into(nib, lo, esc_bytepos, esc_hi, lut16, out_u8)
    w = out_u8.view(torch.bfloat16).view(n, k)
    ref = _LAYER_REFS.get(layer_id)
    if ref is not None:
        if layer_id not in _MISMATCH_LOGGED:
            bad = (out_u8 != ref.reshape(-1).view(torch.uint8)).sum().item()
            if bad:
                _MISMATCH_LOGGED.add(layer_id)
                logger.warning("bf12 PARANOIA: %s decodes %d/%d bytes wrong",
                               _LAYER_NAMES.get(layer_id, layer_id), bad,
                               out_u8.numel())
        if _paranoia() >= 2:
            y_s = dispatch_unquantized_gemm()(None, x, w, bias)
            y_r = dispatch_unquantized_gemm()(None, x, ref, bias)
            if not torch.equal(y_s, y_r):
                key = -layer_id - 1  # separate log-once space for GEMM diffs
                if key not in _MISMATCH_LOGGED:
                    _MISMATCH_LOGGED.add(key)
                    d = (y_s.float() - y_r.float()).abs().max().item()
                    logger.warning(
                        "bf12 PARANOIA2: %s GEMM scratch!=ref  M=%d maxdiff "
                        "%.3e", _LAYER_NAMES.get(layer_id, layer_id),
                        x.shape[0] if x.dim() == 2 else -1, d)
            return y_s
        w = ref  # mode 1: forwards always use the original weight
    if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
        return linear_batch_invariant(x, w, bias)
    y = dispatch_unquantized_gemm()(None, x, w, bias)
    if _force_sync():
        torch.cuda.synchronize()
    return y


def _bf12_linear_fake(
    x: torch.Tensor,
    nib: torch.Tensor,
    lo: torch.Tensor,
    esc_bytepos: torch.Tensor,
    esc_hi: torch.Tensor,
    lut16: torch.Tensor,
    n: int,
    k: int,
    layer_id: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    "bf12_linear",
    _bf12_linear,
    fake_impl=_bf12_linear_fake,
)


class Bf12LinearMethod(UnquantizedLinearMethod):
    """Unquantized linear over bf12 planes: decode to scratch, then the
    stock unquantized GEMM. Weight bytes identical to the BF16 original
    -> outputs bit-identical to the uncompressed path."""

    def __init__(self, n: int, k: int):
        super().__init__()
        self._shape = (n, k)

    def apply(self, layer, x, bias=None):
        n, k = self._shape
        return torch.ops.vllm.bf12_linear(
            x, layer.bf12_nib, layer.bf12_lo, layer.bf12_esc_bytepos,
            layer.bf12_esc_hi, layer.bf12_lut16, n, k,
            getattr(layer, "bf12_layer_id", -1), bias)


def _convert_layer(layer: LinearBase, name: str) -> int:
    """Encode one layer in place; returns bytes freed (0 if skipped)."""
    w = layer.weight.data
    nib, lo, esc_bytepos, esc_hi, lut16 = encode_bf12(w)

    if _verify():
        out = torch.empty(w.numel() * 2, dtype=torch.uint8, device=w.device)
        probe = type("P", (), {})()
        probe.bf12_nib, probe.bf12_lo = nib, lo
        probe.bf12_esc_bytepos, probe.bf12_esc_hi = esc_bytepos, esc_hi
        probe.bf12_lut16 = lut16
        decode_bf12(probe, out)
        # byte-view compare: true bitwise identity (bf16 NaN != NaN)
        if not torch.equal(out, w.reshape(-1).view(torch.uint8)):
            logger.warning("bf12: %s failed bit-exact verify — left as BF16",
                           name)
            return 0

    n, k = w.shape
    layer_id = len(_LAYER_NAMES)
    _LAYER_NAMES[layer_id] = name
    if _paranoia():
        _LAYER_REFS[layer_id] = w.clone()
    if _private_scratch():
        _LAYER_PRIVATE[layer_id] = torch.empty(
            n * k * 2, dtype=torch.uint8, device=w.device)
    del layer._parameters["weight"]
    layer.register_buffer("bf12_nib", nib, persistent=False)
    layer.register_buffer("bf12_lo", lo, persistent=False)
    layer.register_buffer("bf12_esc_bytepos", esc_bytepos, persistent=False)
    layer.register_buffer("bf12_esc_hi", esc_hi, persistent=False)
    layer.register_buffer("bf12_lut16", lut16, persistent=False)
    layer.bf12_layer_id = layer_id
    layer.quant_method = Bf12LinearMethod(n, k)

    raw = n * k * 2
    packed = (nib.numel() + lo.numel() + esc_hi.numel()
              + esc_bytepos.numel() * 8 + 16)
    return raw - packed


def convert_model(model: torch.nn.Module, vllm_config=None) -> None:
    """Walk the loaded model and convert every eligible BF16 Linear.

    Runs once post-load, pre-capture. Never raises past the caller's
    non-fatal guard; a layer that fails verify simply stays BF16.
    """
    if not enabled():
        return
    if vllm_config is not None and vllm_config.parallel_config.use_ubatching:
        # The shared decode scratch is stream-ordered; DBO runs two
        # interleaved forwards -> would race. Not a Phase-0 config.
        logger.warning("bf12: disabled under ubatching/DBO")
        return

    min_numel = _min_numel()
    excludes = _excludes()
    targets: list[tuple[str, LinearBase]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase):
            continue
        if type(mod.quant_method) is not UnquantizedLinearMethod:
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2:
            continue
        if w.numel() < min_numel or w.numel() % 2:
            continue
        if any(p in name for p in excludes):
            continue
        targets.append((name, mod))
    if not targets:
        return

    dev = targets[0][1].weight.device
    scratch_bytes = max(m.weight.numel() for _, m in targets) * 2
    for key in ((dev.index, "main"), (dev.index, "aux")):
        cur = _SCRATCH.get(key)
        if cur is None or cur.numel() < scratch_bytes:
            # grow-only: convert_model runs once per model (target then
            # drafter) and every converted layer must fit its buffer
            _SCRATCH[key] = torch.empty(scratch_bytes, dtype=torch.uint8,
                                        device=dev)

    freed = 0
    esc_total = 0
    n_conv = 0
    for name, mod in targets:
        saved = _convert_layer(mod, name)
        if saved:
            freed += saved
            esc_total += mod.bf12_esc_hi.numel()
            n_conv += 1
    torch.cuda.empty_cache()
    logger.info(
        "bf12: %d/%d BF16 linears -> 12-bit planes, %.2f GiB freed "
        "(scratch 2x%.0f MiB main+aux, escapes %.1e, verify %s)",
        n_conv, len(targets), freed / (1 << 30),
        scratch_bytes / (1 << 20),
        esc_total / max(1, sum(m.bf12_lo.numel() for _, m in targets
                               if hasattr(m, "bf12_lo"))),
        "on" if _verify() else "off")
