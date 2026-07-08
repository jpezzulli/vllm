# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment controls for the portable sparse MLA fallback."""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

_TRITON_MLA_SPARSE_ENV = "VLLM_TRITON_MLA_SPARSE"
_TRITON_MLA_SPARSE_DUMP_ENV = "VLLM_TRITON_MLA_SPARSE_DUMP"
_TRITON_MLA_SPARSE_DUMP_PATH_ENV = "VLLM_TRITON_MLA_SPARSE_DUMP_PATH"
_TRITON_MLA_SPARSE_TOPK_CHUNK_ENV = "VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE"
_TRITON_MLA_SPARSE_QUERY_CHUNK_ENV = "VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE"
_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV = (
    "VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH"
)
_TRITON_MLA_SPARSE_HEAD_BLOCK_ENV = "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE"
_TRITON_MLA_SPARSE_MATMUL_DECODE_ENV = "VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE"
_TRITON_MLA_SPARSE_MATMUL_DECODE_MAX_CANDIDATES_ENV = (
    "VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE_MAX_CANDIDATES"
)
_SPARSE_MLA_CUBIT_ENV = "VLLM_SPARSE_MLA_CUBIT"
_SPARSE_MLA_PREFILL_CACHE_DIRECT_ENV = "VLLM_SPARSE_MLA_PREFILL_CACHE_DIRECT"

_ENV_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENV_FALSE_VALUES = {"0", "false", "no", "off"}

logger = init_logger(__name__)


def _optional_env_flag(name: str) -> bool | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    value = raw_value.lower()
    if value in _ENV_TRUE_VALUES:
        return True
    if value in _ENV_FALSE_VALUES:
        return False
    return None


def _is_sm12x_device(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index)[0] == 12


def is_sparse_mla_attention_dump_enabled() -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_DUMP_ENV)
    if configured is not None:
        return configured
    return False


def sparse_mla_reference_attention_configured() -> bool | None:
    return _optional_env_flag(_TRITON_MLA_SPARSE_ENV)


def sparse_mla_prefill_cache_direct_enabled() -> bool:
    """Prefill attends straight to the paged fp8 caches via global slot ids
    (decode-style kernels) instead of materializing a dense bf16 gather of
    the whole prior context per request chunk. Bounds prefill workspace
    memory independently of context length. Only consulted on the reference
    (triton) attention path; default ON."""
    configured = _optional_env_flag(_SPARSE_MLA_PREFILL_CACHE_DIRECT_ENV)
    if configured is not None:
        return configured
    return True


def is_sparse_mla_reference_attention_enabled_for_platform() -> bool:
    configured = sparse_mla_reference_attention_configured()
    if configured is not None:
        return configured
    return current_platform.is_device_capability_family(120)


def is_sparse_mla_reference_attention_enabled(device: torch.device) -> bool:
    configured = sparse_mla_reference_attention_configured()
    if configured is not None:
        return configured
    return _is_sm12x_device(device)


def sparse_mla_cubit_enabled() -> bool:
    """Opt-in: fused hand-written SASS (cubit) sparse-MLA decode on SM120.

    Experimental. Replaces the Triton accumulate+finish pair for supported
    decode shapes (see cubit_sparse_mla.py); unsupported shapes silently fall
    back to Triton. Requires eager mode (the kernel is launched through the
    CUDA driver API and is not CUDA-graph capturable). Default: off.
    """
    configured = _optional_env_flag(_SPARSE_MLA_CUBIT_ENV)
    if configured is not None:
        return configured
    return False


def _uses_speculative_decoding(vllm_config) -> bool:
    return bool(getattr(vllm_config, "speculative_config", None))


def sparse_mla_reference_cudagraphs_allowed(vllm_config=None) -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV)
    if configured is not None:
        return configured
    return not (
        vllm_config is not None and _uses_speculative_decoding(vllm_config)
    )


def disable_sparse_mla_reference_cudagraphs_if_enabled(vllm_config) -> None:
    if not is_sparse_mla_reference_attention_enabled_for_platform():
        return
    if sparse_mla_reference_cudagraphs_allowed(vllm_config):
        logger.warning_once(
            "Keeping vLLM compile and CUDA graphs enabled for the DeepSeek V4 "
            "Triton sparse MLA fallback because "
            f"{_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV}=1 or speculative "
            "decoding is not configured. This is an "
            "experimental performance mode."
        )
        return

    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    if (
        compilation_config.mode == CompilationMode.NONE
        and compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    ):
        return

    logger.warning_once(
        "Disabling vLLM compile and CUDA graphs for the DeepSeek V4 Triton "
        "sparse MLA fallback because the current fallback path is not "
        "compile/graph-safe yet, or because speculative decoding uses "
        "multi-token sparse MLA decode."
    )
    compilation_config.mode = CompilationMode.NONE
    compilation_config.compile_sizes = []
    compilation_config.compile_ranges_endpoints = []
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    compilation_config.cudagraph_capture_sizes = []
    compilation_config.max_cudagraph_capture_size = 0


def sparse_mla_attention_dump_path() -> str:
    return (
        os.getenv(_TRITON_MLA_SPARSE_DUMP_PATH_ENV)
        or "/tmp/deepseek_v4_triton_mla_sparse_dump.jsonl"
    )


def sparse_mla_reference_topk_chunk_size() -> int:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_TOPK_CHUNK_ENV)
    if raw_value is None:
        return 512
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 512


def sparse_mla_reference_query_chunk_size() -> int:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_QUERY_CHUNK_ENV)
    if raw_value is None:
        return 256
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 256


def sparse_mla_reference_head_block_size() -> int | None:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_HEAD_BLOCK_ENV)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    if value in (1, 2, 4):
        return value
    return None


def sparse_mla_matmul_decode_enabled() -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_MATMUL_DECODE_ENV)
    if configured is not None:
        return configured
    return current_platform.is_device_capability_family(120)


def sparse_mla_matmul_decode_max_candidates() -> int:
    """Candidate-count budget for the matmul decode path.

    C128A layers carry cdiv(max_model_len, 128) candidates, e.g. 2048 at a
    256k max-model-len. Gating the matmul path on the reference path's
    topk_chunk_size (512) silently dropped those layers onto the
    latency-bound chunked accumulate kernel (~97 us/call, ~9 ms/step at
    256k max-len). The dense gather+GEMM handles thousands of candidates in
    tens of microseconds; 8192 covers the 1M-token design point.
    """
    raw_value = os.getenv(_TRITON_MLA_SPARSE_MATMUL_DECODE_MAX_CANDIDATES_ENV)
    if raw_value is None:
        return 8192
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 8192
