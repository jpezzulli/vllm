# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loader for the nvfp4_ds_mla KV cache write kernel.

The kernel lives in csrc/nvfp4_ds_mla/ and is built as a standalone torch
extension on first use (sm_120a only). Proper vllm._C integration is a
follow-up; this keeps the initial change self-contained.

Set VLLM_NVFP4_DS_MLA_EXT_DIR to point at a prebuilt extension directory
(e.g. baked into a docker image) to skip the JIT build.
"""
import functools
import glob
import importlib.util
import os
import pathlib

import torch  # noqa: F401  (loads libc10/libtorch before the .so)


@functools.cache
def _ext():
    ext_dir = os.environ.get("VLLM_NVFP4_DS_MLA_EXT_DIR", "")
    if ext_dir:
        so = sorted(glob.glob(os.path.join(ext_dir, "**/*.so"), recursive=True))
        if len(so) == 1:
            spec = importlib.util.spec_from_file_location(
                "nvfp4_ds_mla_cache_ext", so[0])
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        if len(so) > 1:
            raise RuntimeError(
                f"ambiguous nvfp4_ds_mla extension directory {ext_dir}: "
                f"found {len(so)} shared objects")

    from torch.utils.cpp_extension import load

    src = (pathlib.Path(__file__).resolve().parents[5] / "csrc" /
           "nvfp4_ds_mla" / "concat_and_cache_nvfp4_ds_mla.cu")
    return load(
        name="nvfp4_ds_mla_cache_ext",
        sources=[str(src)],
        extra_cuda_cflags=[
            "-O3", "--generate-code=arch=compute_120a,code=sm_120a"
        ],
    )


def concat_and_cache_nvfp4_ds_mla(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if not kv_c_normed.is_cuda:
        raise RuntimeError("nvfp4_ds_mla cache input must be CUDA")
    cap = torch.cuda.get_device_capability(kv_c_normed.device)
    if cap != (12, 0):
        raise RuntimeError(
            "nvfp4_ds_mla cache extension requires exactly SM120 on the "
            f"input tensor device; got {cap} on {kv_c_normed.device}")
    _ext().concat_and_cache_nvfp4_ds_mla(kv_c_normed, k_pe, kv_cache,
                                         slot_mapping)
