# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fla.ops.utils import input_guard


def test_input_guard_skips_device_index_while_compiling(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def fail_device_index(_):
        raise AssertionError("device_index should not be used while compiling")

    monkeypatch.setattr(torch.accelerator, "device_index", fail_device_index)

    @input_guard
    def guarded_fn(tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.is_contiguous()
        return tensor + 1

    tensor = torch.ones((2, 2)).t()

    result = guarded_fn(tensor)

    torch.testing.assert_close(result, tensor.contiguous() + 1)
