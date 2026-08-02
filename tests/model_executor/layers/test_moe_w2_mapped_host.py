# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import moe_w2_mapped_host


@pytest.fixture(autouse=True)
def reset_mapped_host(monkeypatch):
    monkeypatch.delenv("VLLM_MOE_W2_MAPPED_LAYERS", raising=False)
    monkeypatch.delenv("VLLM_MOE_W2_MAPPED_NUMA_NODE", raising=False)
    monkeypatch.delenv("VLLM_MOE_W2_MAPPED_PCI", raising=False)
    monkeypatch.delenv("VLLM_MOE_W2_MAPPED_AUDIT_PATH", raising=False)
    monkeypatch.delenv("VLLM_MOE_W2_BASE_CACHE_GB", raising=False)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    moe_w2_mapped_host.config.cache_clear()
    yield
    if moe_w2_mapped_host._ALLOCATIONS:
        moe_w2_mapped_host.shutdown()
    assert not moe_w2_mapped_host._ALLOCATIONS
    moe_w2_mapped_host.config.cache_clear()


def _enable(monkeypatch, layers="40,41,42"):
    monkeypatch.setenv("VLLM_MOE_W2_MAPPED_LAYERS", layers)
    moe_w2_mapped_host.config.cache_clear()


def test_config_selects_only_explicit_layer_keys(monkeypatch):
    _enable(monkeypatch, "42,40,41")
    assert moe_w2_mapped_host.configured_layers() == {40, 41, 42}
    assert moe_w2_mapped_host.layer_enabled(40)
    assert not moe_w2_mapped_host.layer_enabled(39)


@pytest.mark.parametrize("layers", ["40,,42", "40,forty", "40,-1", "40,40"])
def test_config_rejects_invalid_layer_selection(monkeypatch, layers):
    _enable(monkeypatch, layers)
    with pytest.raises(ValueError, match="MAPPED_LAYERS"):
        moe_w2_mapped_host.config()


def test_config_rejects_v1_and_base_cache(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    with pytest.raises(ValueError, match="V2_MODEL_RUNNER"):
        moe_w2_mapped_host.config()

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_MOE_W2_BASE_CACHE_GB", "8")
    moe_w2_mapped_host.config.cache_clear()
    with pytest.raises(ValueError, match="BASE_CACHE_GB"):
        moe_w2_mapped_host.config()


def test_config_normalizes_optional_topology_guards(monkeypatch):
    _enable(monkeypatch, "40")
    monkeypatch.setenv("VLLM_MOE_W2_MAPPED_NUMA_NODE", "0")
    monkeypatch.setenv("VLLM_MOE_W2_MAPPED_PCI", "31:00.0")
    cfg = moe_w2_mapped_host.config()
    assert cfg.requested_numa_node == 0
    assert cfg.requested_pci_bdf == "0000:31:00.0"


def test_plan_layout_matches_validated_ds4_layer_size():
    layout = moe_w2_mapped_host.plan_layout(
        {
            "planes13": (256, 4194304),
            "sc13": (256, 524288),
            "planes2": (256, 2097152),
            "sc2": (256, 262144),
        }
    )
    assert layout.total_bytes == 1811939328
    assert layout.total_bytes / 2**30 == 1.6875
    assert layout.offsets == {
        "planes13": 0,
        "sc13": 1073741824,
        "planes2": 1207959552,
        "sc2": 1744830464,
    }


@pytest.mark.parametrize(
    "shapes",
    [
        {"planes13": (1,), "sc13": (1,), "planes2": (1,)},
        {
            "planes13": (1,),
            "sc13": (1,),
            "planes2": (1,),
            "sc2": (0,),
        },
    ],
)
def test_plan_layout_rejects_partial_or_empty_parts(shapes):
    with pytest.raises(ValueError, match="mapped W2"):
        moe_w2_mapped_host.plan_layout(shapes)


def test_unsupported_builder_fails_before_gpu_allocation(monkeypatch):
    _enable(monkeypatch, "40")
    with pytest.raises(RuntimeError, match="redundant complete GPU W2"):
        moe_w2_mapped_host.require_supported_builder(40, "nvfp4")
    moe_w2_mapped_host.require_supported_builder(39, "nvfp4")


class _FakeCudart:
    def __init__(self, pointer_delta=0, alloc_status=0):
        self.pointer_delta = pointer_delta
        self.alloc_status = alloc_status
        self.buffer = None
        self.freed = []
        self.syncs = 0

    def cudaHostAlloc(self, output, nbytes, _flags):
        if self.alloc_status:
            return self.alloc_status
        self.buffer = ctypes.create_string_buffer(nbytes)
        pointer = ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))
        pointer[0] = ctypes.addressof(self.buffer)
        return 0

    def cudaHostGetDevicePointer(self, output, pointer, _flags):
        result = ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))
        result[0] = int(pointer.value) + self.pointer_delta
        return 0

    def cudaFreeHost(self, pointer):
        self.freed.append(int(pointer.value))
        return 0

    def cudaDeviceSynchronize(self):
        self.syncs += 1
        return 0

    @staticmethod
    def cudaGetErrorString(_status):
        return b"injected allocation failure"


def _install_fake_runtime(monkeypatch, runtime):
    topology = moe_w2_mapped_host.Topology("0000:31:00.0", 0, frozenset({0}))
    monkeypatch.setattr(moe_w2_mapped_host, "_resolve_topology", lambda _cfg: topology)
    monkeypatch.setattr(
        moe_w2_mapped_host, "_LocalPolicy", lambda _topology: nullcontext(0)
    )
    monkeypatch.setattr(moe_w2_mapped_host, "_cudart", lambda: runtime)
    monkeypatch.setattr(
        moe_w2_mapped_host, "_page_nodes", lambda _pointer, _size: {0: 1}
    )
    monkeypatch.setattr(moe_w2_mapped_host, "_numa_maps_line", lambda _pointer: "N0=1")


def _small_shapes():
    return {
        "planes13": (1,),
        "sc13": (1,),
        "planes2": (1,),
        "sc2": (1,),
    }


def test_allocation_views_accounting_and_cleanup(monkeypatch):
    _enable(monkeypatch, "40")
    runtime = _FakeCudart()
    _install_fake_runtime(monkeypatch, runtime)
    tensors = moe_w2_mapped_host.allocate_canonical_layer(40, _small_shapes())
    assert tensors is not None
    assert [tensors[name].data_ptr() for name in moe_w2_mapped_host.PART_NAMES] == list(
        range(tensors["planes13"].data_ptr(), tensors["planes13"].data_ptr() + 4)
    )
    moe_w2_mapped_host.record_constructed(40, tensors, 123456)
    assert moe_w2_mapped_host.accounting() == {
        "configured_layers": [40],
        "allocated_layers": [40],
        "total_allocation_bytes": 4,
        "redundant_gpu_w2_bytes": 0,
    }
    del tensors
    moe_w2_mapped_host.shutdown()
    assert runtime.syncs == 1
    assert len(runtime.freed) == 1


def test_duplicate_allocation_is_rejected_and_original_is_released(monkeypatch):
    _enable(monkeypatch, "40")
    runtime = _FakeCudart()
    _install_fake_runtime(monkeypatch, runtime)
    tensors = moe_w2_mapped_host.allocate_canonical_layer(40, _small_shapes())
    with pytest.raises(RuntimeError, match="allocated twice"):
        moe_w2_mapped_host.allocate_canonical_layer(40, _small_shapes())
    del tensors
    moe_w2_mapped_host.shutdown()
    assert len(runtime.freed) == 1


def test_pointer_mismatch_frees_allocation(monkeypatch):
    _enable(monkeypatch, "40")
    runtime = _FakeCudart(pointer_delta=4096)
    _install_fake_runtime(monkeypatch, runtime)
    with pytest.raises(RuntimeError, match="UVA pointer mismatch"):
        moe_w2_mapped_host.allocate_canonical_layer(40, _small_shapes())
    assert len(runtime.freed) == 1
    assert not moe_w2_mapped_host._ALLOCATIONS


def test_allocation_failure_has_actionable_context(monkeypatch):
    _enable(monkeypatch, "40")
    runtime = _FakeCudart(alloc_status=2)
    _install_fake_runtime(monkeypatch, runtime)
    with pytest.raises(RuntimeError, match=r"layer 40 .*NUMA node 0"):
        moe_w2_mapped_host.allocate_canonical_layer(40, _small_shapes())
    assert not runtime.freed
    assert not moe_w2_mapped_host._ALLOCATIONS


def test_record_rejects_tensor_outside_canonical_allocation(monkeypatch):
    _enable(monkeypatch, "40")
    runtime = _FakeCudart()
    _install_fake_runtime(monkeypatch, runtime)
    tensors = moe_w2_mapped_host.allocate_canonical_layer(40, _small_shapes())
    assert tensors is not None
    invalid = dict(tensors)
    invalid["planes13"] = torch.empty(1, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="outside"):
        moe_w2_mapped_host.record_constructed(40, invalid, 0)
    del invalid, tensors
    moe_w2_mapped_host.shutdown()


def test_builder_failure_aborts_partial_layer(monkeypatch):
    _enable(monkeypatch, "40")

    @moe_w2_mapped_host.cleanup_on_failure
    def failing_builder(_layer, _layer_key):
        raise RuntimeError("injected build failure")

    with mock.patch.object(moe_w2_mapped_host, "abort_layer") as abort:
        with pytest.raises(RuntimeError, match="injected build failure"):
            failing_builder(SimpleNamespace(), 40)
        abort.assert_called_once_with(40)
