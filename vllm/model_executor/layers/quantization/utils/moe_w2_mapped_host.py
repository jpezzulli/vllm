"""Canonical CUDA-mapped host backing for selected Runner V2 W2 layers.

This module implements a deliberately narrow placement policy.  Selected
MXFP4 W2 layers allocate their four canonical 2-bit tensors directly with
``cudaHostAllocMapped`` on the NUMA node local to the active GPU.  The normal
Runner V2 cubins consume the resulting stable UVA pointers; there is no
second complete GPU W2 representation, staging cache, or cudaMalloc overflow
fallback.
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import functools
import json
import math
import os
import platform
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PART_NAMES = ("planes13", "sc13", "planes2", "sc2")

_SYS_SET_MEMPOLICY = 238
_SYS_GET_MEMPOLICY = 239
_SYS_MOVE_PAGES = 279
_MPOL_BIND = 2
_MPOL_DEFAULT = 0
_CUDA_HOST_ALLOC_PORTABLE = 0x01
_CUDA_HOST_ALLOC_MAPPED = 0x02
_PCI_RE = re.compile(
    r"^(?P<domain>[0-9a-f]{4}):(?P<bus>[0-9a-f]{2}):"
    r"(?P<device>[0-9a-f]{2})\.(?P<function>[0-7])$"
)

_LOCK = threading.RLock()
_ALLOCATIONS: dict[int, MappedAllocation] = {}
_LAYER_RECORDS: dict[int, dict] = {}
_BUILDING: set[int] = set()
_DISPATCH_RECORDED: set[int] = set()


@dataclass(frozen=True)
class MappedHostConfig:
    """Resolved process-local mapped-W2 configuration."""

    layers: frozenset[int]
    requested_numa_node: int | None
    requested_pci_bdf: str | None
    audit_path: Path | None

    @property
    def enabled(self) -> bool:
        return bool(self.layers)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> MappedHostConfig:
        raw_layers = environ.get("VLLM_MOE_W2_MAPPED_LAYERS", "").strip()
        layers: set[int] = set()
        if raw_layers:
            for raw in raw_layers.split(","):
                value = raw.strip()
                if not value:
                    raise ValueError("VLLM_MOE_W2_MAPPED_LAYERS contains an empty item")
                try:
                    layer = int(value)
                except ValueError as exc:
                    raise ValueError(
                        "VLLM_MOE_W2_MAPPED_LAYERS must be a comma-separated "
                        f"list of non-negative integers; got {raw_layers!r}"
                    ) from exc
                if layer < 0:
                    raise ValueError(
                        "VLLM_MOE_W2_MAPPED_LAYERS cannot contain negative "
                        f"layer key {layer}"
                    )
                if layer in layers:
                    raise ValueError(
                        "VLLM_MOE_W2_MAPPED_LAYERS contains duplicate layer "
                        f"key {layer}"
                    )
                layers.add(layer)

        raw_node = environ.get("VLLM_MOE_W2_MAPPED_NUMA_NODE", "").strip()
        try:
            requested_node = int(raw_node) if raw_node else None
        except ValueError as exc:
            raise ValueError(
                "VLLM_MOE_W2_MAPPED_NUMA_NODE must be a non-negative integer"
            ) from exc
        if requested_node is not None and requested_node < 0:
            raise ValueError("VLLM_MOE_W2_MAPPED_NUMA_NODE must be non-negative")

        raw_pci = environ.get("VLLM_MOE_W2_MAPPED_PCI", "").strip().lower()
        requested_pci = _normalize_pci_bdf(raw_pci) if raw_pci else None
        raw_audit = environ.get("VLLM_MOE_W2_MAPPED_AUDIT_PATH", "").strip()

        config = cls(
            layers=frozenset(layers),
            requested_numa_node=requested_node,
            requested_pci_bdf=requested_pci,
            audit_path=Path(raw_audit) if raw_audit else None,
        )
        if config.enabled:
            if environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "1":
                raise ValueError("mapped W2 layers require VLLM_USE_V2_MODEL_RUNNER=1")
            if environ.get("VLLM_MOE_W2_BASE_CACHE_GB", "").strip():
                raise ValueError(
                    "mapped W2 layers cannot be combined with "
                    "VLLM_MOE_W2_BASE_CACHE_GB; mapped layers are already "
                    "the canonical host-backed representation"
                )
        return config


def _normalize_pci_bdf(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value):
        value = f"0000:{value}"
    if _PCI_RE.fullmatch(value) is None:
        raise ValueError(
            "VLLM_MOE_W2_MAPPED_PCI must be a PCI BDF such as "
            f"0000:31:00.0; got {value!r}"
        )
    return value


@functools.lru_cache(maxsize=1)
def config() -> MappedHostConfig:
    return MappedHostConfig.from_environ(os.environ)


def enabled() -> bool:
    return config().enabled


def configured_layers() -> frozenset[int]:
    return config().layers


def layer_enabled(layer_key: int) -> bool:
    return layer_key in config().layers


def require_supported_builder(layer_key: int, builder: str) -> None:
    """Reject a selected layer before an unsupported builder can duplicate it."""
    if layer_enabled(layer_key) and builder != "mxfp4":
        raise RuntimeError(
            f"mapped W2 layer {layer_key} requires the MXFP4 W2 builder; "
            f"the active builder is {builder!r}. Refusing to allocate a "
            "redundant complete GPU W2 representation"
        )


def cleanup_on_failure(builder):
    """Release a selected layer if its normal construction path raises."""

    @functools.wraps(builder)
    def wrapped(layer, layer_key: int, *args, **kwargs):
        try:
            return builder(layer, layer_key, *args, **kwargs)
        except BaseException:
            if layer_enabled(layer_key):
                abort_layer(layer_key)
            raise

    return wrapped


def _parse_cpu_list(raw: str) -> set[int]:
    result: set[int] = set()
    for item in raw.strip().split(","):
        if not item:
            continue
        if "-" in item:
            lo, hi = (int(value) for value in item.split("-", 1))
            if lo > hi:
                raise ValueError(f"invalid CPU range {item!r}")
            result.update(range(lo, hi + 1))
        else:
            result.add(int(item))
    return result


def _active_gpu_pci_bdf() -> str:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (
        f"{int(props.pci_domain_id):04x}:{int(props.pci_bus_id):02x}:"
        f"{int(props.pci_device_id):02x}.0"
    )


@dataclass(frozen=True)
class Topology:
    pci_bdf: str
    numa_node: int
    local_cpus: frozenset[int]


def _resolve_topology(cfg: MappedHostConfig) -> Topology:
    if platform.machine() != "x86_64":
        raise RuntimeError(
            "mapped W2 currently supports Linux x86_64 NUMA syscalls only"
        )
    actual_pci = _active_gpu_pci_bdf()
    if cfg.requested_pci_bdf is not None and cfg.requested_pci_bdf != actual_pci:
        raise RuntimeError(
            f"active CUDA device is {actual_pci}, but "
            "VLLM_MOE_W2_MAPPED_PCI requested "
            f"{cfg.requested_pci_bdf}"
        )
    pci_path = Path("/sys/bus/pci/devices") / actual_pci
    try:
        actual_node = int((pci_path / "numa_node").read_text().strip())
        local_cpus = _parse_cpu_list((pci_path / "local_cpulist").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot resolve NUMA locality for active GPU {actual_pci} "
            f"from {pci_path}: {exc}"
        ) from exc
    if actual_node < 0:
        raise RuntimeError(f"active GPU {actual_pci} has no NUMA node in sysfs")
    if cfg.requested_numa_node is not None and cfg.requested_numa_node != actual_node:
        raise RuntimeError(
            f"active GPU {actual_pci} is NUMA node {actual_node}, but "
            "VLLM_MOE_W2_MAPPED_NUMA_NODE requested "
            f"{cfg.requested_numa_node}"
        )
    if not local_cpus:
        raise RuntimeError(f"active GPU {actual_pci} has an empty local CPU list")
    return Topology(actual_pci, actual_node, frozenset(local_cpus))


def _load_cudart():
    candidates = [
        os.path.join(
            os.getenv("CUDA_HOME", "/usr/local/cuda"), "lib64", "libcudart.so"
        ),
        ctypes.util.find_library("cudart"),
        "libcudart.so",
    ]
    last_error: OSError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
            lib.cudaHostAlloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            lib.cudaHostAlloc.restype = ctypes.c_int
            lib.cudaHostGetDevicePointer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_uint,
            ]
            lib.cudaHostGetDevicePointer.restype = ctypes.c_int
            lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
            lib.cudaFreeHost.restype = ctypes.c_int
            lib.cudaDeviceSynchronize.argtypes = []
            lib.cudaDeviceSynchronize.restype = ctypes.c_int
            lib.cudaGetErrorString.argtypes = [ctypes.c_int]
            lib.cudaGetErrorString.restype = ctypes.c_char_p
            return lib
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"cannot load the installed CUDA runtime: {last_error}")


@functools.lru_cache(maxsize=1)
def _cudart():
    return _load_cudart()


_LIBC = ctypes.CDLL(None, use_errno=True)


def _cuda_check(code: int, operation: str) -> None:
    if code == 0:
        return
    detail = _cudart().cudaGetErrorString(code)
    text = detail.decode() if detail else f"CUDA status {code}"
    raise RuntimeError(f"{operation}: {text}")


def _get_policy() -> tuple[int, int]:
    mode = ctypes.c_int()
    mask = ctypes.c_ulong()
    rc = _LIBC.syscall(
        _SYS_GET_MEMPOLICY,
        ctypes.byref(mode),
        ctypes.byref(mask),
        ctypes.sizeof(mask) * 8,
        None,
        0,
    )
    if rc != 0:
        value = ctypes.get_errno()
        raise OSError(value, f"get_mempolicy: {os.strerror(value)}")
    return mode.value, mask.value


def _set_policy(mode: int, mask_value: int) -> None:
    mask = ctypes.c_ulong(mask_value)
    mask_pointer = None if mode == _MPOL_DEFAULT else ctypes.byref(mask)
    rc = _LIBC.syscall(
        _SYS_SET_MEMPOLICY,
        mode,
        mask_pointer,
        ctypes.sizeof(mask) * 8,
    )
    if rc != 0:
        value = ctypes.get_errno()
        raise OSError(value, f"set_mempolicy: {os.strerror(value)}")


class _LocalPolicy:
    def __init__(self, topology: Topology):
        self.topology = topology

    def __enter__(self) -> int:
        allowed = set(os.sched_getaffinity(0))
        target = set(self.topology.local_cpus) & allowed
        if not target:
            raise RuntimeError(
                f"no permitted CPU is local to GPU {self.topology.pci_bdf}: "
                f"local={sorted(self.topology.local_cpus)}, "
                f"allowed={sorted(allowed)}"
            )
        self._old_affinity = allowed
        self._old_policy = _get_policy()
        try:
            os.sched_setaffinity(0, target)
            _set_policy(_MPOL_BIND, 1 << self.topology.numa_node)
        except BaseException:
            os.sched_setaffinity(0, self._old_affinity)
            raise
        return min(target)

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            _set_policy(*self._old_policy)
        finally:
            os.sched_setaffinity(0, self._old_affinity)


def _page_nodes(address: int, nbytes: int) -> dict[int, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = (nbytes + page_size - 1) // page_size
    counts: dict[int, int] = {}
    chunk = 65536
    for base in range(0, page_count, chunk):
        count = min(chunk, page_count - base)
        addresses = (ctypes.c_void_p * count)(
            *(address + (base + index) * page_size for index in range(count))
        )
        status = (ctypes.c_int * count)()
        rc = _LIBC.syscall(
            _SYS_MOVE_PAGES,
            0,
            count,
            addresses,
            None,
            status,
            0,
        )
        if rc < 0:
            value = ctypes.get_errno()
            raise OSError(value, f"move_pages: {os.strerror(value)}")
        for value in status:
            if value < 0:
                raise OSError(-value, f"move_pages page status: {os.strerror(-value)}")
            counts[value] = counts.get(value, 0) + 1
    return counts


def _numa_maps_line(address: int) -> str:
    best_address = -1
    best_line = ""
    for line in Path("/proc/self/numa_maps").read_text().splitlines():
        try:
            start = int(line.split(maxsplit=1)[0], 16)
        except (IndexError, ValueError):
            continue
        if start <= address and start >= best_address:
            best_address = start
            best_line = line
    return best_line


@dataclass(frozen=True)
class Layout:
    offsets: dict[str, int]
    sizes: dict[str, int]
    total_bytes: int


def plan_layout(shapes: Mapping[str, Sequence[int]]) -> Layout:
    if set(shapes) != set(PART_NAMES):
        raise ValueError(
            "mapped W2 requires exactly the four canonical tensors; got "
            f"{sorted(shapes)}"
        )
    offsets: dict[str, int] = {}
    sizes: dict[str, int] = {}
    cursor = 0
    for name in PART_NAMES:
        shape = tuple(shapes[name])
        if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(
                f"mapped W2 {name} shape must contain positive integers; got {shape}"
            )
        size = math.prod(shape)
        offsets[name] = cursor
        sizes[name] = size
        cursor += size
    return Layout(offsets, sizes, cursor)


class MappedAllocation:
    """Own one page-locked mapped allocation and its canonical CPU tensor."""

    def __init__(self, layer_key: int, nbytes: int, topology: Topology):
        self.layer_key = layer_key
        self.nbytes = nbytes
        self.topology = topology
        self.host_pointer = 0
        self.device_pointer = 0
        self.tensor: torch.Tensor | None = None
        self._buffer = None
        pointer = ctypes.c_void_p()
        try:
            with _LocalPolicy(topology) as first_cpu:
                _cuda_check(
                    _cudart().cudaHostAlloc(
                        ctypes.byref(pointer),
                        nbytes,
                        _CUDA_HOST_ALLOC_PORTABLE | _CUDA_HOST_ALLOC_MAPPED,
                    ),
                    f"cudaHostAllocMapped layer {layer_key} "
                    f"({nbytes / 2**30:.6f} GiB on NUMA node "
                    f"{topology.numa_node})",
                )
            self.host_pointer = int(pointer.value or 0)
            device_pointer = ctypes.c_void_p()
            _cuda_check(
                _cudart().cudaHostGetDevicePointer(
                    ctypes.byref(device_pointer), pointer, 0
                ),
                f"cudaHostGetDevicePointer layer {layer_key}",
            )
            self.device_pointer = int(device_pointer.value or 0)
            if self.device_pointer != self.host_pointer:
                raise RuntimeError(
                    f"UVA pointer mismatch for mapped W2 layer {layer_key}: "
                    f"host=0x{self.host_pointer:x}, "
                    f"device=0x{self.device_pointer:x}"
                )
            array_type = ctypes.c_ubyte * nbytes
            self._buffer = array_type.from_address(self.host_pointer)
            self.tensor = torch.frombuffer(
                self._buffer, dtype=torch.uint8, count=nbytes
            )
            self.first_cpu = first_cpu
            self.page_nodes = _page_nodes(self.host_pointer, nbytes)
            self.numa_maps_line = _numa_maps_line(self.host_pointer)
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = (nbytes + page_size - 1) // page_size
            if self.page_nodes != {topology.numa_node: page_count}:
                raise RuntimeError(
                    f"mapped W2 layer {layer_key} is not fully local to "
                    f"NUMA node {topology.numa_node}: {self.page_nodes}; "
                    f"expected {page_count} pages"
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        pointer, self.host_pointer = self.host_pointer, 0
        self.device_pointer = 0
        self.tensor = None
        self._buffer = None
        if pointer:
            _cuda_check(
                _cudart().cudaFreeHost(ctypes.c_void_p(pointer)),
                f"cudaFreeHost mapped W2 layer {self.layer_key}",
            )


def _sample_values(tensor: torch.Tensor) -> list[int]:
    flat = tensor.reshape(-1)
    indices = sorted(
        index
        for index in set(
            (0, 1, flat.numel() // 2, max(flat.numel() - 2, 0), flat.numel() - 1)
        )
        if 0 <= index < flat.numel()
    )
    return [int(flat[index].item()) for index in indices]


def allocate_canonical_layer(
    layer_key: int,
    shapes: Mapping[str, Sequence[int]],
) -> dict[str, torch.Tensor] | None:
    """Allocate a selected layer's only complete W2 representation."""
    if not layer_enabled(layer_key):
        return None
    require_supported_builder(layer_key, "mxfp4")
    layout = plan_layout(shapes)
    with _LOCK:
        if layer_key in _ALLOCATIONS or layer_key in _BUILDING:
            raise RuntimeError(f"mapped W2 layer {layer_key} allocated twice")
        _BUILDING.add(layer_key)
    allocation: MappedAllocation | None = None
    try:
        topology = _resolve_topology(config())
        allocation = MappedAllocation(layer_key, layout.total_bytes, topology)
        assert allocation.tensor is not None
        views = {
            name: allocation.tensor.narrow(
                0, layout.offsets[name], layout.sizes[name]
            ).view(tuple(shapes[name]))
            for name in PART_NAMES
        }
        record = {
            "layer_key": layer_key,
            "allocation_phase": "canonical construction",
            "allocation_bytes": layout.total_bytes,
            "allocation_gib": layout.total_bytes / 2**30,
            "host_pointer": f"0x{allocation.host_pointer:x}",
            "device_pointer": f"0x{allocation.device_pointer:x}",
            "uva_pointer_equal": True,
            "first_cpu": allocation.first_cpu,
            "pci_bdf": topology.pci_bdf,
            "numa_node": topology.numa_node,
            "page_nodes": allocation.page_nodes,
            "numa_maps_line": allocation.numa_maps_line,
            "parts": {
                name: {
                    "offset": layout.offsets[name],
                    "bytes": layout.sizes[name],
                    "shape": list(shapes[name]),
                }
                for name in PART_NAMES
            },
            "construction_complete": False,
            "redundant_gpu_w2_bytes": None,
            "kernel_dispatch": None,
        }
        with _LOCK:
            _ALLOCATIONS[layer_key] = allocation
            _LAYER_RECORDS[layer_key] = record
            _BUILDING.remove(layer_key)
            _write_audit_locked()
        logger.info(
            "moe_w2 mapped-host canonical layer %d: %.6f GiB, "
            "UVA=0x%x, PCI=%s, NUMA=%d pages=%s",
            layer_key,
            layout.total_bytes / 2**30,
            allocation.device_pointer,
            topology.pci_bdf,
            topology.numa_node,
            allocation.page_nodes,
        )
        return views
    except BaseException:
        with _LOCK:
            _BUILDING.discard(layer_key)
        if allocation is not None:
            allocation.close()
        raise


def record_constructed(
    layer_key: int,
    tensors: Mapping[str, torch.Tensor],
    cuda_free_bytes: int,
) -> None:
    """Validate the installed representation after quantization synchronizes."""
    if not layer_enabled(layer_key):
        return
    if set(tensors) != set(PART_NAMES):
        raise ValueError(f"mapped W2 layer {layer_key} has incomplete tensors")
    with _LOCK:
        allocation = _ALLOCATIONS.get(layer_key)
        record = _LAYER_RECORDS.get(layer_key)
        if allocation is None or record is None:
            raise RuntimeError(f"mapped W2 layer {layer_key} was not allocated")
        for name in PART_NAMES:
            tensor = tensors[name]
            if tensor.device.type != "cpu":
                raise RuntimeError(
                    f"mapped W2 layer {layer_key} retained canonical part "
                    f"{name} on {tensor.device}; refusing a redundant GPU W2 "
                    "representation"
                )
            start = tensor.data_ptr()
            end = start + tensor.nbytes
            allocation_end = allocation.host_pointer + allocation.nbytes
            if start < allocation.host_pointer or end > allocation_end:
                raise RuntimeError(
                    f"mapped W2 layer {layer_key} part {name} is outside "
                    "its canonical mapped allocation"
                )
            record["parts"][name]["sample_values"] = _sample_values(tensor)
        record["installed_devices"] = {
            name: str(tensors[name].device) for name in PART_NAMES
        }
        record["installed_data_ptrs"] = {
            name: f"0x{tensors[name].data_ptr():x}" for name in PART_NAMES
        }
        record["cuda_free_after_construction"] = cuda_free_bytes
        record["construction_complete"] = True
        record["redundant_gpu_w2_bytes"] = 0
        _write_audit_locked()


def abort_layer(layer_key: int) -> None:
    """Release a partial mapped layer after a construction failure."""
    with _LOCK:
        allocation = _ALLOCATIONS.pop(layer_key, None)
        _LAYER_RECORDS.pop(layer_key, None)
        _BUILDING.discard(layer_key)
        _write_audit_locked()
    if allocation is not None:
        allocation.close()


def note_real_kernel_dispatch(
    layer_key: int,
    tokens: int,
    prefill: bool,
    capturing: bool,
) -> None:
    if not layer_enabled(layer_key):
        return
    with _LOCK:
        if layer_key in _DISPATCH_RECORDED:
            return
        record = _LAYER_RECORDS.get(layer_key)
        if record is None or not record.get("construction_complete"):
            raise RuntimeError(
                f"mapped W2 layer {layer_key} dispatched before construction"
            )
        _DISPATCH_RECORDED.add(layer_key)
        record["kernel_dispatch"] = {
            "tokens": tokens,
            "prefill": prefill,
            "capturing": capturing,
            "production_w2_path": True,
        }
        _write_audit_locked()
    logger.info(
        "moe_w2 mapped-host production cubin dispatch: layer=%d tokens=%d "
        "prefill=%s capturing=%s",
        layer_key,
        tokens,
        prefill,
        capturing,
    )


def accounting() -> dict:
    with _LOCK:
        return {
            "configured_layers": sorted(config().layers),
            "allocated_layers": sorted(_ALLOCATIONS),
            "total_allocation_bytes": sum(
                allocation.nbytes for allocation in _ALLOCATIONS.values()
            ),
            "redundant_gpu_w2_bytes": sum(
                record.get("redundant_gpu_w2_bytes") or 0
                for record in _LAYER_RECORDS.values()
            ),
        }


def _write_audit_locked() -> None:
    target = config().audit_path
    if target is None:
        return
    payload = {
        "mechanism": "canonical cudaHostAllocMapped production-kernel pointers",
        "configured_layers": sorted(config().layers),
        "requested_pci_bdf": config().requested_pci_bdf,
        "requested_numa_node": config().requested_numa_node,
        "total_allocation_bytes": sum(
            allocation.nbytes for allocation in _ALLOCATIONS.values()
        ),
        "redundant_gpu_w2_bytes": sum(
            record.get("redundant_gpu_w2_bytes") or 0
            for record in _LAYER_RECORDS.values()
        ),
        "layers": [_LAYER_RECORDS[key] for key in sorted(_LAYER_RECORDS)],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def shutdown() -> None:
    """Synchronize once, release all mapped allocations, and reset state."""
    with _LOCK:
        allocations = list(_ALLOCATIONS.values())
        _ALLOCATIONS.clear()
        _LAYER_RECORDS.clear()
        _BUILDING.clear()
        _DISPATCH_RECORDED.clear()
    errors: list[str] = []
    if allocations:
        try:
            _cuda_check(
                _cudart().cudaDeviceSynchronize(),
                "cudaDeviceSynchronize before mapped W2 cleanup",
            )
        except Exception as exc:  # cleanup every allocation before raising
            errors.append(str(exc))
    for allocation in reversed(allocations):
        try:
            allocation.close()
        except Exception as exc:  # cleanup remaining layers before raising
            errors.append(str(exc))
    if errors:
        raise RuntimeError("mapped W2 cleanup was incomplete: " + "; ".join(errors))


def _shutdown_at_exit() -> None:
    try:
        shutdown()
    except Exception:
        logger.exception("mapped W2 process-exit cleanup failed")


atexit.register(_shutdown_at_exit)


def _reset_for_tests() -> None:
    """Reset cached configuration after tests have released allocations."""
    if _ALLOCATIONS:
        raise RuntimeError("tests must release mapped W2 allocations first")
    config.cache_clear()
    _cudart.cache_clear()
