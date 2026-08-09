# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded direct loading from the existing MoET planes cache."""

import contextlib
import gc
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class DirectPlaneBatchLoader:
    """Read completed cached planes directly, without reconstruction."""

    def __init__(self, workers: int, batch_layers: int):
        if workers < 1:
            raise ValueError("fast-load workers must be at least 1")
        if batch_layers < 1:
            raise ValueError("fast-load batch size must be at least 1")
        self.workers = workers
        self.batch_layers = batch_layers
        self._layers: dict[int, tuple[dict[str, str], dict[str, int]]] = {}
        self._results: dict[int, dict[str, torch.Tensor]] = {}
        self._consumed: set[int] = set()
        self.batch_events: list[dict] = []

    def add_layer(
        self,
        key: int,
        files: dict[str, str],
        sizes: dict[str, int],
    ) -> None:
        contract = (dict(files), dict(sizes))
        if key in self._layers and self._layers[key] != contract:
            raise RuntimeError(f"fast-load layer {key} cache contract changed")
        self._layers[key] = contract

    @staticmethod
    def _read_layer(
        key: int,
        files: dict[str, str],
        sizes: dict[str, int],
    ) -> tuple[int, dict[str, torch.Tensor]]:
        out = {}
        for part, expected in sizes.items():
            path = files[part]
            raw = np.fromfile(path, dtype=np.uint8)
            if raw.nbytes != expected:
                raise RuntimeError(
                    f"fast-load cache changed while reading layer {key} "
                    f"part {part}: {raw.nbytes} != {expected}")
            out[part] = torch.from_numpy(raw)
        return key, out

    def _load_batch(self, requested: int) -> None:
        pending = sorted(
            key for key in self._layers
            if key not in self._consumed and key not in self._results
        )
        if requested not in pending:
            raise RuntimeError(f"fast-load layer {requested} was not planned")
        start = pending.index(requested)
        keys = pending[start:start + self.batch_layers]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="moe-w2-plane-load",
        ) as pool:
            futures = [
                pool.submit(self._read_layer, key, *self._layers[key])
                for key in keys
            ]
            for future in futures:
                key, result = future.result()
                self._results[key] = result
        elapsed = time.perf_counter() - t0
        read_bytes = sum(
            sum(self._layers[key][1].values()) for key in keys)
        event = {
            "layers": keys,
            "workers": self.workers,
            "batch_limit": self.batch_layers,
            "cache_bytes": read_bytes,
            "seconds": elapsed,
        }
        self.batch_events.append(event)
        logger.info(
            "moe_w2 FAST-LOAD direct batch: layers=%s workers=%d "
            "bytes=%.3f GiB read in %.3f s (no CPU projection or "
            "reconstruction)",
            keys,
            self.workers,
            read_bytes / 2**30,
            elapsed,
        )

    def take(self, key: int) -> dict[str, torch.Tensor]:
        if key not in self._results:
            self._load_batch(key)
        return self._results.pop(key)

    def release_consumed(self, key: int) -> None:
        self._consumed.add(key)
        if not self._results:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if len(self._consumed) == len(self._layers):
            elapsed = sum(event["seconds"] for event in self.batch_events)
            read_bytes = sum(
                event["cache_bytes"] for event in self.batch_events)
            logger.info(
                "moe_w2 FAST-LOAD direct complete: %d layers, %.3f GiB, "
                "%d batches, %.3f s cache reads; zero CPU W2 projection "
                "or reconstruction",
                len(self._consumed),
                read_bytes / 2**30,
                len(self.batch_events),
                elapsed,
            )

    def close(self) -> None:
        self._results.clear()
        self._layers.clear()
        with contextlib.suppress(Exception):
            gc.collect()
