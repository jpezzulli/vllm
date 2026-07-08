# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated CUDA-event timers for prefill anatomy (VLLM_PREFILL_TIMERS=1).

Usage:  with prefill_timers.span("indexer"):  ...
Pairs of events are accumulated per name; every FLUSH_EVERY completed
spans of a name, one synchronize drains all pending pairs and logs
cumulative milliseconds per name. Zero overhead when the env is off.
Skips recording inside CUDA graph capture.
"""

import os
from contextlib import contextmanager

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.getenv("VLLM_PREFILL_TIMERS", "0") == "1"
FLUSH_EVERY = int(os.getenv("VLLM_PREFILL_TIMERS_FLUSH", "172"))

_pending: dict = {}
_total_ms: dict = {}
_count: dict = {}


@contextmanager
def span(name: str):
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        yield
        return
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    try:
        yield
    finally:
        e1.record()
        _pending.setdefault(name, []).append((e0, e1))
        if len(_pending[name]) >= FLUSH_EVERY:
            _flush(name)


def _flush(name):
    torch.cuda.synchronize()
    pairs = _pending.pop(name, [])
    ms = sum(a.elapsed_time(b) for a, b in pairs)
    _total_ms[name] = _total_ms.get(name, 0.0) + ms
    _count[name] = _count.get(name, 0) + len(pairs)
    logger.info("[prefill-timer] %-14s total %8.1f ms over %5d spans",
                name, _total_ms[name], _count[name])
