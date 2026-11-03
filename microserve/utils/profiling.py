from contextlib import contextmanager, nullcontext
from time import perf_counter_ns

import torch

from microserve.utils.metrics import MetricsCollector


_COLLECTOR: MetricsCollector | None = None
_RANGES_ENABLED = False
_LAYERS_ENABLED = False


def configure_profiling(collector: MetricsCollector | None,
                        ranges_enabled: bool = False,
                        layers_enabled: bool = False):
    global _COLLECTOR, _RANGES_ENABLED, _LAYERS_ENABLED
    _COLLECTOR = collector
    _RANGES_ENABLED = ranges_enabled
    _LAYERS_ENABLED = layers_enabled


@contextmanager
def profile_range(name: str, detailed: bool = False,
                  measure_cuda: bool = False):
    ranges = _RANGES_ENABLED and (not detailed or _LAYERS_ENABLED)
    measure = (measure_cuda and _COLLECTOR is not None
               and _COLLECTOR.cuda_events and _COLLECTOR.current is not None)
    start = end = None
    if measure:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    context = torch.profiler.record_function(name) if ranges else nullcontext()
    if ranges:
        torch.cuda.nvtx.range_push(name)
    try:
        with context:
            yield
    finally:
        if ranges:
            torch.cuda.nvtx.range_pop()
        if measure:
            end.record()
            _COLLECTOR.add_cuda_event(name, start, end)


@contextmanager
def cpu_profile(name: str, detailed: bool = False):
    collect = _COLLECTOR is not None and _COLLECTOR.current is not None
    started = perf_counter_ns() if collect else 0
    with profile_range(name, detailed=detailed):
        yield
    if collect:
        _COLLECTOR.add_cpu(name, (perf_counter_ns() - started) / 1_000_000)


def add_metric_value(name: str, value):
    if _COLLECTOR is not None:
        _COLLECTOR.add_value(name, value)
