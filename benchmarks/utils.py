"""Shared timing and reporting utilities for the lgatr benchmark suite."""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from functools import lru_cache
from pathlib import Path

# Make ``import lgatr`` work when this script is launched via
# ``python benchmarks/foo.py`` (sys.path[0] is the script's dir, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from lgatr.primitives.config import PrimitivesConfig


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_fwd_bwd(forward_fn, grad_tensors, *, device, warmup=25, iters=100):
    """Return trimmed-mean (drop top/bottom 20%) wall times of forward and backward in seconds,
    plus the peak memory captured during the timed iters in MB.

    On CUDA, timings come from ``torch.cuda.Event.elapsed_time`` to avoid the per-iter
    ``cudaDeviceSynchronize`` overhead (a few µs each, comparable to the kernels we measure).
    On CPU, ``time.perf_counter`` is used directly.

    Resets peak-memory stats *after* warmup so triton's autotune workspace (a ~256 MB L2-flush
    buffer allocated by ``triton.testing.do_bench`` during the first call) does not contaminate
    the reported peak. Grads are zeroed in place between iters rather than nulled, so backward
    does not re-allocate ``t.grad`` every iteration.

    Returns ``(None, None, None)`` if any iteration triggers a CUDA OOM — useful when sweeping
    batch sizes near the device's memory limit.
    """
    grad_tensors = list(grad_tensors)

    try:
        for _ in range(warmup):
            out = forward_fn()
            out.mean().backward()
            _zero_grads(grad_tensors)
        sync(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None

    reset_peak_mem(device)

    try:
        if device.type == "cuda":
            fwd, bwd = _time_iters_cuda(forward_fn, grad_tensors, iters, device)
        else:
            fwd, bwd = _time_iters_cpu(forward_fn, grad_tensors, iters)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None

    return _trim_mean(fwd), _trim_mean(bwd), peak_mem_mb(device)


def _time_iters_cpu(forward_fn, grad_tensors, iters):
    fwd, bwd = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = forward_fn()
        t1 = time.perf_counter()
        out.mean().backward()
        t2 = time.perf_counter()
        fwd.append(t1 - t0)
        bwd.append(t2 - t1)
        _zero_grads(grad_tensors)
    return fwd, bwd


def _time_iters_cuda(forward_fn, grad_tensors, iters, device):
    # One event per timing point per iter; cudaEventRecord is ~1 µs and avoids host-side syncs
    # in the timed window. We sync once at the end and read elapsed_time (returns ms).
    fwd, bwd = [], []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = forward_fn()
        e1.record()
        out.mean().backward()
        e2.record()
        torch.cuda.synchronize(device)
        fwd.append(e0.elapsed_time(e1) * 1e-3)
        bwd.append(e1.elapsed_time(e2) * 1e-3)
        _zero_grads(grad_tensors)
    return fwd, bwd


def _zero_grads(grad_tensors):
    # Zero in place to keep the grad buffer allocated; otherwise every backward re-allocs.
    for t in grad_tensors:
        if t.grad is not None:
            t.grad.zero_()


def _trim_mean(values, frac=0.2):
    values = sorted(values)
    lo = int(frac * len(values))
    hi = len(values) - lo
    return sum(values[lo:hi]) / (hi - lo)


@lru_cache
def config_for(mode: str) -> PrimitivesConfig:
    """Cached :class:`PrimitivesConfig` per dispatch path (``dense`` / ``sparse`` / ``triton``).

    The cache returns the same instance on repeated calls so ``torch.compile`` dynamo guards
    don't trigger a recompile each time the bench picks a config.
    """
    return PrimitivesConfig(sparse=mode == "sparse", triton=mode == "triton")


def reset_peak_mem(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_mem_mb(device):
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1e6


def maybe_set_single_thread_cpu(device):
    # Multi-threaded BLAS produces 10-100x noisy timings on the small (16x16) primitive
    # kernels we measure. Single-thread is a clean baseline.
    if device.type == "cpu":
        torch.set_num_threads(1)


def machine_info():
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        info["cpu"] = platform.processor() or "unknown"
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _round_sig(x, sig=4):
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def _round_floats(obj, sig=4):
    if isinstance(obj, float):
        return _round_sig(obj, sig)
    if isinstance(obj, dict):
        return {k: _round_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, sig) for v in obj]
    return obj


def save_json(stem, data):
    """Save data to ``benchmarks/<stem>.json`` after rounding floats to 4 sig figs."""
    path = Path(__file__).parent / f"{stem}.json"
    with open(path, "w") as f:
        json.dump(_round_floats(data), f, indent=2, default=str)
    print(f"\nSaved {path}")
    return path


def _fmt_cell(key, value):
    if value is None:
        return "—"
    if key.endswith("_s") and isinstance(value, float):
        if value < 1e-3:
            return f"{value * 1e6:.1f}us"
        if value < 1.0:
            return f"{value * 1e3:.2f}ms"
        return f"{value:.2f}s"
    if key.endswith("_mb") and isinstance(value, float):
        if value < 1024:
            return f"{value:.1f}MB"
        return f"{value / 1024:.2f}GB"
    if key.endswith("_x") and isinstance(value, float):
        return f"{value:.2f}x"
    return str(value)


def print_table(title, columns, rows):
    """Print rows of raw values as an auto-widthed table; cell formatting is by key suffix."""
    cells = [[_fmt_cell(c, r.get(c)) for c in columns] for r in rows]
    widths = [
        max(len(c), *(len(row[i]) for row in cells)) if cells else len(c)
        for i, c in enumerate(columns)
    ]
    print()
    print(f"=== {title} ===")
    print("  ".join(f"{c:>{w}}" for c, w in zip(columns, widths, strict=True)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(f"{cell:>{w}}" for cell, w in zip(row, widths, strict=True)))


def device_label(device):
    return "cuda" if device.type == "cuda" else "cpu"
