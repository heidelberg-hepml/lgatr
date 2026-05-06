"""Speed benchmark for ``equi_linear`` and ``geometric_product`` in isolation.

Compares up to five dispatch options across a batch sweep: ``dense``, ``sparse``, ``triton``
(eager only — kernels are wrapped in ``@torch.compiler.disable``), plus optional ``dense+compile``
and ``sparse+compile``. ``--compile``, ``--triton`` and ``--cpu`` are opt-in.
"""

from __future__ import annotations

import argparse

import torch
import utils

from lgatr.primitives.bilinear import geometric_product
from lgatr.primitives.config import PrimitivesConfig
from lgatr.primitives.linear import equi_linear

DTYPE = torch.float32
WARMUP = 10
# Trim-mean drops top/bottom 20% of samples, so ITERS=15 leaves 9 samples after trimming.
ITERS = 15

IN_C = OUT_C = 32
BATCHES = [64, 1024, 4096]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench",
        choices=["linear", "gp", "both"],
        default="both",
        help="which primitive to bench (default: both)",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="add the dense+compile and sparse+compile legs",
    )
    p.add_argument(
        "--triton",
        action="store_true",
        help="add the triton column (CUDA host with importable triton)",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="force CPU only (skip CUDA even if available)",
    )
    return p.parse_args()


def _have_triton() -> bool:
    """Lazy probe — only imports the triton subpackage on demand to avoid an unconditional
    ``import triton`` on CUDA hosts when the user hasn't asked for it."""
    from lgatr.primitives.triton import _HAVE_TRITON

    return _HAVE_TRITON


def resolve_modes(device, compiled, args):
    """Return dispatch modes to time for one (device, compiled) leg."""
    modes = ["dense", "sparse"]
    if args.triton and device.type == "cuda" and not compiled and _have_triton():
        modes.append("triton")
    return modes


def run_one(fn, inputs, mode, device):
    """Time one (callable, mode) point. ``fn`` is the (possibly compiled) primitive."""
    config = utils.config_for(mode)
    grad = [t for t in inputs if t.requires_grad]
    return utils.time_fwd_bwd(
        lambda: fn(*inputs, config=config),
        grad,
        device=device,
        warmup=WARMUP,
        iters=ITERS,
    )


def build_callables(primitive, compile_modes, modes_per_compile):
    """Return ``{(compiled, mode): callable}`` with one ``torch.compile`` wrapper per mode.

    Pre-building keeps dynamo's per-function cache warm across batches; otherwise each
    ``run_one`` would create a fresh wrapper and re-trace on the first call of every batch.
    """
    fns = {}
    for compiled in compile_modes:
        for mode in modes_per_compile[compiled]:
            if compiled:
                fns[(compiled, mode)] = torch.compile(primitive, fullgraph=True, dynamic=True)
            else:
                fns[(compiled, mode)] = primitive
    return fns


def build_columns(modes):
    """Column order for the printed table — speedups are vs the dense column."""
    cols = ["compiled", "batch"]
    cols += [f"{m}_fwd_s" for m in modes]
    cols += [f"{m}_fwd_x" for m in modes if m != "dense"]
    cols += [f"{m}_bwd_s" for m in modes]
    cols += [f"{m}_bwd_x" for m in modes if m != "dense"]
    cols += [f"{m}_peak_mem_mb" for m in modes]
    return cols


def build_row(compiled, batch, modes, results):
    row = {
        "compiled": compiled,
        "batch": batch,
    }
    d_fwd, d_bwd, _ = results.get("dense", (None, None, None))
    for m in modes:
        fwd, bwd, mem = results[m]
        row[f"{m}_fwd_s"] = fwd
        row[f"{m}_bwd_s"] = bwd
        row[f"{m}_peak_mem_mb"] = mem
        if m != "dense":
            row[f"{m}_fwd_x"] = d_fwd / fwd if (d_fwd and fwd) else None
            row[f"{m}_bwd_x"] = d_bwd / bwd if (d_bwd and bwd) else None
    return row


def check_correctness(primitive, make_inputs, modes, device, rtol=1e-4, atol=1e-5):
    """Sanity check: each non-dense mode matches the dense forward to (rtol, atol)."""
    inputs = make_inputs(BATCHES[0], device)
    with torch.no_grad():
        ref = primitive(*inputs, config=utils.config_for("dense"))
        for m in modes:
            if m == "dense":
                continue
            out = primitive(*inputs, config=utils.config_for(m))
            if not torch.allclose(ref, out, rtol=rtol, atol=atol):
                err = (ref - out).abs().max().item()
                raise RuntimeError(f"{m} forward diverges from dense (max abs err={err:.2e})")


def sweep(primitive, make_inputs, title_prefix, device, args):
    rows = []
    compile_modes = [False, True] if args.compile else [False]
    modes_per_compile = {c: resolve_modes(device, c, args) for c in compile_modes}
    check_correctness(primitive, make_inputs, modes_per_compile[False], device)
    columns = build_columns(modes_per_compile[False])
    fns = build_callables(primitive, compile_modes, modes_per_compile)
    for compiled in compile_modes:
        modes = modes_per_compile[compiled]
        for batch in BATCHES:
            inputs = make_inputs(batch, device)
            results = {m: run_one(fns[(compiled, m)], inputs, m, device) for m in modes}
            rows.append(build_row(compiled, batch, modes, results))
    title = f"{title_prefix}  device={utils.device_label(device)}"
    utils.print_table(title, columns, rows)
    return {"sweep": title, "rows": rows}


def bench_equi_linear(device, args):
    coeffs = torch.randn(
        OUT_C,
        IN_C,
        PrimitivesConfig().num_pin_linear_basis_elements,
        dtype=DTYPE,
        device=device,
        requires_grad=True,
    )

    def make_inputs(batch, device):
        x = torch.randn(batch, IN_C, 16, dtype=DTYPE, device=device, requires_grad=True)
        return (x, coeffs)

    return sweep(
        equi_linear,
        make_inputs,
        f"equi_linear  in_c={IN_C}  out_c={OUT_C}",
        device,
        args,
    )


def bench_geometric_product(device, args):
    def make_inputs(batch, device):
        x = torch.randn(batch, 16, dtype=DTYPE, device=device, requires_grad=True)
        y = torch.randn(batch, 16, dtype=DTYPE, device=device, requires_grad=True)
        return (x, y)

    return sweep(geometric_product, make_inputs, "geometric_product", device, args)


def main():
    args = parse_args()
    torch.manual_seed(0)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available() and not args.cpu:
        devices.append(torch.device("cuda:0"))

    sweeps = []
    for device in devices:
        utils.maybe_set_single_thread_cpu(device)
        if args.bench in ("linear", "both"):
            sweeps.append(bench_equi_linear(device, args))
        if args.bench in ("gp", "both"):
            sweeps.append(bench_geometric_product(device, args))

    utils.save_json(
        "bench_primitives",
        {
            "machine": utils.machine_info(),
            "config": {
                "dtype": str(DTYPE),
                "warmup": WARMUP,
                "iters": ITERS,
                "in_c": IN_C,
                "out_c": OUT_C,
                "batches": BATCHES,
                "bench": args.bench,
                "compile": args.compile,
                "triton": args.triton,
                "cpu": args.cpu,
                # None when --triton was not passed (we don't probe to avoid importing triton).
                "have_triton": _have_triton() if args.triton else None,
            },
            "sweeps": sweeps,
        },
    )


if __name__ == "__main__":
    main()
