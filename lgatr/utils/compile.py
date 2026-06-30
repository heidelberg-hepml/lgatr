"""Helpers for using L-GATr networks with :func:`torch.compile`."""

import torch
import torch._functorch.config
from torch import nn

from ..primitives.compile import warmup_caches


def compile_model(
    model: nn.Module,
    *,
    compile_mode: str = "default",
    compile_dynamic: bool = False,
    compile_fullgraph: bool = False,
    activation_memory_budget: float | None = 0.5,
) -> None:
    """Wrap ``model.forward`` with :func:`torch.compile` in place.

    Rebinding ``self.forward`` rather than patching the class keeps the compilation local
    to this instance.

    Parameters
    ----------
    model
        The :class:`torch.nn.Module` whose ``forward`` should be compiled.
    compile_mode
        Mode passed to :func:`torch.compile` (e.g. ``"default"``, ``"reduce-overhead"``).
    compile_dynamic
        Whether to use dynamic shapes.
    compile_fullgraph
        Whether to require a full graph (no graph breaks).
    activation_memory_budget
        Fraction in ``[0, 1]`` for the partitioner's activation-memory budget. ``1.0`` recomputes
        only cheap pointwise/reduction ops in the backward pass (torch default); lower
        values let the partitioner also recompute compute-intensive ops, ranked by
        memory-saved-per-FLOP, trading backward FLOPs for a smaller activation memory peak. ``None``
        leaves the global setting untouched. At the default ``0.5`` the recomputed ops are
        typically the linear/GLU projections, while attention outputs stay saved.
        Compared to the default ``1.0`` this reduces memory by ~30% for LGATrSlim at ~4% GPU slowdown.
        LGATr is dominated by the custom autograd functions for the geometric product and linear
        which are not affected by activation_memory_budget, therefore LGATr is not affected by this.
        Applied via a scoped patch that is only in effect while this model (re)compiles.
    """
    compiled = torch.compile(
        model.forward,
        mode=compile_mode,
        dynamic=compile_dynamic,
        fullgraph=compile_fullgraph,
    )
    if activation_memory_budget is not None:
        compiled = torch._functorch.config.patch(activation_memory_budget=activation_memory_budget)(
            compiled
        )
    model.forward = compiled


def warmup_after_apply(model: nn.Module) -> None:
    """Warm L-GATr's primitive caches for the model's current device and dtype.

    Intended to be called from a :meth:`torch.nn.Module._apply` override so the caches are
    populated whenever the model is moved or cast (``.to()`` / ``.cuda()`` / ``.float()`` / etc.).

    Parameters
    ----------
    model
        The :class:`torch.nn.Module` whose primitive caches should be warmed.
    """
    p = next(model.parameters(), None)
    if p is not None and p.is_floating_point():
        warmup_caches(p.device, p.dtype)
