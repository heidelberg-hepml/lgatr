"""Helpers for using L-GATr networks with :func:`torch.compile`."""

from collections.abc import Mapping

import torch
import torch._functorch.config
from torch import nn

from ..primitives.compile import warmup_caches


def compile_model(
    model: nn.Module,
    *,
    compile_kwargs: Mapping | None = None,
    activation_memory_budget: float | None = None,
) -> None:
    """Wrap ``model.forward`` with :func:`torch.compile` in place.

    Rebinding ``self.forward`` rather than patching the class keeps the compilation local
    to this instance.

    Parameters
    ----------
    model
        The :class:`torch.nn.Module` whose ``forward`` should be compiled.
    compile_kwargs
        Forwarded verbatim to :func:`torch.compile` (e.g. ``mode``, ``dynamic``,
        ``fullgraph``, ``backend``). Any key omitted falls back to torch's own default.
    activation_memory_budget
        Fraction in ``[0, 1]`` for the partitioner's activation-memory budget; lower values trade
        backward FLOPs for a smaller activation memory peak. ``None`` (default) leaves torch's
        global setting untouched. Applied via a scoped patch only in effect while this model
        (re)compiles.
    """
    compiled = torch.compile(model.forward, **dict(compile_kwargs or {}))
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
