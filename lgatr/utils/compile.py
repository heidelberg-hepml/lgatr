"""Helpers for using L-GATr networks with :func:`torch.compile`."""

import torch
from torch import nn

from ..primitives.compile import warmup_caches


def compile_model(
    model: nn.Module,
    *,
    compile_mode: str = "default",
    compile_dynamic: bool = False,
    compile_fullgraph: bool = False,
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
    """
    model.forward = torch.compile(
        model.forward,
        mode=compile_mode,
        dynamic=compile_dynamic,
        fullgraph=compile_fullgraph,
    )


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
