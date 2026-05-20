"""Misc utilities: residual addition and an activation-callable lookup."""

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F


def residual_add(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    """Residual addition that propagates None.

    Returns None if both operands are None, the non-None operand if exactly one is None, and
    ``a + b`` otherwise. Used by transformer blocks to add residual connections on the optional
    scalar stream without special-casing each call site.
    """
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def get_nonlinearity(label: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the ``torch.nn.functional`` activation for the given label.

    Accepts ``"relu"``, ``"sigmoid"``, ``"tanh"``, ``"gelu"``, or ``"silu"``. Returned
    callables are stateless (no parameters, no buffers) and apply the activation
    elementwise. ``"gelu"`` uses ``approximate="tanh"``.
    """
    if label == "relu":
        return F.relu
    elif label == "sigmoid":
        return F.sigmoid
    elif label == "tanh":
        return F.tanh
    elif label == "gelu":
        return partial(F.gelu, approximate="tanh")
    elif label == "silu":
        return F.silu
    else:
        raise ValueError(f"Unsupported nonlinearity type: {label}")
