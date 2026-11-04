"""Misc utilities: float16 saturation, residual addition and an activation-callable lookup."""

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F


class _Saturate(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        limit = torch.finfo(x.dtype).max
        return x.clamp(-limit, limit)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        pass

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad


def saturate(x: torch.Tensor | None) -> torch.Tensor | None:
    """Clamp a float16 tensor to the finite float16 range; other dtypes and None pass through.

    Overflowed values (``±inf``) become ``±65504`` instead of turning into NaN downstream. The
    gradient is passed through unchanged, so no mask has to be saved for the backward pass.
    Only float16 is affected because bfloat16 and float32 share the much larger float32 range.
    """
    if x is None or x.dtype != torch.float16:
        return x
    return _Saturate.apply(x)


def residual_add(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    """Residual addition that propagates None and saturates float16 overflow.

    Returns None if both operands are None, the non-None operand if exactly one is None, and
    ``saturate(a + b)`` otherwise. Used by transformer blocks for all residual connections, so
    the optional scalar stream needs no special-casing at each call site.
    """
    if a is None:
        return b
    if b is None:
        return a
    return saturate(a + b)


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
