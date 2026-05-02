"""Geometric product."""

from functools import lru_cache
from pathlib import Path

import torch

from .linear import DEFAULT_DEVICE, DEFAULT_DTYPE


def _load_gp_tensor() -> torch.Tensor:
    path = Path(__file__).parent.resolve() / "geometric_product.pt"
    return torch.load(path).to(DEFAULT_DTYPE).to_dense()


# Loaded once at import time. Subsequent device/dtype views are pure tensor ops.
_GP = _load_gp_tensor()


@lru_cache
def _load_geometric_product_tensor(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Load the geometric-product tensor for multivectors.

    Cached via :func:`functools.lru_cache`.

    Returns
    -------
    basis
        Geometric-product tensor of shape ``(16, 16, 16)``.
    """
    return _GP.to(device=device, dtype=dtype)


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the geometric product ``f(x, y) = x * y``.

    Parameters
    ----------
    x
        First input multivector of shape ``(..., 16)``.
        Batch dimensions must be broadcastable between ``x`` and ``y``.
    y
        Second input multivector of shape ``(..., 16)``.
        Batch dimensions must be broadcastable between ``x`` and ``y``.

    Returns
    -------
    outputs
        Result of shape ``(..., 16)``. Batch dimensions are the broadcast of ``x`` and ``y``.
    """
    gp = _load_geometric_product_tensor(device=x.device, dtype=x.dtype)
    # Equivalent to ``einsum("i j k, ... j, ... k -> ... i", gp, x, y)``: form the
    # ``(..., 16, 16)`` outer product of ``x`` and ``y``, then matmul against the flattened
    # ``gp`` slice. Faster than letting opt_einsum choose a path on either CPU or GPU.
    outer = x.unsqueeze(-1) * y.unsqueeze(-2)
    return outer.flatten(-2, -1) @ gp.flatten(1, 2).T
