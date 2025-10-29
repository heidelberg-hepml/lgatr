"""Geometric product."""

from functools import lru_cache
from pathlib import Path

import torch

from ..utils.einsum import cached_einsum
from .linear import DEFAULT_DEVICE, DEFAULT_DTYPE


@lru_cache
def _load_geometric_product_tensor(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) -> torch.Tensor:
    """Loads geometric product tensor for geometric product between multivectors.

    This function is cached.

    Parameters
    ----------
    device : torch.Device or str
        Device
    dtype : torch.Dtype
        Data type

    Returns
    -------
    basis : torch.Tensor
        Geometric product tensor with shape (16, 16, 16)
    """

    # To avoid duplicate loading, base everything on float32 CPU version
    if device not in [DEFAULT_DEVICE, "cpu"] and dtype != DEFAULT_DTYPE:
        gmt = _load_geometric_product_tensor()
    else:
        filename = Path(__file__).parent.resolve() / "geometric_product.pt"
        gmt = torch.load(filename).to(DEFAULT_DTYPE).to_dense()

    return gmt.to(device=device, dtype=dtype)


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the geometric product ``f(x,y) = x*y``.

    Parameters
    ----------
    x : torch.Tensor
        First input multivector with shape (..., 16).
        Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor
        Second input multivector with shape (..., 16).
        Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor
        Result with shape (..., 16).
        Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    # Select kernel on correct device
    gp = _load_geometric_product_tensor(device=x.device, dtype=x.dtype)

    # Compute geometric product
    outputs = cached_einsum("i j k, ... j, ... k -> ... i", gp, x, y)

    return outputs
