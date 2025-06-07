"""Invariants, e.g. inner product, absolute squared norm, pin invariants."""
import math
from functools import lru_cache

import torch

from ..utils.einsum import cached_einsum
from ..utils.misc import minimum_autocast_precision
from .linear import grade_project


@lru_cache()
def _load_inner_product_factors(
    device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Constructs an array of 1's and -1's for the metric of the space,
    used to compute the inner product.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    ip_factors : torch.Tensor
        Inner product factors with shape (16,)
    """

    _INNER_PRODUCT_FACTORS = [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]
    factors = torch.tensor(
        _INNER_PRODUCT_FACTORS, dtype=torch.float32, device=torch.device("cpu")
    ).to_dense()
    return factors.to(device=device, dtype=dtype)


@lru_cache()
def _load_metric_grades(
    device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Generate tensor of the diagonal of the GA metric, combined with a grade projection.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    torch.Tensor
        Metric grades with shape (5, 16)
    """
    m = _load_inner_product_factors(device=torch.device("cpu"), dtype=torch.float32)
    m_grades = torch.zeros(5, 16, device=torch.device("cpu"), dtype=torch.float32)
    offset = 0
    for k in range(4 + 1):
        d = math.comb(4, k)
        m_grades[k, offset : offset + d] = m[offset : offset + d]
        offset += d
    return m_grades.to(device=device, dtype=dtype)


def inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the inner product of multivectors ``f(x,y) = <x, y> = <~x y>_0``.

    Equal to ``geometric_product(reverse(x), y)[..., [0]]`` (but faster).

    Parameters
    ----------
    x : torch.Tensor
        First input multivector with shape (..., 16) or (..., channels, 16).
        Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor
        Second input multivector with shape (..., 16) or (..., channels, 16).
        Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor
        Result with shape (..., 1).
        Batch dimensions are result of broadcasting between x and y.
    """

    x = x * _load_inner_product_factors(device=x.device, dtype=x.dtype)

    outputs = cached_einsum("... i, ... i -> ...", x, y)

    # We want the output to have shape (..., 1)
    outputs = outputs.unsqueeze(-1)

    return outputs


@minimum_autocast_precision(torch.float32)
def abs_squared_norm(x: torch.Tensor) -> torch.Tensor:
    """Computes a modified version of the squared norm that is positive semidefinite and can
    therefore be used in layer normalization.

    Parameters
    ----------
    x : torch.Tensor
        Input multivector with shape (..., 16).

    Returns
    -------
    outputs : torch.Tensor
        Geometric algebra norm of x with shape (..., 1).
    """
    m = _load_metric_grades(device=x.device, dtype=x.dtype)
    abs_squared_norms = (
        cached_einsum("... i, ... i, g i -> ... g", x, x, m).abs().sum(-1, keepdim=True)
    )
    return abs_squared_norms
