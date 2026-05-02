"""Invariants: inner product, absolute squared norm, and Pin-invariant utilities."""

import math
from functools import lru_cache

import torch

from ..utils.einsum import custom_einsum
from ..utils.misc import minimum_autocast_precision
from .linear import DEFAULT_DEVICE, DEFAULT_DTYPE


@lru_cache
def _load_inner_product_factors(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) -> torch.Tensor:
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
    ip_factors
        Inner-product factors of shape ``(16,)`` (entries are +1 or -1).
    """

    _INNER_PRODUCT_FACTORS = [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]
    factors = torch.tensor(
        _INNER_PRODUCT_FACTORS, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    ).to_dense()
    return factors.to(device=device, dtype=dtype)


@lru_cache
def _load_metric_grades(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Construct the GA metric diagonal combined with a grade projection.

    Returns
    -------
    metric_grades
        Tensor of shape ``(5, 16)``; row ``g`` holds the metric entries of grade ``g`` and zeros
        elsewhere.
    """
    m = _load_inner_product_factors(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    m_grades = torch.zeros(5, 16, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    offset = 0
    for k in range(4 + 1):
        d = math.comb(4, k)
        m_grades[k, offset : offset + d] = m[offset : offset + d]
        offset += d
    return m_grades.to(device=device, dtype=dtype)


def inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the inner product of multivectors ``f(x, y) = <x, y> = <~x y>_0``.

    Equal to ``geometric_product(reverse(x), y)[..., [0]]``, but faster.

    Parameters
    ----------
    x
        First input multivector of shape ``(..., 16)`` or ``(..., channels, 16)``.
        Batch dimensions must be broadcastable between ``x`` and ``y``.
    y
        Second input multivector of shape ``(..., 16)`` or ``(..., channels, 16)``.
        Batch dimensions must be broadcastable between ``x`` and ``y``.

    Returns
    -------
    outputs
        Result of shape ``(..., 1)``. Batch dimensions are the broadcast of ``x`` and ``y``.
    """

    x = x * _load_inner_product_factors(device=x.device, dtype=x.dtype)

    outputs = (x * y).sum(-1, keepdim=True)

    return outputs


@minimum_autocast_precision(torch.float32)
def abs_squared_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute a positive-semidefinite modification of the squared norm.

    Suitable for layer normalization (the standard GA squared norm is not positive semidefinite).

    Parameters
    ----------
    x
        Input multivector of shape ``(..., 16)``.

    Returns
    -------
    outputs
        Geometric-algebra norm of ``x``, shape ``(..., 1)``.
    """
    m = _load_metric_grades(device=x.device, dtype=x.dtype)
    # Path captured via opt_einsum's "optimal" strategy for shapes (...,16), (...,16), (5,16);
    # deterministic across batch sizes.
    abs_squared_norms = (
        custom_einsum("... i, ... i, g i -> ... g", x, x, m, path=[0, 1, 0, 1])
        .abs()
        .sum(-1, keepdim=True)
    )
    return abs_squared_norms
