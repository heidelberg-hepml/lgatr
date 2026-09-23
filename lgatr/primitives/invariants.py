"""Invariants: inner product, absolute squared norm, and Pin-invariant utilities."""

from functools import lru_cache

import torch

from .linear import DEFAULT_DEVICE, DEFAULT_DTYPE

# Diagonal of the GA metric (signature of the inner product on each multivector grade).
_INNER_PRODUCT_FACTORS = torch.tensor(
    [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
    dtype=DEFAULT_DTYPE,
    device=DEFAULT_DEVICE,
)


@lru_cache
def _load_inner_product_factors(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Diagonal of the GA metric used by the inner product, shape (16,) (+/-1 entries).
    return _INNER_PRODUCT_FACTORS.to(device=device, dtype=dtype)


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
    # Per-grade slice sums rather than a single matmul: lower activation memory under compile.
    signed = x * x * _load_inner_product_factors(device=x.device, dtype=x.dtype)
    return (
        signed[..., 0:1].sum(-1, keepdim=True).abs()
        + signed[..., 1:5].sum(-1, keepdim=True).abs()
        + signed[..., 5:11].sum(-1, keepdim=True).abs()
        + signed[..., 11:15].sum(-1, keepdim=True).abs()
        + signed[..., 15:16].sum(-1, keepdim=True).abs()
    )
