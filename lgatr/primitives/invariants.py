"""Invariants: inner product, absolute squared norm, and Pin-invariant utilities."""

from functools import lru_cache

import torch

from .config import PrimitivesConfig
from .linear import _LIGHTCONE_FRAME_MV, DEFAULT_DEVICE, DEFAULT_DTYPE

# Diagonal of the GA metric (signature of the inner product on each multivector grade).
_INNER_PRODUCT_FACTORS = torch.tensor(
    [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
    dtype=DEFAULT_DTYPE,
    device=DEFAULT_DEVICE,
)
# In light-cone coordinates, the metric is a signed permutation.
_LIGHTCONE_METRIC = (
    _LIGHTCONE_FRAME_MV @ _INNER_PRODUCT_FACTORS.double().diag() @ _LIGHTCONE_FRAME_MV.T
).round()
_LIGHTCONE_METRIC_PERM = _LIGHTCONE_METRIC.abs().argmax(dim=-1)
_LIGHTCONE_METRIC_SIGNS = (
    torch.gather(_LIGHTCONE_METRIC, -1, _LIGHTCONE_METRIC_PERM.unsqueeze(-1))
    .squeeze(-1)
    .to(DEFAULT_DTYPE)
)


@lru_cache
def _load_inner_product_factors(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Diagonal of the GA metric used by the inner product, shape (16,) (+/-1 entries).
    return _INNER_PRODUCT_FACTORS.to(device=device, dtype=dtype)


@lru_cache
def _load_lightcone_metric(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    # (permutation, signs) of shape (16,) each of the light-cone metric, cast to (device, dtype).
    perm = _LIGHTCONE_METRIC_PERM.to(device=device)
    return perm, _LIGHTCONE_METRIC_SIGNS.to(device=device, dtype=dtype)


def _apply_metric(x: torch.Tensor, lightcone: bool) -> torch.Tensor:
    """GA metric applied to multivectors over the component dim (-1)."""
    if lightcone:
        perm, signs = _load_lightcone_metric(device=x.device, dtype=x.dtype)
        return x[..., perm] * signs
    return x * _load_inner_product_factors(device=x.device, dtype=x.dtype)


def inner_product(x: torch.Tensor, y: torch.Tensor, *, config: PrimitivesConfig) -> torch.Tensor:
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
    config
        LGATr primitives configuration.

    Returns
    -------
    outputs
        Result of shape ``(..., 1)``. Batch dimensions are the broadcast of ``x`` and ``y``.
    """

    x = _apply_metric(x, config.lightcone)

    outputs = (x * y).sum(-1, keepdim=True)

    return outputs


def abs_squared_norm(x: torch.Tensor, *, config: PrimitivesConfig) -> torch.Tensor:
    """Compute a positive-semidefinite modification of the squared norm.

    Suitable for layer normalization (the standard GA squared norm is not positive semidefinite).

    Parameters
    ----------
    x
        Input multivector of shape ``(..., 16)``.
    config
        LGATr primitives configuration.

    Returns
    -------
    outputs
        Geometric-algebra norm of ``x``, shape ``(..., 1)``.
    """
    if config.lightcone:
        # Components 1, (+, -, 1, 2), (+-, +1, +2, -1, -2, 12), (+-1, +-2, +12, -12), +-12.
        sq = x * x
        return (
            sq[..., 0:1]
            + (2 * x[..., 1:2] * x[..., 2:3] - sq[..., 3:5].sum(-1, keepdim=True)).abs()
            + (
                sq[..., 10:11]
                - sq[..., 5:6]
                - 2 * (x[..., 6:8] * x[..., 8:10]).sum(-1, keepdim=True)
            ).abs()
            + (sq[..., 11:13].sum(-1, keepdim=True) + 2 * x[..., 13:14] * x[..., 14:15]).abs()
            + sq[..., 15:16]
        )
    # Per-grade slice sums rather than a single matmul: lower activation memory under compile.
    signed = x * x * _load_inner_product_factors(device=x.device, dtype=x.dtype)
    return (
        signed[..., 0:1].sum(-1, keepdim=True).abs()
        + signed[..., 1:5].sum(-1, keepdim=True).abs()
        + signed[..., 5:11].sum(-1, keepdim=True).abs()
        + signed[..., 11:15].sum(-1, keepdim=True).abs()
        + signed[..., 15:16].sum(-1, keepdim=True).abs()
    )
