"""Geometric product."""

from functools import lru_cache
from pathlib import Path

import torch

from .config import PrimitivesConfig
from .linear import _LIGHTCONE_FRAME_MV, DEFAULT_DEVICE, DEFAULT_DTYPE

# Module-level constants loaded once at import. Lru_cache helpers below only do `.to(...)`
# casts on these, keeping them traceable under torch.compile(fullgraph=True).
_GP = (
    torch.load(Path(__file__).parent.resolve() / "geometric_product.pt", weights_only=True)
    .to(DEFAULT_DTYPE)
    .to_dense()
)
# Each row gp[i, j, :] has exactly one nonzero (a +/-1 entry); store its column and sign.
_GP_INDICES = _GP.abs().argmax(dim=-1)
_GP_SIGNS = torch.gather(_GP, -1, _GP_INDICES.unsqueeze(-1)).squeeze(-1)

# Light-cone coordinates, see _LIGHTCONE_FRAME_MV.
_GP_LIGHTCONE = (
    torch.einsum(
        "ai,ijk,bj,ck->abc",
        _LIGHTCONE_FRAME_MV,
        _GP.double(),
        _LIGHTCONE_FRAME_MV,
        _LIGHTCONE_FRAME_MV,
    )
    .round()
    .to(DEFAULT_DTYPE)
    .contiguous()
)
# Unlike above, fixing (i, j) can give zero or two k (e.g. e+ e- = 1 + e+^e-), so store the x index
# j and the y index k of each of the 16 terms per output i.
assert torch.equal((_GP_LIGHTCONE != 0).sum(dim=(1, 2)), torch.full((16,), 16))
_GP_LIGHTCONE_XIDX, _GP_LIGHTCONE_YIDX = _GP_LIGHTCONE.nonzero()[:, 1:].view(16, 16, 2).unbind(-1)
_GP_LIGHTCONE_SIGNS = _GP_LIGHTCONE[_GP_LIGHTCONE != 0].view(16, 16)


@lru_cache
def _load_geometric_product_tensor(
    lightcone: bool = False,
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Geometric-product tensor of shape (16, 16, 16), cast to (device, dtype).
    gp = _GP_LIGHTCONE if lightcone else _GP
    return gp.to(device=device, dtype=dtype)


@lru_cache
def _compute_sparse_gp_indices(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    # (indices, signs) of shape (16, 16) each, cast to (device, dtype).
    return _GP_INDICES.to(device=device), _GP_SIGNS.to(device=device, dtype=dtype)


@lru_cache
def _compute_sparse_gp_lightcone_indices(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (x indices, y indices, signs) of shape (16, 16) each, cast to (device, dtype).
    return (
        _GP_LIGHTCONE_XIDX.to(device=device),
        _GP_LIGHTCONE_YIDX.to(device=device),
        _GP_LIGHTCONE_SIGNS.to(device=device, dtype=dtype),
    )


def _geometric_product_dense(x: torch.Tensor, y: torch.Tensor, lightcone: bool) -> torch.Tensor:
    # Equation: out[..., i] = sum_{j, k} gp[i, j, k] * x[..., j] * y[..., k]
    gp = _load_geometric_product_tensor(lightcone, device=x.device, dtype=x.dtype)
    # Build the (..., 16, 16) outer product x_j * y_k, then matmul against the flattened gp[i, j*k]
    outer = x.unsqueeze(-1) * y.unsqueeze(-2)
    return outer.flatten(-2, -1) @ gp.flatten(1, 2).T


def _geometric_product_sparse(x: torch.Tensor, y: torch.Tensor, lightcone: bool) -> torch.Tensor:
    # out[..., i] = sum_j signs[i, j] * x[..., j] * y[..., indices[i, j]]. torch.compile fuses
    # away the (..., 16, 16) gather, which eager materializes. Its compiled backward is slow in
    # (b)float16, so gather in at least float32.
    dtype = torch.promote_types(x.dtype, y.dtype)
    gather_dtype = torch.promote_types(dtype, torch.float32)
    x, y = x.to(gather_dtype), y.to(gather_dtype)
    if lightcone:
        # out[..., i] = sum_n signs[i, n] * x[..., xidx[i, n]] * y[..., yidx[i, n]]
        xidx, yidx, signs = _compute_sparse_gp_lightcone_indices(device=x.device, dtype=x.dtype)
        return (signs * x[..., xidx] * y[..., yidx]).sum(-1).to(dtype)
    indices, signs = _compute_sparse_gp_indices(device=x.device, dtype=x.dtype)
    return (signs * y[..., indices] * x.unsqueeze(-2)).sum(-1).to(dtype)


def geometric_product(
    x: torch.Tensor, y: torch.Tensor, *, config: PrimitivesConfig
) -> torch.Tensor:
    """Compute the geometric product ``f(x, y) = x * y``.

    Parameters
    ----------
    x
        First input multivector of shape ``(..., 16)``.
        Batch dimensions must be broadcastable between ``x`` and ``y``.
    y
        Second input multivector of shape ``(..., 16)``.
        Batch dimensions must be broadcastable between ``x`` and ``y``.
    config
        LGATr primitives configuration.

    Returns
    -------
    outputs
        Result of shape ``(..., 16)``. Batch dimensions are the broadcast of ``x`` and ``y``.
    """
    if config.sparse_gp:
        return _geometric_product_sparse(x, y, config.lightcone)
    return _geometric_product_dense(x, y, config.lightcone)
