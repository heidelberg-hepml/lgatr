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
# Unlike above, gp[i, j, :] can have two nonzeros, at a blade and its product with e+- (e.g.
# e+ e- = 1 + e+-). Extending y by y_a - y_b and y_a + y_b for these pairs (a, b) leaves one.
_GP_LIGHTCONE_PAIRS = torch.tensor([[0, 3, 4, 10], [5, 11, 12, 15]])
_A, _B = torch.eye(16)[_GP_LIGHTCONE_PAIRS]
_Y_EXT = torch.cat([torch.eye(16), _A - _B, _A + _B])
_GP_ROWS = _GP_LIGHTCONE.unsqueeze(-2)
_GP_LIGHTCONE_EXT = (_GP_ROWS == _Y_EXT).all(-1).float() - (_GP_ROWS == -_Y_EXT).all(-1).float()
assert torch.equal(_GP_LIGHTCONE_EXT @ _Y_EXT, _GP_LIGHTCONE)
_GP_LIGHTCONE_INDICES = _GP_LIGHTCONE_EXT.abs().argmax(dim=-1)
_GP_LIGHTCONE_SIGNS = torch.gather(
    _GP_LIGHTCONE_EXT, -1, _GP_LIGHTCONE_INDICES.unsqueeze(-1)
).squeeze(-1)


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
    lightcone: bool = False,
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (indices, signs) of shape (16, 16), cast to (device, dtype), and the light-cone pairs.
    indices, signs = (
        (_GP_LIGHTCONE_INDICES, _GP_LIGHTCONE_SIGNS) if lightcone else (_GP_INDICES, _GP_SIGNS)
    )
    return (
        indices.to(device=device),
        signs.to(device=device, dtype=dtype),
        _GP_LIGHTCONE_PAIRS.to(device=device),
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
    indices, signs, (a, b) = _compute_sparse_gp_indices(lightcone, device=x.device, dtype=x.dtype)
    if lightcone:
        ya, yb = y[..., a], y[..., b]
        y = torch.cat([y, ya - yb, ya + yb], dim=-1)
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
