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


def _to_pair_basis(x: torch.Tensor) -> torch.Tensor:
    # Light-cone multivectors with the blade pairs (a, b) = (1, e+-), (e1, e+-1), (e2, e+-2),
    # (e12, e+-12) replaced by (a + b, a - b), ordered as (a + b, a - b, other blades). A pair has
    # equal boost weight, so this never mixes large and small components.
    a = torch.cat([x[..., 0:1], x[..., 3:5], x[..., 10:11]], dim=-1)
    b = torch.cat([x[..., 5:6], x[..., 11:13], x[..., 15:16]], dim=-1)
    rest = torch.cat([x[..., 1:3], x[..., 6:10], x[..., 13:15]], dim=-1)
    return torch.cat([a + b, a - b, rest], dim=-1)


# Unlike _GP, _GP_LIGHTCONE[i, j, :] can have two nonzeros, at a blade and its product with e+-
# (e.g. e+ e- = 1 + e+-). With x and y in the pair basis, there is again at most one.
_PAIR_BASIS_INV = torch.linalg.inv(_to_pair_basis(torch.eye(16, dtype=torch.float64)).T)
_GP_PAIR = torch.einsum("ijk,jl,km->ilm", _GP_LIGHTCONE.double(), _PAIR_BASIS_INV, _PAIR_BASIS_INV)
assert ((_GP_PAIR != 0).sum(dim=-1) <= 1).all()
_GP_LIGHTCONE_INDICES = _GP_PAIR.abs().argmax(dim=-1)
_GP_LIGHTCONE_SIGNS = torch.gather(_GP_PAIR, -1, _GP_LIGHTCONE_INDICES.unsqueeze(-1)).squeeze(-1)
# Point the zero entries at the unused columns, making each row a permutation as in the Cartesian
# case. Else the compiled backward piles its scatter-add atomics onto column 0.
for _indices, _signs in zip(_GP_LIGHTCONE_INDICES, _GP_LIGHTCONE_SIGNS, strict=True):
    _indices[_signs == 0] = torch.tensor(
        [k for k in range(16) if k not in _indices[_signs != 0]], dtype=torch.long
    )
assert torch.equal(_GP_LIGHTCONE_INDICES.sort(dim=-1).values, torch.arange(16).expand(16, 16))
_GP_LIGHTCONE_SIGNS = _GP_LIGHTCONE_SIGNS.to(DEFAULT_DTYPE)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    # (indices, signs) of shape (16, 16), cast to (device, dtype). Light-cone signs are on the pair
    # basis, with values in {0, +/-1/2, +/-1}.
    indices, signs = (
        (_GP_LIGHTCONE_INDICES, _GP_LIGHTCONE_SIGNS) if lightcone else (_GP_INDICES, _GP_SIGNS)
    )
    return indices.to(device=device), signs.to(device=device, dtype=dtype)


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
    indices, signs = _compute_sparse_gp_indices(lightcone, device=x.device, dtype=x.dtype)
    if lightcone:
        x, y = _to_pair_basis(x), _to_pair_basis(y)
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
