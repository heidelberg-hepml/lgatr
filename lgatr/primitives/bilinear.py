"""Geometric product."""

from functools import lru_cache
from pathlib import Path

import torch

from .config import PrimitivesConfig
from .linear import DEFAULT_DEVICE, DEFAULT_DTYPE

# Module-level constants loaded once at import. Lru_cache helpers below only do `.to(...)`
# casts on these, keeping them traceable under torch.compile(fullgraph=True).
_GP = (
    torch.load(Path(__file__).parent.resolve() / "geometric_product.pt")
    .to(DEFAULT_DTYPE)
    .to_dense()
)
# Each row gp[i, j, :] has exactly one nonzero (a +/-1 entry); store its column and sign.
_GP_INDICES = _GP.abs().argmax(dim=-1)
_GP_SIGNS = torch.gather(_GP, -1, _GP_INDICES.unsqueeze(-1)).squeeze(-1)


@lru_cache
def _load_geometric_product_tensor(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Geometric-product tensor of shape (16, 16, 16), cast to (device, dtype).
    return _GP.to(device=device, dtype=dtype)


@lru_cache
def _compute_sparse_gp_indices(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    # (indices, signs) of shape (16, 16) each, cast to (device, dtype).
    return _GP_INDICES.to(device=device), _GP_SIGNS.to(device=device, dtype=dtype)


def _geometric_product_dense(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Equation: out[..., i] = sum_{j, k} gp[i, j, k] * x[..., j] * y[..., k]
    gp = _load_geometric_product_tensor(device=x.device, dtype=x.dtype)
    # Build the (..., 16, 16) outer product x_j * y_k, then matmul against the flattened gp[i, j*k]
    outer = x.unsqueeze(-1) * y.unsqueeze(-2)
    return outer.flatten(-2, -1) @ gp.flatten(1, 2).T


def _geometric_product_sparse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Each gp row has one nonzero, so the contraction collapses to:
    #   out[..., i] = sum_j signs[i, j] * x[..., j] * y[..., indices[i, j]]
    indices, signs = _compute_sparse_gp_indices(device=x.device, dtype=x.dtype)
    # Advanced indexing gathers the (..., 16, 16) y-slice indexed by the sparse indices,
    # so the per-row sum over k collapses to a single position.
    y_gathered = y[..., indices]
    # Absorb the +/-1 signs into the gathered y, then contract over j as a (16) @ (16, 1) matmul.
    m = signs * y_gathered
    return torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)


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
    if config.triton:
        try:
            from .triton import geometric_product_triton
            from .triton._utils import can_dispatch
        except ImportError:
            pass
        else:
            if can_dispatch(x, y):
                return geometric_product_triton(x, y)
    if config.sparse:
        return _geometric_product_sparse(x, y)
    return _geometric_product_dense(x, y)
