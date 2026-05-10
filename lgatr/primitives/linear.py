"""Linear operations on multivectors, in particular linear basis maps."""

import math
from functools import lru_cache
from pathlib import Path

import torch

from .config import PrimitivesConfig

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


# Module-level constants loaded once at import. lru_cache helpers below only do `.to(...)`
# casts on these, keeping them traceable under torch.compile(fullgraph=True).
def _load_basis(name: str) -> torch.Tensor:
    return torch.load(Path(__file__).parent.resolve() / name).to(DEFAULT_DTYPE).to_dense()


_BASIS_SUBGROUP = _load_basis("linear_basis_subgroup.pt")
_BASIS_FULL = _load_basis("linear_basis_full.pt")

# Subgroup dual basis: summing the 5 dual rows yields a (16, 16) sign-permutation
# (one nonzero per row). Sparse path replaces the 16-wide gather x[..., dual_perm]
# with per-grade slice + flip(-1); this assert pins the basis layout it relies on.
_DUAL = _BASIS_SUBGROUP[5:10].sum(dim=0)
_DUAL_PERM = _DUAL.abs().argmax(dim=-1)
_DUAL_SIGN = torch.gather(_DUAL, -1, _DUAL_PERM.unsqueeze(-1)).squeeze(-1)
assert torch.equal(_DUAL_PERM, torch.arange(16).flip(0)), "sparse linear: bad basis layout"


@lru_cache
def _compute_pin_equi_linear_basis(
    subgroup: bool = True,
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Lorentz-equivariant basis of shape (10, 16, 16) for the proper orthochronous subgroup,
    # or (5, 16, 16) for the full Pin group.
    src = _BASIS_SUBGROUP if subgroup else _BASIS_FULL
    return src.to(device=device, dtype=dtype)


@lru_cache
def _compute_grade_projection_mask(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Mask of shape (5, 16) selecting positions belonging to grade g in row g.
    mask = torch.zeros(5, 16, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    offset = 0
    for k in range(5):
        d = math.comb(4, k)
        mask[k, offset : offset + d] = 1
        offset += d
    return mask.to(device=device, dtype=dtype)


@lru_cache
def _compute_reversal(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Diagonal of shape (16,) for the multivector reversal (+/-1 entries).
    reversal_flat = torch.ones(16, device=device, dtype=dtype)
    reversal_flat[5:15] = -1
    return reversal_flat


@lru_cache
def _compute_grade_involution(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Diagonal of shape (16,) for the multivector grade involution (+/-1 entries).
    involution_flat = torch.ones(16, device=device, dtype=dtype)
    involution_flat[1:5] = -1
    involution_flat[11:15] = -1
    return involution_flat


@lru_cache
def _compute_dual_sign(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Per-position dual-basis signs of shape (16,), cast to (device, dtype).
    return _DUAL_SIGN.to(device=device, dtype=dtype)


def _equi_linear_dense(
    x: torch.Tensor, coeffs: torch.Tensor, *, config: PrimitivesConfig
) -> torch.Tensor:
    # Equation: out[..., y, i] = sum_{x, a, j} coeffs[y, x, a] * basis[a, i, j] * x[..., x, j]
    # basis is ~1% nonzero, so this path spends most of its FLOPs on zero entries; the sparse
    # path skips those at the cost of less optimized kernels (no fused BLAS GEMM).
    basis = _compute_pin_equi_linear_basis(config.subgroup, device=x.device, dtype=x.dtype)
    # Fold (coeffs, basis) into an effective (out_c, in_c, 16, 16) weight via one GEMM,
    # then contract that weight with x.
    weight = (coeffs @ basis.flatten(-2)).unflatten(-1, (16, 16))
    return torch.einsum("y x i j, ... x j -> ... y i", weight, x)


def _equi_linear_sparse(
    x: torch.Tensor, coeffs: torch.Tensor, *, config: PrimitivesConfig
) -> torch.Tensor:
    # Per-grade matmul on contiguous slices of x: avoids materializing the (10, 16, 16) basis
    # and the multi-axis einsum.
    batch_shape = x.shape[:-2]
    in_c = x.shape[-2]

    def grade_mm(W: torch.Tensor, x_g: torch.Tensor) -> torch.Tensor:
        # (out_c, in_c) @ (..., in_c, dim_g) -> (..., out_c, dim_g) as one 2-D GEMM via
        # F.linear; the natural per-batch BMM is several times slower for small in_c.
        dim_g = x_g.shape[-1]
        x_flat = x_g.transpose(-1, -2).reshape(-1, in_c)
        y_flat = torch.nn.functional.linear(x_flat, W)
        return y_flat.unflatten(0, (*batch_shape, dim_g)).transpose(-1, -2)

    # Contiguous grade slices of widths 1, 4, 6, 4, 1.
    x0 = x[..., 0:1]
    x1 = x[..., 1:5]
    x2 = x[..., 5:11]
    x3 = x[..., 11:15]
    x4 = x[..., 15:16]

    if config.subgroup:
        # 10 basis elements: indices 0..4 are grade-preserving, 5..9 are the Hodge dual mapping
        # grade g -> grade 4-g. The dual reduces to a sign-flip + per-grade reverse view, so each
        # term stays a single GEMM (no full 16-wide gather).
        s = _compute_dual_sign(device=x.device, dtype=x.dtype)
        y0 = grade_mm(coeffs[..., 0], x0) + grade_mm(coeffs[..., 9], s[0:1] * x4.flip(-1))
        y1 = grade_mm(coeffs[..., 1], x1) + grade_mm(coeffs[..., 8], s[1:5] * x3.flip(-1))
        y2 = grade_mm(coeffs[..., 2], x2) + grade_mm(coeffs[..., 7], s[5:11] * x2.flip(-1))
        y3 = grade_mm(coeffs[..., 3], x3) + grade_mm(coeffs[..., 6], s[11:15] * x1.flip(-1))
        y4 = grade_mm(coeffs[..., 4], x4) + grade_mm(coeffs[..., 5], s[15:16] * x0.flip(-1))
    else:
        # Full Pin group: only the 5 grade-preserving basis elements.
        y0 = grade_mm(coeffs[..., 0], x0)
        y1 = grade_mm(coeffs[..., 1], x1)
        y2 = grade_mm(coeffs[..., 2], x2)
        y3 = grade_mm(coeffs[..., 3], x3)
        y4 = grade_mm(coeffs[..., 4], x4)

    return torch.cat((y0, y1, y2, y3, y4), dim=-1)


def equi_linear(x: torch.Tensor, coeffs: torch.Tensor, *, config: PrimitivesConfig) -> torch.Tensor:
    """Pin-equivariant linear map ``f(x) = sum_{a,j} coeffs_a W^a_ij x_j``.

    The :math:`W^a` are 5 or 10 pre-defined basis elements (see :func:`_compute_pin_equi_linear_basis`).

    Parameters
    ----------
    x
        Input multivector of shape ``(..., in_channels, 16)``.
    coeffs
        Coefficients for the basis elements of shape ``(out_channels, in_channels, num_basis_elements)``,
        where ``num_basis_elements`` is 10 (fully connected subgroup) or 5 (full Lorentz group).
    config
        LGATr primitives configuration.

    Returns
    -------
    outputs
        Result of shape ``(..., out_channels, 16)``.
    """
    if config.sparse:
        return _equi_linear_sparse(x, coeffs, config=config)
    return _equi_linear_dense(x, coeffs, config=config)


def grade_project(x: torch.Tensor) -> torch.Tensor:
    """Project a multivector onto its individual grades.

    The result is a single tensor with a new grade dimension.

    Parameters
    ----------
    x
        Input multivector of shape ``(..., 16)``.

    Returns
    -------
    outputs
        Output multivector of shape ``(..., 5, 16)``. The second-to-last dimension indexes grades.
    """
    # Equivalent to ``einsum("g i j, ... j -> ... g i", basis[:5], x)``: the first five basis
    # elements are pure-diagonal grade projectors, so the (5, 16, 16) operand collapses to a
    # (5, 16) mask of their diagonals.
    mask = _compute_grade_projection_mask(device=x.device, dtype=x.dtype)
    return x.unsqueeze(-2) * mask


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Compute the reversal of a multivector.

    The reversal preserves the scalar, vector, and pseudoscalar components and flips the sign of
    the bivector and axialvector components.

    Parameters
    ----------
    x
        Input multivector of shape ``(..., 16)``.

    Returns
    -------
    outputs
        Output multivector of shape ``(..., 16)``.
    """
    return _compute_reversal(device=x.device, dtype=x.dtype) * x


def grade_involute(x: torch.Tensor) -> torch.Tensor:
    """Compute the grade involution of a multivector.

    The grade involution preserves the scalar, bivector, and pseudoscalar components and flips the
    sign of the vector and axialvector components.

    Parameters
    ----------
    x
        Input multivector of shape ``(..., 16)``.

    Returns
    -------
    outputs
        Output multivector of shape ``(..., 16)``.
    """

    return _compute_grade_involution(device=x.device, dtype=x.dtype) * x
