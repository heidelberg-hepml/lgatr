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
    path = Path(__file__).parent.resolve() / name
    return torch.load(path, weights_only=True).to(DEFAULT_DTYPE).to_dense()


_BASIS_SUBGROUP = _load_basis("linear_basis_subgroup.pt")
_BASIS_FULL = _load_basis("linear_basis_full.pt")

# Subgroup dual basis: summing the 5 dual rows gives a (16, 16) sign-permutation. The sparse path
# replaces the gather x[..., dual_perm] with a slice and a flip per grade; the assert pins that.
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
    # or (5, 16, 16) for the full Lorentz group.
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
    # Fold (coeffs, basis) into an effective (out_c, in_c, 16, 16) weight via one GEMM, then
    # contract it with x. That block is 12.5% nonzero (6.25% for the full Lorentz group), so this
    # path does 8-16x the necessary multiply-adds; _equi_linear_sparse trades it for narrow GEMMs.
    basis = _compute_pin_equi_linear_basis(config.subgroup, device=x.device, dtype=x.dtype)
    weight = (coeffs @ basis.flatten(-2)).unflatten(-1, (16, 16))
    return torch.einsum("y x i j, ... x j -> ... y i", weight, x)


def _pair_coeffs(coeffs: torch.Tensor, subgroup: bool) -> torch.Tensor:
    # (out_c, in_c, 10) -> (5, 2*out_c, in_c): the grade-preserving weight of grade g stacked on
    # the Hodge-dual weight consuming grade g, one GEMM operand per grade.
    cp = coeffs.permute(2, 0, 1)
    if subgroup:
        return torch.cat((cp[:5], cp[5:]), dim=1)
    return cp.contiguous()


def _equi_linear_sparse(
    x: torch.Tensor, coeffs: torch.Tensor, *, config: PrimitivesConfig
) -> torch.Tensor:
    # Per-grade GEMMs on grade slices of x, in the component-major layout (16, batch, channels)
    # where each grade is a contiguous row block. Avoids the (10, 16, 16) basis.
    weights = _pair_coeffs(coeffs, config.subgroup)
    # (..., in_c, 16) -> (16, batch, in_c), batch dims merged before the copy so that every grade
    # operand is a view of it. Both transposes must be copies, or the views pin the whole input.
    xt = x.reshape(x.shape[:-2].numel(), *x.shape[-2:]).movedim(-1, 0).contiguous()

    def linear(a, b, w):
        return torch.nn.functional.linear(xt[a:b].flatten(0, 1), w).unflatten(0, (b - a, -1))

    z0 = linear(0, 1, weights[0])
    z1 = linear(1, 5, weights[1])
    z2 = linear(5, 11, weights[2])
    z3 = linear(11, 15, weights[3])
    z4 = linear(15, 16, weights[4])

    if config.subgroup:
        # Basis elements 5..9 are the Hodge dual, mapping grade g -> 4-g by a sign and a position
        # reversal. Both act on the position dim, which the GEMM treats as batch, so each grade
        # rides along with its dual partner in one GEMM on the _pair_coeffs weights:
        # z_g[..., :out_c] is the direct term, z_g[..., out_c:] the dual term.
        out_c = weights.shape[-2] // 2
        s = _compute_dual_sign(device=xt.device, dtype=z0.dtype)[:, None, None]

        def dual(z):
            return z[..., out_c:].flip(0)

        zs = (
            z0[..., :out_c] + s[0:1] * dual(z4),
            z1[..., :out_c] + s[1:5] * dual(z3),
            z2[..., :out_c] + s[5:11] * dual(z2),
            z3[..., :out_c] + s[11:15] * dual(z1),
            z4[..., :out_c] + s[15:16] * dual(z0),
        )
    else:
        zs = (z0, z1, z2, z3, z4)  # full Lorentz group: grade-preserving basis elements only

    yt = torch.cat(zs, dim=0)
    return yt.view(16, *x.shape[:-2], yt.shape[-1]).movedim(0, -1).contiguous()


def equi_linear(x: torch.Tensor, coeffs: torch.Tensor, *, config: PrimitivesConfig) -> torch.Tensor:
    """Pin-equivariant linear map ``f(x) = sum_{a,j} coeffs_a W^a_ij x_j``.

    The :math:`W^a` are 5 or 10 pre-defined, Lorentz-equivariant basis elements (10 for the
    connected subgroup, 5 for the full Lorentz group; selected via the ``subgroup`` option in
    :class:`~lgatr.primitives.config.PrimitivesConfig`).

    Parameters
    ----------
    x
        Input multivector of shape ``(..., in_channels, 16)``.
    coeffs
        Coefficients for the basis elements of shape ``(out_channels, in_channels, num_basis_elements)``,
        where ``num_basis_elements`` is 10 (connected subgroup) or 5 (full Lorentz group).
    config
        LGATr primitives configuration.

    Returns
    -------
    outputs
        Result of shape ``(..., out_channels, 16)``.
    """
    if config.sparse_linear:
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
