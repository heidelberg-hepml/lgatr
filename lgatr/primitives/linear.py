"""Linear operations on multivectors, in particular linear basis maps."""

import math
from functools import lru_cache
from pathlib import Path

import torch

from .config import gatr_config

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


def _load_basis_tensor(filename: str) -> torch.Tensor:
    path = Path(__file__).parent.resolve() / filename
    return torch.load(path).to(DEFAULT_DTYPE).to_dense()


# Loaded once at import time. Subsequent device/dtype views are pure tensor ops, so the
# `_compute_pin_equi_linear_basis` helper body is traceable by ``torch.compile`` without
# touching the filesystem.
_BASIS_SUBGROUP = _load_basis_tensor("linear_basis_subgroup.pt")
_BASIS_FULL = _load_basis_tensor("linear_basis_full.pt")


@lru_cache
def _compute_pin_equi_linear_basis(
    use_fully_connected_subgroup: bool = True,
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Construct basis elements for Lorentz-equivariant linear maps between multivectors.

    Cached via :func:`functools.lru_cache`.

    Parameters
    ----------
    use_fully_connected_subgroup
        If True, model is only equivariant with respect to
        the fully connected subgroup of the Lorentz group,
        the proper orthochronous Lorentz group :math:`SO^+(1,3)`,
        which does not include parity and time reversal.
        This setting affects how the EquiLinear maps work:
        For :math:`SO^+(1,3)`, they include transitions scalars/pseudoscalars
        vectors/axialvectors and among bivectors, effectively
        treating the pseudoscalar/axialvector representations
        like another scalar/vector.
        Defaults to True, because parity-odd representations
        are usually not important in high-energy physics simulations.

    Returns
    -------
    basis
        Basis elements of shape ``(num_basis_elements, 16, 16)``, with ``num_basis_elements = 5``
        for the full Lorentz group (including parity and time reversal) and ``10`` for the fully
        connected subgroup.
    """
    src = _BASIS_SUBGROUP if use_fully_connected_subgroup else _BASIS_FULL
    return src.to(device=device, dtype=dtype)


@lru_cache
def _compute_grade_projection_mask(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Construct the grade-projection mask used by :func:`grade_project`.

    The mask is independent of ``gatr_config.use_fully_connected_subgroup``: the first five
    elements of both basis files are pure-diagonal grade projectors and agree with each other
    on those rows.

    Returns
    -------
    mask
        Tensor of shape ``(5, 16)`` with ones on positions belonging to grade ``g`` of row ``g``,
        zeros elsewhere.
    """
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
    """Construct the diagonal of the multivector reversal matrix.

    Returns
    -------
    reversal_diag
        Diagonal of shape ``(16,)``, consisting of +1 and -1 entries.
    """
    reversal_flat = torch.ones(16, device=device, dtype=dtype)
    reversal_flat[5:15] = -1
    return reversal_flat


@lru_cache
def _compute_grade_involution(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Construct the diagonal of the multivector grade-involution matrix.

    Returns
    -------
    involution_diag
        Diagonal of shape ``(16,)``, consisting of +1 and -1 entries.
    """
    involution_flat = torch.ones(16, device=device, dtype=dtype)
    involution_flat[1:5] = -1
    involution_flat[11:15] = -1
    return involution_flat


def equi_linear(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant linear map ``f(x) = sum_{a,j} coeffs_a W^a_ij x_j``.

    The :math:`W^a` are 5 or 10 pre-defined basis elements (see :func:`_compute_pin_equi_linear_basis`).

    Parameters
    ----------
    x
        Input multivector of shape ``(..., in_channels, 16)``.
        Batch dimensions must be broadcastable between ``x`` and ``coeffs``.
    coeffs
        Coefficients for the basis elements of shape ``(out_channels, in_channels, num_basis_elements)``,
        where ``num_basis_elements`` is 10 (fully connected subgroup) or 5 (full Lorentz group).
        Batch dimensions must be broadcastable between ``x`` and ``coeffs``.

    Returns
    -------
    outputs
        Result of shape ``(..., out_channels, 16)``.
        Batch dimensions are the broadcast of ``x`` and ``coeffs``.
    """
    basis = _compute_pin_equi_linear_basis(
        gatr_config.use_fully_connected_subgroup, device=x.device, dtype=x.dtype
    )
    # Same as ``torch.einsum("y x a, a i j, ... x j -> ... y i", coeffs, basis, x)`` but
    # with the contraction path pinned: opt_einsum's default picks an order that is 10-50x
    # slower on CPU for these shapes; ``path=[0, 1, 0, 1]`` precomputes the effective
    # ``(out_c, in_c, 16, 16)`` weight first, then contracts it with ``x``.
    return torch._VF.einsum(
        "y x a, a i j, ... x j -> ... y i", (coeffs, basis, x), path=[0, 1, 0, 1]
    )


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
    the bivector and trivector components.

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
    sign of the vector and trivector components.

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
