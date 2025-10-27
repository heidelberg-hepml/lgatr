"""Linear operations on multivectors, in particular linear basis maps."""

from functools import lru_cache
from pathlib import Path

import torch

from ..utils.einsum import cached_einsum, custom_einsum
from .config import gatr_config

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


@lru_cache
def _compute_pin_equi_linear_basis(
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
) -> torch.Tensor:
    """Constructs basis elements for Lorentz-equivariant linear maps between multivectors.

    This function is cached.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    basis : torch.Tensor
        Basis elements for equivariant linear maps with shape (NUM_PIN_LINEAR_BASIS_ELEMENTS, 16, 16),
        with NUM_PIN_LINEAR_BASIS_ELEMENTS=5 for the full Lorentz group (including the discrete
        operations of parity and time reversal) and 10 for the fully connected subgroup.
    """

    if device not in [DEFAULT_DEVICE, "cpu"] and dtype != DEFAULT_DTYPE:
        basis = _compute_pin_equi_linear_basis()
    else:
        file = (
            "linear_basis_subgroup.pt"
            if gatr_config.use_fully_connected_subgroup
            else "linear_basis_full.pt"
        )
        filename = Path(__file__).parent.resolve() / file
        basis = torch.load(filename).to(DEFAULT_DTYPE).to_dense()
    return basis.to(device=device, dtype=dtype)


@lru_cache
def _compute_reversal(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) -> torch.Tensor:
    """Constructs a matrix that computes multivector reversal.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    reversal_diag : torch.Tensor
        The diagonal of the reversal matrix with shape (16,), consisting of +1 and -1 entries.
    """
    reversal_flat = torch.ones(16, device=device, dtype=dtype)
    reversal_flat[5:15] = -1
    return reversal_flat


@lru_cache
def _compute_grade_involution(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) -> torch.Tensor:
    """Constructs a matrix that computes multivector grade involution.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    involution_diag : torch.Tensor
        The diagonal of the involution matrix with shape (16,), consisting of +1 and -1 entries.
    """
    involution_flat = torch.ones(16, device=device, dtype=dtype)
    involution_flat[1:5] = -1
    involution_flat[11:15] = -1
    return involution_flat


def equi_linear(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant linear map ``f(x) = sum_{a,j} coeffs_a W^a_ij x_j``.

    The W^a are seven pre-defined basis elements.

    Parameters
    ----------
    x : torch.Tensor
        Input multivector with shape (..., in_channels, 16).
        Batch dimensions must be broadcastable between x and coeffs.
    coeffs : torch.Tensor
        Coefficients for the basis elements with shape (out_channels, in_channels, 10).
        Batch dimensions must be broadcastable between x and coeffs.

    Returns
    -------
    outputs : torch.Tensor
        Result with shape (..., 16).
        Batch dimensions are result of broadcasting between x and coeffs.
    """
    basis = _compute_pin_equi_linear_basis(device=x.device, dtype=x.dtype)
    return custom_einsum("y x a, a i j, ... x j -> ... y i", coeffs, basis, x, path=[0, 1, 0, 1])


def grade_project(x: torch.Tensor) -> torch.Tensor:
    """Projects an input tensor to the individual grades.

    The return value is a single tensor with a new grade dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input multivector with shape (..., 16).

    Returns
    -------
    outputs : torch.Tensor
        Output multivector with shape (..., 5, 16).
        The second-to-last dimension indexes the grades.
    """

    # Select kernel on correct device
    basis = _compute_pin_equi_linear_basis(
        device=x.device,
        dtype=x.dtype,
    )

    # First five basis elements are grade projections
    basis = basis[:5]

    # Project to grades
    projections = cached_einsum("g i j, ... j -> ... g i", basis, x)

    return projections


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Computes the reversal of a multivector.

    The reversal has the same scalar, vector, and pseudoscalar components, but flips sign in the
    bivector and trivector components.

    Parameters
    ----------
    x : torch.Tensor
        Input multivector with shape (..., 16).

    Returns
    -------
    outputs : torch.Tensor
        Output multivector with shape (..., 16).
    """
    return _compute_reversal(device=x.device, dtype=x.dtype) * x


def grade_involute(x: torch.Tensor) -> torch.Tensor:
    """Computes the grade involution of a multivector.

    The reversal has the same scalar, bivector, and pseudoscalar components, but flips sign in the
    vector and trivector components.

    Parameters
    ----------
    x : torch.Tensor
        Input multivector with shape (..., 16).

    Returns
    -------
    outputs : torch.Tensor
        Output multivector with shape (..., 16).
    """

    return _compute_grade_involution(device=x.device, dtype=x.dtype) * x
