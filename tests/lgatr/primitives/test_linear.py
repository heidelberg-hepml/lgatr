"""Unit tests of linear primitives."""

import pytest
import torch

from lgatr.primitives.config import gatr_config
from lgatr.primitives.linear import equi_linear, grade_involute, grade_project, reverse
from tests.helpers import (
    BATCH_DIMS,
    TOLERANCES,
    check_consistence_with_grade_involution,
    check_consistence_with_reversal,
    check_pin_equivariance,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_reverse_correctness(batch_dims: list[int]) -> None:
    # reverse matches the clifford-library reference for multivector reversal.
    check_consistence_with_reversal(reverse, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_grade_involution_correctness(batch_dims: list[int]) -> None:
    # grade_involute matches the clifford-library reference for grade involution.
    check_consistence_with_grade_involution(grade_involute, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_identity_equivariance(batch_dims: list[int]) -> None:
    # The identity is trivially equivariant (smoke-tests the equivariance harness itself).
    check_pin_equivariance(lambda x: x, 1, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_grade_project_equivariance(batch_dims: list[int]) -> None:
    # grade_project is Pin-equivariant.
    check_pin_equivariance(grade_project, 1, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize(
    "input_batch_dims,coeff_batch_dims",
    [
        ((7,), (5, 7)),
        ((3, 7), (5, 7)),
        ((2, 3, 7), (5, 7)),
    ],
)
def test_linear_equivariance(
    input_batch_dims: tuple[int, ...], coeff_batch_dims: tuple[int, ...]
) -> None:
    # equi_linear is Pin-equivariant for several input/coeff broadcasting shapes.
    fn_kwargs = dict(
        coeffs=torch.randn(*coeff_batch_dims, gatr_config.num_pin_linear_basis_elements),
    )
    check_pin_equivariance(
        equi_linear, 1, fn_kwargs=fn_kwargs, batch_dims=input_batch_dims, **TOLERANCES
    )
