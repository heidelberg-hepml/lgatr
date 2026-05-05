"""Unit tests of linear primitives."""

import pytest
import torch

from lgatr.primitives.config import PrimitivesConfig
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
    config = PrimitivesConfig()
    fn_kwargs = dict(
        coeffs=torch.randn(*coeff_batch_dims, config.num_pin_linear_basis_elements),
        config=config,
    )
    check_pin_equivariance(
        equi_linear, 1, fn_kwargs=fn_kwargs, batch_dims=input_batch_dims, **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("use_fully_connected_subgroup", [True, False])
def test_equi_linear_sparse_dense_equivalence(
    batch_dims: list[int], use_fully_connected_subgroup: bool
) -> None:
    # The sparse path agrees with the dense path within TOLERANCES on shared inputs.
    config_dense = PrimitivesConfig(
        use_fully_connected_subgroup=use_fully_connected_subgroup, sparse=False
    )
    config_sparse = PrimitivesConfig(
        use_fully_connected_subgroup=use_fully_connected_subgroup, sparse=True
    )
    in_c, out_c = 4, 7
    x = torch.randn(*batch_dims, in_c, 16)
    coeffs = torch.randn(out_c, in_c, config_dense.num_pin_linear_basis_elements)
    out_dense = equi_linear(x, coeffs, config=config_dense)
    out_sparse = equi_linear(x, coeffs, config=config_sparse)
    torch.testing.assert_close(out_sparse, out_dense, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("use_fully_connected_subgroup", [True, False])
def test_equi_linear_sparse_equivariance(
    batch_dims: list[int], use_fully_connected_subgroup: bool
) -> None:
    # The sparse path is Pin-equivariant. check_pin_equivariance builds inputs as
    # (*batch_dims, 16), so the trailing batch dim plays the role of in_channels.
    config = PrimitivesConfig(
        use_fully_connected_subgroup=use_fully_connected_subgroup, sparse=True
    )
    in_c = batch_dims[-1]
    fn_kwargs = dict(
        coeffs=torch.randn(7, in_c, config.num_pin_linear_basis_elements),
        config=config,
    )
    check_pin_equivariance(equi_linear, 1, fn_kwargs=fn_kwargs, batch_dims=batch_dims, **TOLERANCES)
