"""Unit tests of linear primitives."""

import pytest
import torch

from lgatr.primitives.config import PrimitivesConfig
from lgatr.primitives.linear import equi_linear, grade_involute, grade_project, reverse
from tests.helpers import (
    BATCH_DIMS,
    TOLERANCES,
    check_against_clifford,
    check_pin_equivariance,
)


def test_reverse_correctness() -> None:
    # reverse matches the clifford-library reference for multivector reversal.
    check_against_clifford(reverse, lambda x: ~x, BATCH_DIMS, **TOLERANCES)


def test_grade_involution_correctness() -> None:
    # grade_involute matches the clifford-library reference for grade involution.
    check_against_clifford(grade_involute, lambda x: x.gradeInvol(), BATCH_DIMS, **TOLERANCES)


def test_grade_project_equivariance() -> None:
    # grade_project is Pin-equivariant: the sandwich product preserves grades.
    check_pin_equivariance(grade_project, 1, batch_dims=BATCH_DIMS, spin=False, **TOLERANCES)


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
    # equi_linear is Spin-equivariant for several input/coeff broadcasting shapes.
    config = PrimitivesConfig()
    fn_kwargs = dict(
        coeffs=torch.randn(*coeff_batch_dims, config.num_pin_linear_basis_elements),
        config=config,
    )
    check_pin_equivariance(
        equi_linear, 1, fn_kwargs=fn_kwargs, batch_dims=input_batch_dims, **TOLERANCES
    )


@pytest.mark.parametrize("subgroup", [True, False])
def test_equi_linear_sparse_dense_equivalence(subgroup: bool) -> None:
    # The sparse path agrees with the dense path within TOLERANCES on shared inputs.
    config_dense = PrimitivesConfig(subgroup=subgroup, sparse_linear=False)
    config_sparse = PrimitivesConfig(subgroup=subgroup, sparse_linear=True)
    in_c, out_c = 4, 7
    x = torch.randn(*BATCH_DIMS, in_c, 16)
    coeffs = torch.randn(out_c, in_c, config_dense.num_pin_linear_basis_elements)
    out_dense = equi_linear(x, coeffs, config=config_dense)
    out_sparse = equi_linear(x, coeffs, config=config_sparse)
    torch.testing.assert_close(out_sparse, out_dense, **TOLERANCES)


@pytest.mark.parametrize("subgroup,spin", [(True, True), (False, False)])
def test_equi_linear_sparse_equivariance(subgroup: bool, spin: bool) -> None:
    # The sparse path is equivariant under the group it is built for: Spin for the subgroup, the
    # full Pin group (including reflections) otherwise. check_pin_equivariance builds inputs of
    # shape (*batch_dims, 16), so the trailing batch dim plays the role of in_channels.
    config = PrimitivesConfig(subgroup=subgroup, sparse_linear=True)
    in_c = BATCH_DIMS[-1]
    fn_kwargs = dict(
        coeffs=torch.randn(7, in_c, config.num_pin_linear_basis_elements),
        config=config,
    )
    check_pin_equivariance(
        equi_linear, 1, fn_kwargs=fn_kwargs, batch_dims=BATCH_DIMS, spin=spin, **TOLERANCES
    )
