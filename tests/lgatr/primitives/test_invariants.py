"""Unit tests of invariant primitives."""

import pytest

from lgatr.primitives import (
    abs_squared_norm,
    inner_product,
)
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_invariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_inner_product_invariance(batch_dims: list[int]) -> None:
    # inner_product is Pin-invariant in both arguments.
    check_pin_invariance(inner_product, 2, batch_dims=[batch_dims] * 2, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_abs_squared_norm_invariance(batch_dims: list[int]) -> None:
    # abs_squared_norm is Pin-invariant.
    check_pin_invariance(abs_squared_norm, 1, batch_dims=batch_dims, **TOLERANCES)
