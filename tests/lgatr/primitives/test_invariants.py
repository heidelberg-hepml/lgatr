"""Unit tests of invariant primitives."""

from lgatr.primitives import abs_squared_norm, inner_product
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_invariance


def test_inner_product_invariance() -> None:
    # inner_product is Pin-invariant in both arguments.
    check_pin_invariance(inner_product, 2, batch_dims=[BATCH_DIMS] * 2, spin=False, **TOLERANCES)


def test_abs_squared_norm_invariance() -> None:
    # abs_squared_norm is Pin-invariant.
    check_pin_invariance(abs_squared_norm, 1, batch_dims=BATCH_DIMS, spin=False, **TOLERANCES)
