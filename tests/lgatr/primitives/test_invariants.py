"""Unit tests of invariant primitives."""

import torch

from lgatr.interface import to_lightcone_mv
from lgatr.primitives import abs_squared_norm, inner_product
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import (
    BATCH_DIMS,
    STRICT_TOLERANCES,
    TOLERANCES,
    check_pin_invariance,
    random_lightcone_frame,
)

CONFIG = PrimitivesConfig()


def test_inner_product_invariance() -> None:
    # inner_product is Pin-invariant in both arguments.
    check_pin_invariance(
        inner_product,
        2,
        batch_dims=[BATCH_DIMS] * 2,
        fn_kwargs=dict(config=CONFIG),
        spin=False,
        **TOLERANCES,
    )


def test_abs_squared_norm_invariance() -> None:
    # abs_squared_norm is Pin-invariant.
    check_pin_invariance(
        abs_squared_norm,
        1,
        batch_dims=BATCH_DIMS,
        fn_kwargs=dict(config=CONFIG),
        spin=False,
        **TOLERANCES,
    )


def test_invariants_lightcone_match_cartesian() -> None:
    # The light-cone coordinates are an exact change of basis, so both invariants are unchanged.
    x = torch.randn(*BATCH_DIMS, 16, dtype=torch.float64)
    y = torch.randn(*BATCH_DIMS, 16, dtype=torch.float64)
    frame = random_lightcone_frame(x)
    x_lc, y_lc = to_lightcone_mv(x, frame), to_lightcone_mv(y, frame)
    lightcone = PrimitivesConfig(lightcone=True)

    torch.testing.assert_close(
        inner_product(x_lc, y_lc, config=lightcone),
        inner_product(x, y, config=CONFIG),
        **STRICT_TOLERANCES,
    )
    torch.testing.assert_close(
        abs_squared_norm(x_lc, config=lightcone),
        abs_squared_norm(x, config=CONFIG),
        **STRICT_TOLERANCES,
    )
