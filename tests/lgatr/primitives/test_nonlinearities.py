"""Unit tests of nonlinearity primitives."""

import pytest
import torch

from lgatr.primitives import gated_relu, gated_sigmoid
from lgatr.primitives.nonlinearities import gated_gelu
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("fn", [gated_relu, gated_gelu, gated_sigmoid])
@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_gated_nonlin_equivariance(fn, batch_dims):
    """Tests an identity map for equivariance (testing the test)."""
    gates = torch.randn(*batch_dims, 1)
    check_pin_equivariance(
        fn, 1, fn_kwargs=dict(gates=gates), batch_dims=batch_dims, **TOLERANCES
    )
