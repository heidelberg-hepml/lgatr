"""Unit tests of normalization primitives."""

import pytest
import torch

from lgatr.primitives import abs_squared_norm, equi_layer_norm
from tests.helpers import TOLERANCES, check_pin_equivariance

BATCH_DIMS = (7, 9)


@pytest.mark.parametrize("scale", [0.1, 1.0, 50.0])
def test_equi_layer_norm_correctness(scale: float) -> None:
    # With a negligible epsilon, equi_layer_norm rescales inputs to unit mean squared GA norm
    # regardless of the input scale.
    inputs = scale * torch.randn(*BATCH_DIMS, 16)
    normalized_inputs = equi_layer_norm(inputs, gain=1.0, epsilon=1e-9)
    variance = torch.mean(abs_squared_norm(normalized_inputs))
    torch.testing.assert_close(variance, torch.ones_like(variance), **TOLERANCES)


def test_equi_layer_norm_default_epsilon() -> None:
    # The shipped epsilon=0.01 acts as a floor on the squared norm, not as a bias: ordinary inputs
    # still come out at unit norm, while tiny inputs are scaled by 1/epsilon rather than blown up.
    inputs = torch.randn(*BATCH_DIMS, 16)
    variance = torch.mean(abs_squared_norm(equi_layer_norm(inputs, gain=1.0)))
    torch.testing.assert_close(variance, torch.ones_like(variance), **TOLERANCES)

    tiny = 1e-3 * inputs
    variance = torch.mean(abs_squared_norm(equi_layer_norm(tiny, gain=1.0)))
    torch.testing.assert_close(variance, torch.mean(abs_squared_norm(tiny)) / 0.01, **TOLERANCES)


def test_equi_layer_norm_equivariance() -> None:
    # equi_layer_norm is Pin-equivariant.
    check_pin_equivariance(equi_layer_norm, 1, batch_dims=BATCH_DIMS, spin=False, **TOLERANCES)
