import pytest
import torch

from lgatr.layers.layer_norm import EquiLayerNorm
from lgatr.primitives import abs_squared_norm
from tests.helpers import TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", [(7, 9)])
@pytest.mark.parametrize("num_scalars", [9])
def test_equi_layer_norm_layer_correctness(batch_dims: tuple[int, ...], num_scalars: int) -> None:
    # EquiLayerNorm rescales the multivector input to unit mean GA squared norm.
    inputs = torch.randn(*batch_dims, 16)
    scalars = torch.randn(*batch_dims, num_scalars)
    layer = EquiLayerNorm(epsilon=1e-9)
    normalized_inputs, _ = layer(inputs, scalars=scalars)
    dims = tuple(range(1, len(batch_dims) + 1))
    variance = torch.mean(abs_squared_norm(normalized_inputs), dim=dims)
    torch.testing.assert_close(variance, torch.ones_like(variance), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(7, 9)])
@pytest.mark.parametrize("num_scalars", [9])
def test_equi_layer_norm_layer_equivariance(batch_dims: tuple[int, ...], num_scalars: int) -> None:
    # EquiLayerNorm is Pin-equivariant.
    layer = EquiLayerNorm()
    scalars = torch.randn(*batch_dims, num_scalars)
    check_pin_equivariance(
        layer, 1, batch_dims=batch_dims, fn_kwargs=dict(scalars=scalars), **TOLERANCES
    )


def test_equi_layer_norm_none_scalars() -> None:
    # EquiLayerNorm propagates scalars=None.
    layer = EquiLayerNorm()
    inputs = torch.randn(4, 5, 16)
    outputs_mv, outputs_s = layer(inputs, scalars=None)
    assert outputs_mv.shape == inputs.shape
    assert outputs_s is None
