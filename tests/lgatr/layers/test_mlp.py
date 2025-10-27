import pytest
import torch

from lgatr.layers import GeoMLP
from lgatr.layers.mlp.config import MLPConfig
from lgatr.primitives.config import gatr_config
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance

_CHANNELS = [((5), (12)), ((4), (10)), ((4), None)]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("mv_channels,s_channels", _CHANNELS)
@pytest.mark.parametrize("use_geometric_product", [True, False])
def test_geo_mlp_shape(batch_dims, mv_channels, s_channels, activation, use_geometric_product):
    """Tests the output shape of GeoMLP()."""
    gatr_config.use_geometric_product = use_geometric_product

    inputs = torch.randn(*batch_dims, mv_channels, 16)
    scalars = None if s_channels is None else torch.randn(*batch_dims, s_channels)

    try:
        net = GeoMLP(
            MLPConfig(mv_channels=mv_channels, s_channels=s_channels, activation=activation)
        )
    except NotImplementedError:
        return  # "GeoMLP not implemented for this configuration"
    outputs, outputs_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (*batch_dims, mv_channels, 16)
    if s_channels is not None:
        assert outputs_scalars.shape == (*batch_dims, s_channels)


@pytest.mark.parametrize("batch_dims", [[100]])
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("mv_channels,s_channels", _CHANNELS)
def test_geo_mlp_equivariance(batch_dims, mv_channels, s_channels, activation):
    """Tests GeoMLP() for Pin equivariance."""
    net = GeoMLP(MLPConfig(mv_channels=mv_channels, s_channels=s_channels, activation=activation))
    data_dims = tuple(list(batch_dims) + [mv_channels])
    scalars = None if s_channels is None else torch.randn(*batch_dims, s_channels)

    # Because of the fixed reference MV, we only test Spin equivariance
    check_pin_equivariance(
        net,
        1,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars),
        spin=True,
        **TOLERANCES,
    )
