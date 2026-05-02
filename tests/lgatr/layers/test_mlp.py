import pytest
import torch

from lgatr.layers import GeoMLP
from lgatr.layers.mlp.config import MLPConfig
from lgatr.primitives.config import gatr_config
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance

_CHANNELS = [(5, 12), (4, 10), (4, 0)]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("mv_channels,s_channels", _CHANNELS)
@pytest.mark.parametrize("use_geometric_product", [True, False])
def test_geo_mlp_shape(
    batch_dims: list[int],
    mv_channels: int,
    s_channels: int,
    activation: str,
    use_geometric_product: bool,
) -> None:
    # GeoMLP outputs match the input multivector and scalar shapes.
    gatr_config.use_geometric_product = use_geometric_product

    inputs = torch.randn(*batch_dims, mv_channels, 16)
    scalars = torch.randn(*batch_dims, s_channels) if s_channels else None

    try:
        net = GeoMLP(
            MLPConfig(mv_channels=mv_channels, s_channels=s_channels, activation=activation)
        )
    except NotImplementedError:
        return  # "GeoMLP not implemented for this configuration"
    outputs, outputs_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (*batch_dims, mv_channels, 16)
    if s_channels:
        assert outputs_scalars.shape == (*batch_dims, s_channels)


@pytest.mark.parametrize("batch_dims", [[100]])
@pytest.mark.parametrize("activation", ["gelu"])
@pytest.mark.parametrize("mv_channels,s_channels", _CHANNELS)
def test_geo_mlp_equivariance(
    batch_dims: list[int],
    mv_channels: int,
    s_channels: int,
    activation: str,
) -> None:
    # GeoMLP is Spin-equivariant (Pin tested via the lower-level primitives).
    net = GeoMLP(MLPConfig(mv_channels=mv_channels, s_channels=s_channels, activation=activation))
    data_dims = tuple(list(batch_dims) + [mv_channels])
    scalars = torch.randn(*batch_dims, s_channels) if s_channels else None

    # Because of the fixed reference MV, we only test Spin equivariance
    check_pin_equivariance(
        net,
        1,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars),
        spin=True,
        **TOLERANCES,
    )
