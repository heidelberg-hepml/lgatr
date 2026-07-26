import pytest
import torch

from lgatr.layers import GeoMLP
from lgatr.layers.mlp.config import MLPConfig
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance

# GeometricBilinear needs a scalar stream, so s_channels=0 requires geometric_product=False.
CHANNELS = [(5, 12, True), (4, 10, True), (4, 0, False), (5, 12, False)]


@pytest.mark.parametrize("mv_channels,s_channels,geometric_product", CHANNELS)
def test_geo_mlp_shape(mv_channels: int, s_channels: int, geometric_product: bool) -> None:
    # GeoMLP outputs match the input multivector and scalar shapes.
    net = GeoMLP(
        MLPConfig(mv_channels=mv_channels, s_channels=s_channels),
        primitives=PrimitivesConfig(geometric_product=geometric_product),
    )

    inputs = torch.randn(*BATCH_DIMS, mv_channels, 16)
    scalars = torch.randn(*BATCH_DIMS, s_channels) if s_channels else None
    outputs, outputs_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (*BATCH_DIMS, mv_channels, 16)
    if s_channels:
        assert outputs_scalars.shape == (*BATCH_DIMS, s_channels)


@pytest.mark.parametrize("mv_channels,s_channels,geometric_product", CHANNELS)
def test_geo_mlp_equivariance(mv_channels: int, s_channels: int, geometric_product: bool) -> None:
    # GeoMLP is Spin-equivariant. Only Spin, because of the fixed reference multivector.
    net = GeoMLP(
        MLPConfig(mv_channels=mv_channels, s_channels=s_channels),
        primitives=PrimitivesConfig(geometric_product=geometric_product),
    )
    data_dims = (100, mv_channels)
    scalars = torch.randn(100, s_channels) if s_channels else None

    check_pin_equivariance(
        net, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **TOLERANCES
    )
