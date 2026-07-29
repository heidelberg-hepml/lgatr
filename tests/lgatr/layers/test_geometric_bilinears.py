import pytest
import torch

from lgatr.layers.mlp.geometric_bilinears import GeometricBilinear
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance

IN_MV_CHANNELS, OUT_MV_CHANNELS = 8, 10
IN_S_CHANNELS, OUT_S_CHANNELS = 3, 5


def _layer(bivector: bool) -> GeometricBilinear:
    """A GeometricBilinear with the standard test channel counts."""
    return GeometricBilinear(
        IN_MV_CHANNELS,
        OUT_MV_CHANNELS,
        primitives=PrimitivesConfig(bivector=bivector),
        in_s_channels=IN_S_CHANNELS,
        out_s_channels=OUT_S_CHANNELS,
    )


@pytest.mark.parametrize("bivector", [True, False])
def test_geometric_bilinears_equivariance(bivector: bool) -> None:
    # GeometricBilinear is Spin-equivariant, with and without bivector outputs.
    layer = _layer(bivector)
    data_dims = (*BATCH_DIMS, IN_MV_CHANNELS)
    scalars = torch.randn(*BATCH_DIMS, IN_S_CHANNELS)

    check_pin_equivariance(
        layer, 1, fn_kwargs=dict(scalars=scalars), batch_dims=data_dims, **TOLERANCES
    )


def test_geometric_bilinears_bivector_toggle() -> None:
    # PrimitivesConfig(bivector=False) zeroes the bivector components of the geometric product, so
    # it must change the output of an otherwise identical layer.
    layer = _layer(bivector=True)
    zeroed = _layer(bivector=False)
    zeroed.load_state_dict(layer.state_dict())

    multivectors = torch.randn(*BATCH_DIMS, IN_MV_CHANNELS, 16)
    scalars = torch.randn(*BATCH_DIMS, IN_S_CHANNELS)
    outputs, _ = layer(multivectors, scalars=scalars)
    outputs_zeroed, _ = zeroed(multivectors, scalars=scalars)

    assert not torch.allclose(outputs, outputs_zeroed, **TOLERANCES)


def test_geometric_bilinears_rejects_none_scalars() -> None:
    # A GeometricBilinear built with scalar channels rejects scalars=None at runtime.
    layer = GeometricBilinear(
        in_mv_channels=8,
        out_mv_channels=10,
        primitives=PrimitivesConfig(),
        in_s_channels=3,
    )
    with pytest.raises(ValueError):
        layer(torch.randn(4, 8, 16), scalars=None)
