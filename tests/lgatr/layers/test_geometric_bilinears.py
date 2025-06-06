import pytest
import torch

from lgatr.layers.mlp.geometric_bilinears import GeometricBilinear
from lgatr.primitives.config import gatr_config
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_mv_channels", [8])
@pytest.mark.parametrize("out_mv_channels", [10])
@pytest.mark.parametrize("in_s_channels", [3])
@pytest.mark.parametrize("out_s_channels", [5])
@pytest.mark.parametrize("use_bivector", [True, False])
def test_geometric_bilinears_equivariance(
    batch_dims,
    in_mv_channels,
    out_mv_channels,
    in_s_channels,
    out_s_channels,
    use_bivector,
):
    """Tests GeometricBilinear() for equivariance."""
    gatr_config.use_bivector = use_bivector

    layer = GeometricBilinear(
        in_mv_channels,
        out_mv_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
    )
    data_dims = tuple(list(batch_dims) + [in_mv_channels])
    scalars = torch.randn(*batch_dims, in_s_channels)

    check_pin_equivariance(
        layer,
        1,
        fn_kwargs=dict(scalars=scalars),
        batch_dims=data_dims,
        **TOLERANCES,
    )
