import pytest
import torch

from lgatr.layers import LGATrBlock
from lgatr.layers.attention.config import SelfAttentionConfig
from lgatr.layers.mlp.config import MLPConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,mv_channels", [(8, 6)])
@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("s_channels", [2, 6])
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.3])
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_lgatr_block_shape(
    batch_dims,
    num_items,
    mv_channels,
    num_heads,
    s_channels,
    multi_query_attention,
    dropout_prob,
):
    """Tests the output shape of LGATrBlock."""
    inputs = torch.randn(*batch_dims, num_items, mv_channels, 16)
    scalars = (
        None if s_channels is None else torch.randn(*batch_dims, num_items, s_channels)
    )

    try:
        net = LGATrBlock(
            mv_channels,
            s_channels=s_channels,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=MLPConfig(),
            dropout_prob=dropout_prob,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    outputs, output_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (*batch_dims, num_items, mv_channels, 16)
    if s_channels is not None:
        assert output_scalars.shape == (*batch_dims, num_items, s_channels)


@pytest.mark.parametrize("batch_dims", [(64,)])
@pytest.mark.parametrize("num_items,mv_channels", [(8, 6)])
@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("s_channels", [2, 6])
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_lgatr_block_equivariance(
    batch_dims,
    num_items,
    mv_channels,
    num_heads,
    s_channels,
    multi_query_attention,
):
    """Tests LGATrBlock for equivariance."""
    try:
        net = LGATrBlock(
            mv_channels,
            s_channels=s_channels,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=MLPConfig(),
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    scalars = (
        None if s_channels is None else torch.randn(*batch_dims, num_items, s_channels)
    )
    data_dims = tuple(list(batch_dims) + [num_items, mv_channels])
    check_pin_equivariance(
        net, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **MILD_TOLERANCES
    )
