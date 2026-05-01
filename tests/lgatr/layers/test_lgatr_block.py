import pytest
import torch

from lgatr.layers import LGATrBlock
from lgatr.layers.attention.config import SelfAttentionConfig
from lgatr.layers.mlp.config import MLPConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,mv_channels", [(8, 6)])
@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("s_channels", [0, 2, 6])
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
    scalars = torch.randn(*batch_dims, num_items, s_channels) if s_channels else None

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
    if s_channels:
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

    scalars = torch.randn(*batch_dims, num_items, s_channels) if s_channels else None
    data_dims = tuple(list(batch_dims) + [num_items, mv_channels])
    check_pin_equivariance(
        net, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **MILD_TOLERANCES
    )


def test_lgatr_block_none_scalars_at_runtime():
    """Tests LGATrBlock accepts scalars=None at runtime."""
    net = LGATrBlock(
        mv_channels=4,
        s_channels=2,
        attention=SelfAttentionConfig(num_heads=2),
        mlp=MLPConfig(),
    )
    net(torch.randn(3, 5, 4, 16), scalars=None)
