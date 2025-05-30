import pytest
import torch

from lgatr.layers.attention.config import SelfAttentionConfig
from lgatr.layers.mlp.config import MLPConfig
from lgatr.nets import LGATr
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

S_CHANNELS = [(None, None, 7), (4, 5, 6)]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "num_items,in_mv_channels,out_mv_channels,hidden_mv_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_heads,num_blocks", [(4, 1)])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", S_CHANNELS)
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.3])
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_lgatr_shape(
    batch_dims,
    num_items,
    in_mv_channels,
    out_mv_channels,
    hidden_mv_channels,
    num_blocks,
    num_heads,
    in_s_channels,
    out_s_channels,
    hidden_s_channels,
    multi_query_attention,
    dropout_prob,
):
    """Tests the output shape of LGATr."""
    inputs = torch.randn(*batch_dims, num_items, in_mv_channels, 16)
    scalars = (
        None
        if in_s_channels is None
        else torch.randn(*batch_dims, num_items, in_s_channels)
    )

    try:
        net = LGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            num_blocks=num_blocks,
            mlp=MLPConfig(),
            dropout_prob=dropout_prob,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    outputs, output_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (*batch_dims, num_items, out_mv_channels, 16)
    if in_s_channels is not None:
        assert output_scalars.shape == (*batch_dims, num_items, out_s_channels)


@pytest.mark.parametrize("batch_dims", [(64,)])
@pytest.mark.parametrize(
    "num_items,in_mv_channels,out_mv_channels,hidden_mv_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_heads,num_blocks", [(4, 1)])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", S_CHANNELS)
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_lgatr_equivariance(
    batch_dims,
    num_items,
    in_mv_channels,
    out_mv_channels,
    hidden_mv_channels,
    num_blocks,
    num_heads,
    in_s_channels,
    out_s_channels,
    hidden_s_channels,
    multi_query_attention,
):
    """Tests LGATr for equivariance."""
    try:
        net = LGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            num_blocks=num_blocks,
            mlp=MLPConfig(),
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    scalars = (
        None
        if in_s_channels is None
        else torch.randn(*batch_dims, num_items, in_s_channels)
    )
    data_dims = tuple(list(batch_dims) + [num_items, in_mv_channels])
    check_pin_equivariance(
        net, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **MILD_TOLERANCES
    )
