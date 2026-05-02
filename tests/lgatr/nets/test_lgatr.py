import pytest
import torch

from lgatr.layers.attention.config import SelfAttentionConfig
from lgatr.layers.mlp.config import MLPConfig
from lgatr.nets import LGATr
from lgatr.primitives.config import gatr_config
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

S_CHANNELS = [(0, 0, 7), (0, 0, 0), (4, 5, 6)]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "num_items,in_mv_channels,out_mv_channels,hidden_mv_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_heads,num_blocks", [(4, 1)])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", S_CHANNELS)
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.3])
@pytest.mark.parametrize("multi_query_attention", [False, True])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
@pytest.mark.parametrize("use_fully_connected_subgroup", [True, False])
def test_lgatr_shape(
    batch_dims: list[int],
    num_items: int,
    in_mv_channels: int,
    out_mv_channels: int,
    hidden_mv_channels: int,
    num_blocks: int,
    num_heads: int,
    in_s_channels: int,
    out_s_channels: int,
    hidden_s_channels: int,
    multi_query_attention: bool,
    dropout_prob: float | None,
    checkpoint_blocks: bool,
    use_fully_connected_subgroup: bool,
) -> None:
    # LGATr's outputs match the expected shapes for all combinations of channels and config.
    gatr_config.use_fully_connected_subgroup = use_fully_connected_subgroup

    inputs = torch.randn(*batch_dims, num_items, in_mv_channels, 16)
    scalars = torch.randn(*batch_dims, num_items, in_s_channels) if in_s_channels else None

    try:
        net = LGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            attention=dict(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            num_blocks=num_blocks,
            mlp=dict(),
            dropout_prob=dropout_prob,
            checkpoint_blocks=checkpoint_blocks,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    outputs, output_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (*batch_dims, num_items, out_mv_channels, 16)
    if out_s_channels:
        assert output_scalars.shape == (*batch_dims, num_items, out_s_channels)

    # restore defaults
    gatr_config.use_fully_connected_subgroup = True


@pytest.mark.parametrize("batch_dims", [(64,)])
@pytest.mark.parametrize(
    "num_items,in_mv_channels,out_mv_channels,hidden_mv_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_heads,num_blocks", [(4, 1)])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", S_CHANNELS)
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_lgatr_equivariance(
    batch_dims: tuple[int, ...],
    num_items: int,
    in_mv_channels: int,
    out_mv_channels: int,
    hidden_mv_channels: int,
    num_blocks: int,
    num_heads: int,
    in_s_channels: int,
    out_s_channels: int,
    hidden_s_channels: int,
    multi_query_attention: bool,
) -> None:
    # LGATr (full network) is Pin-equivariant.
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

    scalars = torch.randn(*batch_dims, num_items, in_s_channels) if in_s_channels else None
    data_dims = tuple(list(batch_dims) + [num_items, in_mv_channels])
    check_pin_equivariance(
        net, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **MILD_TOLERANCES
    )
