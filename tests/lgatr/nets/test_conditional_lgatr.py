import pytest
import torch

from lgatr.layers import CrossAttentionConfig, MLPConfig, SelfAttentionConfig
from lgatr.nets import ConditionalLGATr
from lgatr.primitives.config import gatr_config
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

S_CHANNELS = [(3, 5), (2, 2), (0, 0)]
BATCH_DIMS = [b[:-1] for b in BATCH_DIMS]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_condition", [(2, 2), (2, 9)])
@pytest.mark.parametrize("in_mv_channels,in_mv_channels_condition", [(6, 6), (7, 11)])
@pytest.mark.parametrize("num_blocks,num_heads", [(1, 4)])
@pytest.mark.parametrize("in_s_channels,in_s_channels_condition", S_CHANNELS)
@pytest.mark.parametrize("hidden_mv_channels,hidden_s_channels", [(9, 4)])
@pytest.mark.parametrize("out_mv_channels,out_s_channels", [(8, 5)])
@pytest.mark.parametrize("dropout_prob", [None])
@pytest.mark.parametrize("multi_query_attention", [False, True])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
@pytest.mark.parametrize("use_fully_connected_subgroup", [True, False])
def test_conditional_gatr_shape(
    batch_dims: list[int],
    num_items: int,
    num_items_condition: int,
    in_mv_channels: int,
    in_mv_channels_condition: int,
    hidden_mv_channels: int,
    out_mv_channels: int,
    num_heads: int,
    num_blocks: int,
    in_s_channels: int,
    in_s_channels_condition: int,
    hidden_s_channels: int,
    out_s_channels: int,
    multi_query_attention: bool,
    dropout_prob: float | None,
    checkpoint_blocks: bool,
    use_fully_connected_subgroup: bool,
) -> None:
    # ConditionalLGATr's outputs match the expected shapes across all combinations.
    gatr_config.use_fully_connected_subgroup = use_fully_connected_subgroup

    inputs = torch.randn(*batch_dims, num_items, in_mv_channels, 16)
    scalars = torch.randn(*batch_dims, num_items, in_s_channels) if in_s_channels else None
    condition_mv = torch.randn(*batch_dims, num_items_condition, in_mv_channels_condition, 16)
    condition_s = (
        torch.randn(*batch_dims, num_items_condition, in_s_channels_condition)
        if in_s_channels_condition
        else None
    )

    try:
        net = ConditionalLGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            condition_mv_channels=in_mv_channels_condition,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            condition_s_channels=in_s_channels_condition,
            attention=dict(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            crossattention=dict(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=dict(),
            num_blocks=num_blocks,
            dropout_prob=dropout_prob,
            checkpoint_blocks=checkpoint_blocks,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    outputs, output_scalars = net(
        inputs,
        scalars=scalars,
        multivectors_condition=condition_mv,
        scalars_condition=condition_s,
    )

    assert outputs.shape == (*batch_dims, num_items, out_mv_channels, 16)
    if out_s_channels:
        assert output_scalars.shape == (*batch_dims, num_items, out_s_channels)

    # restore defaults
    gatr_config.use_fully_connected_subgroup = True


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_condition", [(2, 2), (2, 9)])
@pytest.mark.parametrize("in_mv_channels,in_mv_channels_condition", [(6, 6), (7, 11)])
@pytest.mark.parametrize("num_blocks,num_heads", [(1, 4)])
@pytest.mark.parametrize("in_s_channels,in_s_channels_condition", S_CHANNELS)
@pytest.mark.parametrize("hidden_mv_channels,hidden_s_channels", [(9, 4)])
@pytest.mark.parametrize("out_mv_channels,out_s_channels", [(8, 5)])
@pytest.mark.parametrize("dropout_prob", [None])
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_conditional_gatr_equivariance(
    batch_dims: list[int],
    num_items: int,
    num_items_condition: int,
    in_mv_channels: int,
    in_mv_channels_condition: int,
    hidden_mv_channels: int,
    out_mv_channels: int,
    num_heads: int,
    num_blocks: int,
    in_s_channels: int,
    in_s_channels_condition: int,
    hidden_s_channels: int,
    out_s_channels: int,
    multi_query_attention: bool,
    dropout_prob: float | None,
) -> None:
    # ConditionalLGATr (full network) is Pin-equivariant in both inputs and condition.
    try:
        net = ConditionalLGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            condition_mv_channels=in_mv_channels_condition,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            condition_s_channels=in_s_channels_condition,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            crossattention=CrossAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=MLPConfig(),
            num_blocks=num_blocks,
            dropout_prob=dropout_prob,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    scalars = torch.randn(*batch_dims, num_items, in_s_channels)
    scalars_condition = torch.randn(*batch_dims, num_items_condition, in_s_channels_condition)

    data_dims = [
        tuple(list(batch_dims) + [num_items, in_mv_channels]),
        tuple(list(batch_dims) + [num_items_condition, in_mv_channels_condition]),
    ]
    check_pin_equivariance(
        net,
        2,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars, scalars_condition=scalars_condition),
        **MILD_TOLERANCES,
    )
