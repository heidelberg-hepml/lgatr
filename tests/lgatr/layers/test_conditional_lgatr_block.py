import pytest
import torch

from lgatr.layers import (
    ConditionalLGATrBlock,
    CrossAttentionConfig,
    MLPConfig,
    SelfAttentionConfig,
)
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

S_CHANNELS = [(3, 5), (2, 2), (0, 0)]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_condition", [(2, 2), (2, 9)])
@pytest.mark.parametrize("mv_channels,mv_channels_condition", [(6, 6), (7, 11)])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("s_channels,s_channels_condition", S_CHANNELS)
@pytest.mark.parametrize("dropout_prob", [None])
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_conditional_gatr_block_shape(
    batch_dims,
    num_items,
    num_items_condition,
    mv_channels,
    mv_channels_condition,
    num_heads,
    s_channels,
    s_channels_condition,
    multi_query_attention,
    dropout_prob,
):
    """Tests the output shape of ConditionalLGATrBlock."""
    inputs = torch.randn(*batch_dims, num_items, mv_channels, 16)
    scalars = torch.randn(*batch_dims, num_items, s_channels) if s_channels else None
    condition_mv = torch.randn(*batch_dims, num_items_condition, mv_channels_condition, 16)
    condition_s = (
        torch.randn(*batch_dims, num_items_condition, s_channels_condition)
        if s_channels_condition
        else None
    )

    try:
        net = ConditionalLGATrBlock(
            mv_channels,
            s_channels=s_channels,
            condition_mv_channels=mv_channels_condition,
            condition_s_channels=s_channels_condition,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            crossattention=CrossAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=MLPConfig(),
            dropout_prob=dropout_prob,
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

    assert outputs.shape == (*batch_dims, num_items, mv_channels, 16)
    if s_channels:
        assert output_scalars.shape == (*batch_dims, num_items, s_channels)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_condition", [(2, 2), (2, 9)])
@pytest.mark.parametrize("mv_channels,mv_channels_condition", [(6, 6), (7, 11)])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("s_channels,s_channels_condition", S_CHANNELS)
@pytest.mark.parametrize("dropout_prob", [None])
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_conditional_gatr_block_equivariance(
    batch_dims,
    num_items,
    num_items_condition,
    mv_channels,
    mv_channels_condition,
    num_heads,
    s_channels,
    s_channels_condition,
    multi_query_attention,
    dropout_prob,
):
    """Tests ConditionalLGATrBlock for equivariance."""
    try:
        net = ConditionalLGATrBlock(
            mv_channels,
            s_channels=s_channels,
            condition_mv_channels=mv_channels_condition,
            condition_s_channels=s_channels_condition,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            crossattention=CrossAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=MLPConfig(),
            dropout_prob=dropout_prob,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    scalars = torch.randn(*batch_dims, num_items, s_channels)
    scalars_condition = torch.randn(*batch_dims, num_items_condition, s_channels_condition)

    data_dims = [
        tuple(list(batch_dims) + [num_items, mv_channels]),
        tuple(list(batch_dims) + [num_items_condition, mv_channels_condition]),
    ]
    check_pin_equivariance(
        net,
        2,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars, scalars_condition=scalars_condition),
        **MILD_TOLERANCES,
    )


def test_conditional_lgatr_block_none_scalars_at_runtime():
    """Tests ConditionalLGATrBlock accepts scalars=None and scalars_condition=None."""
    net = ConditionalLGATrBlock(
        mv_channels=4,
        s_channels=2,
        condition_mv_channels=4,
        condition_s_channels=3,
        attention=SelfAttentionConfig(num_heads=2),
        crossattention=CrossAttentionConfig(num_heads=2),
        mlp=MLPConfig(),
    )
    mv = torch.randn(3, 5, 4, 16)
    net(mv, multivectors_condition=mv, scalars=None, scalars_condition=None)
