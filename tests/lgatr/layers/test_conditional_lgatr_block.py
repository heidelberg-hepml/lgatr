import pytest
import torch

from lgatr.layers import (
    ConditionalLGATrBlock,
    CrossAttentionConfig,
    MLPConfig,
    SelfAttentionConfig,
)
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

S_CHANNELS = [(3, 5), (2, 2), (0, 0)]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_cond", [(2, 2), (2, 9)])
@pytest.mark.parametrize("mv_channels,mv_channels_cond", [(6, 6), (7, 11)])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("s_channels,s_channels_cond", S_CHANNELS)
@pytest.mark.parametrize("dropout_prob", [None])
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_conditional_gatr_block_shape(
    batch_dims: list[int],
    num_items: int,
    num_items_cond: int,
    mv_channels: int,
    mv_channels_cond: int,
    num_heads: int,
    s_channels: int,
    s_channels_cond: int,
    multi_query_attention: bool,
    dropout_prob: float | None,
) -> None:
    # ConditionalLGATrBlock outputs match the input shape (multivectors and scalars).
    inputs = torch.randn(*batch_dims, num_items, mv_channels, 16)
    scalars = torch.randn(*batch_dims, num_items, s_channels) if s_channels else None
    mv_cond = torch.randn(*batch_dims, num_items_cond, mv_channels_cond, 16)
    s_cond = torch.randn(*batch_dims, num_items_cond, s_channels_cond) if s_channels_cond else None

    try:
        net = ConditionalLGATrBlock(
            mv_channels,
            s_channels=s_channels,
            mv_channels_cond=mv_channels_cond,
            s_channels_cond=s_channels_cond,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            crossattention=CrossAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=MLPConfig(),
            primitives=PrimitivesConfig(),
            dropout_prob=dropout_prob,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    outputs, output_scalars = net(
        inputs,
        scalars=scalars,
        multivectors_cond=mv_cond,
        scalars_cond=s_cond,
    )

    assert outputs.shape == (*batch_dims, num_items, mv_channels, 16)
    if s_channels:
        assert output_scalars.shape == (*batch_dims, num_items, s_channels)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_cond", [(2, 2), (2, 9)])
@pytest.mark.parametrize("mv_channels,mv_channels_cond", [(6, 6), (7, 11)])
@pytest.mark.parametrize("num_heads,multi_query_attention", [(1, False), (4, True)])
@pytest.mark.parametrize("s_channels,s_channels_cond", S_CHANNELS)
@pytest.mark.parametrize("dropout_prob", [None])
def test_conditional_gatr_block_equivariance(
    batch_dims: list[int],
    num_items: int,
    num_items_cond: int,
    mv_channels: int,
    mv_channels_cond: int,
    num_heads: int,
    s_channels: int,
    s_channels_cond: int,
    multi_query_attention: bool,
    dropout_prob: float | None,
) -> None:
    # ConditionalLGATrBlock is Pin-equivariant in both query and condition multivector inputs.
    try:
        net = ConditionalLGATrBlock(
            mv_channels,
            s_channels=s_channels,
            mv_channels_cond=mv_channels_cond,
            s_channels_cond=s_channels_cond,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            crossattention=CrossAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=MLPConfig(),
            primitives=PrimitivesConfig(),
            dropout_prob=dropout_prob,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    scalars = torch.randn(*batch_dims, num_items, s_channels)
    scalars_cond = torch.randn(*batch_dims, num_items_cond, s_channels_cond)

    data_dims = [
        tuple(list(batch_dims) + [num_items, mv_channels]),
        tuple(list(batch_dims) + [num_items_cond, mv_channels_cond]),
    ]
    check_pin_equivariance(
        net,
        2,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars, scalars_cond=scalars_cond),
        **MILD_TOLERANCES,
    )


def test_conditional_lgatr_block_none_scalars_at_runtime() -> None:
    # ConditionalLGATrBlock accepts scalars=None and scalars_cond=None at runtime.
    net = ConditionalLGATrBlock(
        mv_channels=4,
        s_channels=2,
        mv_channels_cond=4,
        s_channels_cond=3,
        attention=SelfAttentionConfig(num_heads=2),
        crossattention=CrossAttentionConfig(num_heads=2),
        mlp=MLPConfig(),
        primitives=PrimitivesConfig(),
    )
    mv = torch.randn(3, 5, 4, 16)
    net(mv, multivectors_cond=mv, scalars=None, scalars_cond=None)
