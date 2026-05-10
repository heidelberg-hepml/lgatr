import pytest
import torch

from lgatr.layers import CrossAttentionConfig, MLPConfig, SelfAttentionConfig
from lgatr.nets import ConditionalLGATr
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

S_CHANNELS = [(3, 5), (2, 2), (0, 0)]
BATCH_DIMS = [b[:-1] for b in BATCH_DIMS]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_cond", [(2, 2), (2, 9)])
@pytest.mark.parametrize("in_mv_channels,mv_channels_cond", [(6, 6), (7, 11)])
@pytest.mark.parametrize("num_blocks,num_heads", [(1, 4)])
@pytest.mark.parametrize("in_s_channels,s_channels_cond", S_CHANNELS)
@pytest.mark.parametrize("hidden_mv_channels,hidden_s_channels", [(9, 4)])
@pytest.mark.parametrize("out_mv_channels,out_s_channels", [(8, 5)])
@pytest.mark.parametrize("dropout_prob", [None])
@pytest.mark.parametrize("multi_query_attention", [False, True])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
@pytest.mark.parametrize("subgroup", [True, False])
def test_conditional_gatr_shape(
    batch_dims: list[int],
    num_items: int,
    num_items_cond: int,
    in_mv_channels: int,
    mv_channels_cond: int,
    hidden_mv_channels: int,
    out_mv_channels: int,
    num_heads: int,
    num_blocks: int,
    in_s_channels: int,
    s_channels_cond: int,
    hidden_s_channels: int,
    out_s_channels: int,
    multi_query_attention: bool,
    dropout_prob: float | None,
    checkpoint_blocks: bool,
    subgroup: bool,
) -> None:
    # ConditionalLGATr's outputs match the expected shapes across all combinations.
    inputs = torch.randn(*batch_dims, num_items, in_mv_channels, 16)
    scalars = torch.randn(*batch_dims, num_items, in_s_channels) if in_s_channels else None
    mv_cond = torch.randn(*batch_dims, num_items_cond, mv_channels_cond, 16)
    s_cond = torch.randn(*batch_dims, num_items_cond, s_channels_cond) if s_channels_cond else None

    try:
        net = ConditionalLGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            mv_channels_cond=mv_channels_cond,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            s_channels_cond=s_channels_cond,
            attention=dict(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            crossattention=dict(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=dict(),
            primitives=PrimitivesConfig(subgroup=subgroup),
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
        multivectors_cond=mv_cond,
        scalars_cond=s_cond,
    )

    assert outputs.shape == (*batch_dims, num_items, out_mv_channels, 16)
    if out_s_channels:
        assert output_scalars.shape == (*batch_dims, num_items, out_s_channels)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_cond", [(2, 2), (2, 9)])
@pytest.mark.parametrize("in_mv_channels,mv_channels_cond", [(6, 6), (7, 11)])
@pytest.mark.parametrize("num_blocks,num_heads", [(1, 4)])
@pytest.mark.parametrize("in_s_channels,s_channels_cond", S_CHANNELS)
@pytest.mark.parametrize("hidden_mv_channels,hidden_s_channels", [(9, 4)])
@pytest.mark.parametrize("out_mv_channels,out_s_channels", [(8, 5)])
@pytest.mark.parametrize("dropout_prob", [None])
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_conditional_gatr_equivariance(
    batch_dims: list[int],
    num_items: int,
    num_items_cond: int,
    in_mv_channels: int,
    mv_channels_cond: int,
    hidden_mv_channels: int,
    out_mv_channels: int,
    num_heads: int,
    num_blocks: int,
    in_s_channels: int,
    s_channels_cond: int,
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
            mv_channels_cond=mv_channels_cond,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
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
            num_blocks=num_blocks,
            dropout_prob=dropout_prob,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    scalars = torch.randn(*batch_dims, num_items, in_s_channels)
    scalars_cond = torch.randn(*batch_dims, num_items_cond, s_channels_cond)

    data_dims = [
        tuple(list(batch_dims) + [num_items, in_mv_channels]),
        tuple(list(batch_dims) + [num_items_cond, mv_channels_cond]),
    ]
    check_pin_equivariance(
        net,
        2,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars, scalars_cond=scalars_cond),
        **MILD_TOLERANCES,
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_cond", [(2, 9)])
@pytest.mark.parametrize("in_mv_channels,mv_channels_cond", [(7, 11)])
@pytest.mark.parametrize("num_blocks,num_heads", [(1, 4)])
@pytest.mark.parametrize("in_s_channels,s_channels_cond", [(3, 5)])
@pytest.mark.parametrize("hidden_mv_channels,hidden_s_channels", [(9, 4)])
@pytest.mark.parametrize("out_mv_channels,out_s_channels", [(8, 5)])
def test_conditional_gatr_equivariance_compiled(
    batch_dims: list[int],
    num_items: int,
    num_items_cond: int,
    in_mv_channels: int,
    mv_channels_cond: int,
    hidden_mv_channels: int,
    out_mv_channels: int,
    num_heads: int,
    num_blocks: int,
    in_s_channels: int,
    s_channels_cond: int,
    hidden_s_channels: int,
    out_s_channels: int,
    compile: bool = True,
) -> None:
    # torch.compile-wrapped ConditionalLGATr still preserves shapes and Pin-equivariance.
    net = ConditionalLGATr(
        in_mv_channels=in_mv_channels,
        out_mv_channels=out_mv_channels,
        hidden_mv_channels=hidden_mv_channels,
        mv_channels_cond=mv_channels_cond,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        s_channels_cond=s_channels_cond,
        attention=SelfAttentionConfig(num_heads=num_heads),
        crossattention=CrossAttentionConfig(num_heads=num_heads),
        mlp=MLPConfig(),
        num_blocks=num_blocks,
        compile=compile,
    )

    scalars = torch.randn(*batch_dims, num_items, in_s_channels)
    scalars_cond = torch.randn(*batch_dims, num_items_cond, s_channels_cond)

    data_dims = [
        tuple(list(batch_dims) + [num_items, in_mv_channels]),
        tuple(list(batch_dims) + [num_items_cond, mv_channels_cond]),
    ]
    check_pin_equivariance(
        net,
        2,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars, scalars_cond=scalars_cond),
        **MILD_TOLERANCES,
    )
