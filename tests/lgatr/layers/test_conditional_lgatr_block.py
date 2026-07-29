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

NUM_ITEMS, NUM_ITEMS_COND = 2, 9
MV_CHANNELS, MV_CHANNELS_COND = 7, 11

# GeometricBilinear needs a scalar stream, so s_channels=0 requires geometric_product=False.
S_CHANNELS = [(0, 0, False), (3, 5, True), (2, 2, True)]


def _block(
    s_channels: int,
    s_channels_cond: int,
    geometric_product: bool,
    num_heads: int,
    multi_query: bool,
    **kwargs,
) -> ConditionalLGATrBlock:
    """A ConditionalLGATrBlock with the standard test channel counts."""
    return ConditionalLGATrBlock(
        MV_CHANNELS,
        s_channels=s_channels,
        mv_channels_cond=MV_CHANNELS_COND,
        s_channels_cond=s_channels_cond,
        attention=SelfAttentionConfig(num_heads=num_heads, multi_query=multi_query),
        crossattention=CrossAttentionConfig(num_heads=num_heads, multi_query=multi_query),
        mlp=MLPConfig(),
        primitives=PrimitivesConfig(geometric_product=geometric_product),
        **kwargs,
    )


@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("s_channels,s_channels_cond,geometric_product", S_CHANNELS)
@pytest.mark.parametrize("multi_query", [False, True])
def test_conditional_gatr_block_shape(
    num_heads: int,
    s_channels: int,
    s_channels_cond: int,
    geometric_product: bool,
    multi_query: bool,
) -> None:
    # ConditionalLGATrBlock outputs match the input shape (multivectors and scalars).
    net = _block(s_channels, s_channels_cond, geometric_product, num_heads, multi_query)

    inputs = torch.randn(*BATCH_DIMS, NUM_ITEMS, MV_CHANNELS, 16)
    scalars = torch.randn(*BATCH_DIMS, NUM_ITEMS, s_channels) if s_channels else None
    mv_cond = torch.randn(*BATCH_DIMS, NUM_ITEMS_COND, MV_CHANNELS_COND, 16)
    s_cond = torch.randn(*BATCH_DIMS, NUM_ITEMS_COND, s_channels_cond) if s_channels_cond else None

    outputs, output_scalars = net(
        inputs, scalars=scalars, multivectors_cond=mv_cond, scalars_cond=s_cond
    )

    assert outputs.shape == (*BATCH_DIMS, NUM_ITEMS, MV_CHANNELS, 16)
    if s_channels:
        assert output_scalars.shape == (*BATCH_DIMS, NUM_ITEMS, s_channels)
    else:
        assert output_scalars is None


@pytest.mark.parametrize("s_channels,s_channels_cond,geometric_product", S_CHANNELS[:2])
@pytest.mark.parametrize("num_heads,multi_query", [(1, False), (4, True)])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_conditional_gatr_block_equivariance(
    s_channels: int,
    s_channels_cond: int,
    geometric_product: bool,
    num_heads: int,
    multi_query: bool,
    norm_elementwise_affine: bool,
) -> None:
    # ConditionalLGATrBlock is Spin-equivariant in both query and condition multivector inputs.
    net = _block(
        s_channels,
        s_channels_cond,
        geometric_product,
        num_heads,
        multi_query,
        norm_elementwise_affine=norm_elementwise_affine,
    )

    scalars = torch.randn(*BATCH_DIMS, NUM_ITEMS, s_channels) if s_channels else None
    scalars_cond = (
        torch.randn(*BATCH_DIMS, NUM_ITEMS_COND, s_channels_cond) if s_channels_cond else None
    )
    data_dims = [
        (*BATCH_DIMS, NUM_ITEMS, MV_CHANNELS),
        (*BATCH_DIMS, NUM_ITEMS_COND, MV_CHANNELS_COND),
    ]
    check_pin_equivariance(
        net,
        2,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars, scalars_cond=scalars_cond),
        **MILD_TOLERANCES,
    )


def test_conditional_lgatr_block_rejects_none_scalars() -> None:
    # A ConditionalLGATrBlock built with scalar channels rejects scalars=None at runtime.
    net = _block(
        s_channels=2, s_channels_cond=3, geometric_product=True, num_heads=2, multi_query=False
    )
    mv = torch.randn(3, 5, MV_CHANNELS, 16)
    mv_cond = torch.randn(3, 5, MV_CHANNELS_COND, 16)
    with pytest.raises(ValueError):
        net(mv, multivectors_cond=mv_cond, scalars=None, scalars_cond=None)
