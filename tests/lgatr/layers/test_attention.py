import pytest
import torch
from torch import nn

from lgatr.layers import SelfAttention, SelfAttentionConfig
from lgatr.layers.attention.qkv import MultiQueryQKVModule, QKVModule
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance

NUM_ITEMS, MV_CHANNELS, ATTN_RATIO = 2, 4, 2


@pytest.mark.parametrize("in_s_channels,out_s_channels", [(17, 13), (11, 0)])
@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("multi_query", [False, True])
@pytest.mark.parametrize("head_scale", [False, True])
def test_attention_equivariance(
    num_heads: int,
    head_scale: bool,
    in_s_channels: int,
    out_s_channels: int,
    multi_query: bool,
) -> None:
    # SelfAttention is Spin-equivariant when scalar inputs are provided. Only Spin, because the
    # scalar stream mixes into the multivectors through a fixed reference multivector.
    config = SelfAttentionConfig(
        in_mv_channels=MV_CHANNELS,
        out_mv_channels=MV_CHANNELS,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        num_heads=num_heads,
        head_scale=head_scale,
        multi_query=multi_query,
        attn_ratio=ATTN_RATIO,
    )
    layer = SelfAttention(config, PrimitivesConfig())
    if head_scale:
        # default init is all-ones, which is indistinguishable from off, so randomize
        nn.init.normal_(layer.head_scale)

    data_dims = (*BATCH_DIMS, NUM_ITEMS, MV_CHANNELS)
    scalars = torch.randn(*BATCH_DIMS, NUM_ITEMS, in_s_channels)
    check_pin_equivariance(
        layer, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **TOLERANCES
    )


@pytest.mark.parametrize("multi_query", [False, True])
def test_qkv_module_none_scalars(multi_query: bool) -> None:
    # With no scalar stream (in_s_channels=0), both QKV modules return None scalar Q/K/V. Tested at
    # the module level since a scalar-free full LGATr raises (its bilinear layers require scalars).
    config = SelfAttentionConfig(
        in_mv_channels=3, num_heads=2, in_s_channels=0, multi_query=multi_query
    )
    module_cls = MultiQueryQKVModule if multi_query else QKVModule
    module = module_cls(config, PrimitivesConfig())

    q_mv, k_mv, v_mv, q_s, k_s, v_s = module(torch.randn(2, 5, 3, 16), scalars=None)

    assert q_s is None and k_s is None and v_s is None
    assert q_mv.shape[-1] == 16 and k_mv.shape[-1] == 16 and v_mv.shape[-1] == 16
