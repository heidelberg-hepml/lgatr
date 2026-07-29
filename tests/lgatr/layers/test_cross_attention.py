import pytest
import torch
from torch import nn

from lgatr.layers import CrossAttention, CrossAttentionConfig
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

ITEMS_Q, ITEMS_KV = 3, 8
Q_MV_CHANNELS, KV_MV_CHANNELS = 3, 2
Q_S_CHANNELS, KV_S_CHANNELS = 5, 4


@pytest.mark.parametrize("multi_query", [False, True])
@pytest.mark.parametrize("head_scale", [False, True])
@pytest.mark.parametrize("dropout_prob", [None, 0.5])
def test_crossattention_equivariance(
    multi_query: bool,
    head_scale: bool,
    dropout_prob: float | None,
) -> None:
    # CrossAttention is Spin-equivariant in both query and key/value multivector inputs. Eval mode
    # is required: a train-mode dropout mask is random and not equivariant.
    config = CrossAttentionConfig(
        kv_mv_channels=KV_MV_CHANNELS,
        q_mv_channels=Q_MV_CHANNELS,
        out_mv_channels=Q_MV_CHANNELS,
        kv_s_channels=KV_S_CHANNELS,
        q_s_channels=Q_S_CHANNELS,
        out_s_channels=Q_S_CHANNELS,
        num_heads=3,
        head_scale=head_scale,
        attn_ratio=2,
        multi_query=multi_query,
        dropout_prob=dropout_prob,
    )
    layer = CrossAttention(config, PrimitivesConfig())
    if head_scale:
        # default init is all-ones, which is indistinguishable from off, so randomize
        nn.init.normal_(layer.head_scale)
    layer.eval()

    scalars_q = torch.randn(*BATCH_DIMS, ITEMS_Q, Q_S_CHANNELS)
    scalars_kv = torch.randn(*BATCH_DIMS, ITEMS_KV, KV_S_CHANNELS)
    data_dims = [
        (*BATCH_DIMS, ITEMS_Q, Q_MV_CHANNELS),
        (*BATCH_DIMS, ITEMS_KV, KV_MV_CHANNELS),
    ]
    check_pin_equivariance(
        layer,
        2,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars_kv=scalars_kv, scalars_q=scalars_q),
        **MILD_TOLERANCES,
    )


def test_cross_attention_rejects_none_scalars() -> None:
    # A CrossAttention built with scalar channels rejects scalars_q/scalars_kv=None at runtime.
    config = CrossAttentionConfig(
        kv_mv_channels=2,
        q_mv_channels=3,
        out_mv_channels=3,
        kv_s_channels=4,
        q_s_channels=5,
        out_s_channels=0,
        num_heads=2,
    )
    layer = CrossAttention(config, PrimitivesConfig())
    mv_q = torch.randn(2, 3, 3, 16)
    mv_kv = torch.randn(2, 4, 2, 16)
    with pytest.raises(ValueError):
        layer(mv_q, mv_kv, scalars_q=None, scalars_kv=None)


def test_cross_attention_zero_scalar_channels() -> None:
    # With no scalar stream (query and key/value scalar channels both 0), the scalar Q/K/V and the
    # output scalars are all None.
    config = CrossAttentionConfig(
        kv_mv_channels=2,
        q_mv_channels=3,
        out_mv_channels=3,
        kv_s_channels=0,
        q_s_channels=0,
        out_s_channels=0,
        num_heads=2,
    )
    layer = CrossAttention(config, PrimitivesConfig())
    outputs_mv, outputs_s = layer(torch.randn(2, 3, 3, 16), torch.randn(2, 4, 2, 16))
    assert outputs_mv.shape == (2, 3, 3, 16)
    assert outputs_s is None
