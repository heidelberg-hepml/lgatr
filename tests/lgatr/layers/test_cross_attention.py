import pytest
import torch

from lgatr.layers import CrossAttention, CrossAttentionConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("items,items_condition", [(3, 8)])
@pytest.mark.parametrize(
    "in_kv_mv_channels,in_q_mv_channels,in_kv_s_channels,in_q_s_channels",
    [(2, 3, 4, 5)],
)
@pytest.mark.parametrize("multi_query,head_scale,", [(True, True), (False, False)])
@pytest.mark.parametrize("num_heads,increase_hidden_channels", [(3, 2)])
@pytest.mark.parametrize("dropout_prob", [None])
def test_crossattention_equivariance(
    batch_dims,
    items,
    items_condition,
    in_kv_mv_channels,
    in_q_mv_channels,
    in_kv_s_channels,
    in_q_s_channels,
    multi_query,
    head_scale,
    num_heads,
    increase_hidden_channels,
    dropout_prob,
):
    """Test cross attention equivariance."""

    config = CrossAttentionConfig(
        in_kv_mv_channels=in_kv_mv_channels,
        in_q_mv_channels=in_q_mv_channels,
        out_mv_channels=in_q_mv_channels,
        in_kv_s_channels=in_kv_s_channels,
        in_q_s_channels=in_q_s_channels,
        out_s_channels=in_q_s_channels,
        num_heads=num_heads,
        head_scale=head_scale,
        increase_hidden_channels=increase_hidden_channels,
        multi_query=multi_query,
        dropout_prob=dropout_prob,
    )
    layer = CrossAttention(config)

    scalars = torch.randn(*batch_dims, items, in_q_s_channels)
    scalars_condition = torch.randn(*batch_dims, items_condition, in_kv_s_channels)

    data_dims = [
        tuple(list(batch_dims) + [items_condition, in_kv_mv_channels]),
        tuple(list(batch_dims) + [items, in_q_mv_channels]),
    ]
    check_pin_equivariance(
        layer,
        2,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars_kv=scalars_condition, scalars_q=scalars),
        **MILD_TOLERANCES,
    )


def test_cross_attention_none_scalars():
    """Tests CrossAttention accepts scalars_kv=None and scalars_q=None at runtime."""
    config = CrossAttentionConfig(
        in_kv_mv_channels=2,
        in_q_mv_channels=3,
        out_mv_channels=3,
        in_kv_s_channels=4,
        in_q_s_channels=5,
        out_s_channels=0,
        num_heads=2,
    )
    layer = CrossAttention(config)
    mv_q = torch.randn(2, 3, 3, 16)
    mv_kv = torch.randn(2, 4, 2, 16)
    layer(mv_kv, mv_q, scalars_kv=None, scalars_q=None)
