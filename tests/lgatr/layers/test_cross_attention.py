import pytest
import torch

from lgatr.layers import CrossAttention, CrossAttentionConfig
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("items_q,items_kv", [(3, 8)])
@pytest.mark.parametrize(
    "kv_mv_channels,q_mv_channels,kv_s_channels,q_s_channels",
    [(2, 3, 4, 5)],
)
@pytest.mark.parametrize("multi_query,head_scale", [(True, True), (False, False)])
@pytest.mark.parametrize("num_heads,increase_hidden_channels", [(3, 2)])
@pytest.mark.parametrize("dropout_prob", [None])
def test_crossattention_equivariance(
    batch_dims: list[int],
    items_q: int,
    items_kv: int,
    kv_mv_channels: int,
    q_mv_channels: int,
    kv_s_channels: int,
    q_s_channels: int,
    multi_query: bool,
    head_scale: bool,
    num_heads: int,
    increase_hidden_channels: int,
    dropout_prob: float | None,
) -> None:
    # CrossAttention is Pin-equivariant in both query and key/value multivector inputs.
    config = CrossAttentionConfig(
        kv_mv_channels=kv_mv_channels,
        q_mv_channels=q_mv_channels,
        out_mv_channels=q_mv_channels,
        kv_s_channels=kv_s_channels,
        q_s_channels=q_s_channels,
        out_s_channels=q_s_channels,
        num_heads=num_heads,
        head_scale=head_scale,
        increase_hidden_channels=increase_hidden_channels,
        multi_query=multi_query,
        dropout_prob=dropout_prob,
    )
    layer = CrossAttention(config, PrimitivesConfig())

    scalars_q = torch.randn(*batch_dims, items_q, q_s_channels)
    scalars_kv = torch.randn(*batch_dims, items_kv, kv_s_channels)

    data_dims = [
        tuple(list(batch_dims) + [items_q, q_mv_channels]),
        tuple(list(batch_dims) + [items_kv, kv_mv_channels]),
    ]
    check_pin_equivariance(
        layer,
        2,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars_kv=scalars_kv, scalars_q=scalars_q),
        **MILD_TOLERANCES,
    )


def test_cross_attention_none_scalars() -> None:
    # CrossAttention accepts scalars_kv=None and scalars_q=None at runtime.
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
    layer(mv_q, mv_kv, scalars_q=None, scalars_kv=None)
