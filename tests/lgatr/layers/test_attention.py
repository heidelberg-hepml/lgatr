import pytest
import torch

from lgatr.layers import SelfAttention, SelfAttentionConfig
from lgatr.primitives.attention import sdp_attention
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "num_items,in_channels,out_channels,increase_hidden_channels", [(2, 4, 4, 2)]
)
@pytest.mark.parametrize("in_s_channels,out_s_channels", [(17, 13), (11, 0)])
@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("multi_query,head_scale", [(True, True), (False, False)])
def test_attention_equivariance(
    batch_dims,
    num_items,
    in_channels,
    out_channels,
    num_heads,
    head_scale,
    in_s_channels,
    out_s_channels,
    multi_query,
    increase_hidden_channels,
):
    """Tests the SelfAttention layer for Pin equivariance, with scalar inputs."""

    config = SelfAttentionConfig(
        in_mv_channels=in_channels,
        out_mv_channels=out_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        num_heads=num_heads,
        head_scale=head_scale,
        multi_query=multi_query,
        increase_hidden_channels=increase_hidden_channels,
    )
    layer = SelfAttention(config)

    data_dims = tuple(list(batch_dims) + [num_items, in_channels])
    scalars = torch.randn(*batch_dims, num_items, in_s_channels)
    check_pin_equivariance(
        layer,
        1,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars),
        spin=True,
        **TOLERANCES,
    )


def test_sdp_attention_none_scalars():
    """Tests sdp_attention propagates None when q_s/k_s/v_s are None."""
    q = k = v = torch.randn(2, 3, 4, 16)
    _, out_s = sdp_attention(q, k, v, q_s=None, k_s=None, v_s=None)
    assert out_s is None
