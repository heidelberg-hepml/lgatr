import pytest
import torch

from lgatr.layers import SelfAttention, SelfAttentionConfig
from lgatr.layers.attention.qkv import MultiQueryQKVModule, QKVModule
from lgatr.primitives.attention import sdp_attention
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "num_items,in_channels,out_channels,increase_hidden_channels", [(2, 4, 4, 2)]
)
@pytest.mark.parametrize("in_s_channels,out_s_channels", [(17, 13), (11, 0)])
@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("multi_query,head_scale", [(True, True), (False, False)])
def test_attention_equivariance(
    batch_dims: list[int],
    num_items: int,
    in_channels: int,
    out_channels: int,
    num_heads: int,
    head_scale: bool,
    in_s_channels: int,
    out_s_channels: int,
    multi_query: bool,
    increase_hidden_channels: int,
) -> None:
    # SelfAttention is Pin-equivariant when scalar inputs are provided.
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
    layer = SelfAttention(config, PrimitivesConfig())

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


def test_sdp_attention_none_scalars() -> None:
    # sdp_attention propagates None when q_s/k_s/v_s are all None.
    q = k = v = torch.randn(2, 3, 4, 16)
    _, outputs_s = sdp_attention(q, k, v, q_s=None, k_s=None, v_s=None)
    assert outputs_s is None


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
