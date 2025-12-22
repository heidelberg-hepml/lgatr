import pytest
import torch

from lgatr.nets.conditional_lgatr_slim import (
    ConditionalLGATrSlim,
    ConditionalLGATrSlimBlock,
    CrossAttention,
)

from ...helpers.constants import BATCH_DIMS, TOLERANCES
from ...helpers.equivariance_noga import check_equivariance

BATCH_DIMS = [b[:-1] for b in BATCH_DIMS]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("N,Ncond", [(3, 7), (13, 2)])
@pytest.mark.parametrize("v_channels,cond_v_channels,s_channels,cond_s_channels", [(24, 6, 14, 20)])
@pytest.mark.parametrize("num_heads,attn_ratio", [(2, 1), (1, 2)])
def test_CrossAttention_equivariance(
    batch_dims,
    N,
    Ncond,
    v_channels,
    cond_v_channels,
    s_channels,
    cond_s_channels,
    num_heads,
    attn_ratio,
):
    layer = CrossAttention(
        q_v_channels=v_channels,
        kv_v_channels=cond_v_channels,
        q_s_channels=s_channels,
        kv_s_channels=cond_s_channels,
        num_heads=num_heads,
        attn_ratio=attn_ratio,
    )
    s = torch.randn(*batch_dims, N, s_channels)
    cond_s = torch.randn(*batch_dims, Ncond, cond_s_channels)

    v = torch.randn(*batch_dims, N, v_channels, 4)
    cond_v = torch.randn(*batch_dims, Ncond, cond_v_channels, 4)
    out_v, out_s = layer(v, cond_v, s, cond_s)
    assert out_v.shape == v.shape
    assert out_s.shape == s.shape

    batch_dims = [batch_dims + [N, v_channels], batch_dims + [Ncond, cond_v_channels]]
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        num_args=2,
        fn_kwargs=dict(scalars=s, scalars_condition=cond_s),
        **TOLERANCES,
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("N,Ncond", [(3, 7), (13, 2)])
@pytest.mark.parametrize(
    "v_channels,cond_v_channels,s_channels,cond_s_channels,num_heads", [(24, 6, 14, 20, 2)]
)
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.5])
def test_ConditionalLGATrSlimBlock_equivariance(
    batch_dims,
    N,
    Ncond,
    v_channels,
    cond_v_channels,
    s_channels,
    cond_s_channels,
    num_heads,
    dropout_prob,
):
    layer = ConditionalLGATrSlimBlock(
        v_channels=v_channels,
        condition_v_channels=cond_v_channels,
        s_channels=s_channels,
        condition_s_channels=cond_s_channels,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
    )
    layer.eval()

    s = torch.randn(*batch_dims, N, s_channels)
    cond_s = torch.randn(*batch_dims, Ncond, cond_s_channels)
    batch_dims = [batch_dims + [N, v_channels], batch_dims + [Ncond, cond_v_channels]]

    # equivariance
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        num_args=2,
        fn_kwargs=dict(scalars=s, scalars_condition=cond_s),
        **TOLERANCES,
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("N,Ncond", [(3, 7), (13, 2)])
@pytest.mark.parametrize(
    "in_v_channels,in_s_channels,out_v_channels,out_s_channels,cond_v_channels,cond_s_channels",
    [
        (4, 3, 9, 2, 5, 6),
        (2, 9, 0, 3, 4, 7),
        (3, 5, 7, 0, 6, 8),
        (8, 3, 0, 0, 7, 9),
    ],
)
@pytest.mark.parametrize("hidden_v_channels,hidden_s_channels,num_heads", [(32, 4, 1), (16, 8, 4)])
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.5])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_ConditionalLGATrSlim_equivariance(
    batch_dims,
    N,
    Ncond,
    in_v_channels,
    in_s_channels,
    out_v_channels,
    out_s_channels,
    cond_v_channels,
    cond_s_channels,
    hidden_v_channels,
    hidden_s_channels,
    num_heads,
    num_blocks,
    dropout_prob,
    checkpoint_blocks,
):
    layer = ConditionalLGATrSlim(
        in_v_channels=in_v_channels,
        condition_v_channels=cond_v_channels,
        out_v_channels=out_v_channels,
        hidden_v_channels=hidden_v_channels,
        in_s_channels=in_s_channels,
        condition_s_channels=cond_s_channels,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        checkpoint_blocks=checkpoint_blocks,
    )
    layer.eval()

    s = torch.randn(*batch_dims, N, in_s_channels)
    cond_s = torch.randn(*batch_dims, Ncond, cond_s_channels)
    v = torch.randn(*batch_dims, N, in_v_channels, 4)
    cond_v = torch.randn(*batch_dims, Ncond, cond_v_channels, 4)

    out_v, out_s = layer(
        vectors=v,
        vectors_condition=cond_v,
        scalars=s,
        scalars_condition=cond_s,
    )
    assert out_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert out_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = [batch_dims + [N, in_v_channels], batch_dims + [Ncond, cond_v_channels]]
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        num_args=2,
        fn_kwargs=dict(scalars=s, scalars_condition=cond_s),
        **TOLERANCES,
    )
