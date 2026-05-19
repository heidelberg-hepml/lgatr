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
@pytest.mark.parametrize("N,N_cond", [(3, 7), (13, 2)])
@pytest.mark.parametrize("v_channels,v_channels_cond,s_channels,s_channels_cond", [(24, 6, 14, 20)])
@pytest.mark.parametrize("num_heads,attn_ratio", [(2, 1), (1, 2)])
def test_CrossAttention_equivariance(
    batch_dims: list[int],
    N: int,
    N_cond: int,
    v_channels: int,
    v_channels_cond: int,
    s_channels: int,
    s_channels_cond: int,
    num_heads: int,
    attn_ratio: int,
) -> None:
    # Slim CrossAttention preserves shapes and is SO(1, 3)-equivariant in both inputs.
    layer = CrossAttention(
        q_v_channels=v_channels,
        kv_v_channels=v_channels_cond,
        q_s_channels=s_channels,
        kv_s_channels=s_channels_cond,
        num_heads=num_heads,
        attn_ratio=attn_ratio,
    )
    s = torch.randn(*batch_dims, N, s_channels)
    s_cond = torch.randn(*batch_dims, N_cond, s_channels_cond)

    v = torch.randn(*batch_dims, N, v_channels, 4)
    v_cond = torch.randn(*batch_dims, N_cond, v_channels_cond, 4)
    outputs_v, outputs_s = layer(v, v_cond, s, s_cond)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    batch_dims = [batch_dims + [N, v_channels], batch_dims + [N_cond, v_channels_cond]]
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        num_args=2,
        fn_kwargs=dict(scalars_q=s, scalars_kv=s_cond),
        **TOLERANCES,
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("N,N_cond", [(3, 7), (13, 2)])
@pytest.mark.parametrize(
    "v_channels,v_channels_cond,s_channels,s_channels_cond,num_heads", [(24, 6, 14, 20, 2)]
)
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.5])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_ConditionalLGATrSlimBlock_equivariance(
    batch_dims: list[int],
    N: int,
    N_cond: int,
    v_channels: int,
    v_channels_cond: int,
    s_channels: int,
    s_channels_cond: int,
    num_heads: int,
    dropout_prob: float | None,
    norm_elementwise_affine: bool,
) -> None:
    # ConditionalLGATrSlimBlock is SO(1, 3)-equivariant at eval time.
    layer = ConditionalLGATrSlimBlock(
        v_channels=v_channels,
        v_channels_cond=v_channels_cond,
        s_channels=s_channels,
        s_channels_cond=s_channels_cond,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        norm_elementwise_affine=norm_elementwise_affine,
    )
    layer.eval()

    s = torch.randn(*batch_dims, N, s_channels)
    s_cond = torch.randn(*batch_dims, N_cond, s_channels_cond)
    batch_dims = [batch_dims + [N, v_channels], batch_dims + [N_cond, v_channels_cond]]

    # equivariance
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        num_args=2,
        fn_kwargs=dict(scalars=s, scalars_cond=s_cond),
        **TOLERANCES,
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("N,N_cond", [(3, 7), (13, 2)])
@pytest.mark.parametrize(
    "in_v_channels,in_s_channels,out_v_channels,out_s_channels,v_channels_cond,s_channels_cond",
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
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_ConditionalLGATrSlim_equivariance(
    batch_dims: list[int],
    N: int,
    N_cond: int,
    in_v_channels: int,
    in_s_channels: int,
    out_v_channels: int,
    out_s_channels: int,
    v_channels_cond: int,
    s_channels_cond: int,
    hidden_v_channels: int,
    hidden_s_channels: int,
    num_heads: int,
    num_blocks: int,
    dropout_prob: float | None,
    checkpoint_blocks: bool,
    norm_elementwise_affine: bool,
) -> None:
    # ConditionalLGATrSlim (full network) preserves shapes and is SO(1, 3)-equivariant at eval time.
    layer = ConditionalLGATrSlim(
        in_v_channels=in_v_channels,
        v_channels_cond=v_channels_cond,
        out_v_channels=out_v_channels,
        hidden_v_channels=hidden_v_channels,
        in_s_channels=in_s_channels,
        s_channels_cond=s_channels_cond,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        checkpoint_blocks=checkpoint_blocks,
        norm_elementwise_affine=norm_elementwise_affine,
    )
    layer.eval()

    s = torch.randn(*batch_dims, N, in_s_channels)
    s_cond = torch.randn(*batch_dims, N_cond, s_channels_cond)
    v = torch.randn(*batch_dims, N, in_v_channels, 4)
    v_cond = torch.randn(*batch_dims, N_cond, v_channels_cond, 4)

    outputs_v, outputs_s = layer(
        vectors=v,
        vectors_cond=v_cond,
        scalars=s,
        scalars_cond=s_cond,
    )
    assert outputs_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert outputs_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = [batch_dims + [N, in_v_channels], batch_dims + [N_cond, v_channels_cond]]
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        num_args=2,
        fn_kwargs=dict(scalars=s, scalars_cond=s_cond),
        **TOLERANCES,
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("N,N_cond", [(3, 7)])
@pytest.mark.parametrize(
    "in_v_channels,in_s_channels,out_v_channels,out_s_channels,v_channels_cond,s_channels_cond",
    [(4, 3, 9, 2, 5, 6)],
)
@pytest.mark.parametrize(
    "hidden_v_channels,hidden_s_channels,num_heads,num_blocks", [(16, 8, 4, 1)]
)
def test_ConditionalLGATrSlim_equivariance_compiled(
    batch_dims: list[int],
    N: int,
    N_cond: int,
    in_v_channels: int,
    in_s_channels: int,
    out_v_channels: int,
    out_s_channels: int,
    v_channels_cond: int,
    s_channels_cond: int,
    hidden_v_channels: int,
    hidden_s_channels: int,
    num_heads: int,
    num_blocks: int,
    compile: bool = True,
) -> None:
    # torch.compile-wrapped ConditionalLGATrSlim still preserves shapes and SO(1, 3)-equivariance.
    layer = ConditionalLGATrSlim(
        in_v_channels=in_v_channels,
        v_channels_cond=v_channels_cond,
        out_v_channels=out_v_channels,
        hidden_v_channels=hidden_v_channels,
        in_s_channels=in_s_channels,
        s_channels_cond=s_channels_cond,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        compile=compile,
    )
    layer.eval()

    s = torch.randn(*batch_dims, N, in_s_channels)
    s_cond = torch.randn(*batch_dims, N_cond, s_channels_cond)
    v = torch.randn(*batch_dims, N, in_v_channels, 4)
    v_cond = torch.randn(*batch_dims, N_cond, v_channels_cond, 4)

    outputs_v, outputs_s = layer(
        vectors=v,
        vectors_cond=v_cond,
        scalars=s,
        scalars_cond=s_cond,
    )
    assert outputs_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert outputs_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = [batch_dims + [N, in_v_channels], batch_dims + [N_cond, v_channels_cond]]
    check_equivariance(
        layer,
        batch_dims=batch_dims,
        num_args=2,
        fn_kwargs=dict(scalars=s, scalars_cond=s_cond),
        **TOLERANCES,
    )
