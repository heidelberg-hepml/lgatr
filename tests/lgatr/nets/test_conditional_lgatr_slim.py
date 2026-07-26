import pytest
import torch

from lgatr.nets.conditional_slim import ConditionalLGATrSlim
from lgatr.nets.slim_layers import ConditionalSlimBlock, SlimCrossAttention
from tests.helpers import BATCH_DIMS, TOLERANCES, check_equivariance

BATCH_DIMS = BATCH_DIMS[:-1]
V_CHANNELS, V_CHANNELS_COND = 24, 6
S_CHANNELS, S_CHANNELS_COND = 14, 20


@pytest.mark.parametrize("N,N_cond", [(3, 7), (13, 2)])
@pytest.mark.parametrize("num_heads,attn_ratio", [(2, 1), (1, 2)])
def test_SlimCrossAttention_equivariance(
    N: int, N_cond: int, num_heads: int, attn_ratio: int
) -> None:
    # SlimCrossAttention preserves shapes and is SO(1, 3)-equivariant in both inputs.
    layer = SlimCrossAttention(
        q_v_channels=V_CHANNELS,
        kv_v_channels=V_CHANNELS_COND,
        q_s_channels=S_CHANNELS,
        kv_s_channels=S_CHANNELS_COND,
        num_heads=num_heads,
        attn_ratio=attn_ratio,
    )
    s = torch.randn(*BATCH_DIMS, N, S_CHANNELS)
    s_cond = torch.randn(*BATCH_DIMS, N_cond, S_CHANNELS_COND)
    v = torch.randn(*BATCH_DIMS, N, 4, V_CHANNELS)
    v_cond = torch.randn(*BATCH_DIMS, N_cond, 4, V_CHANNELS_COND)

    outputs_v, outputs_s = layer(v, v_cond, s, s_cond)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    check_equivariance(
        layer,
        batch_dims=[
            (*BATCH_DIMS, N, V_CHANNELS),
            (*BATCH_DIMS, N_cond, V_CHANNELS_COND),
        ],
        num_args=2,
        fn_kwargs=dict(scalars_q=s, scalars_kv=s_cond),
        vector_dim=-2,
        **TOLERANCES,
    )


@pytest.mark.parametrize("N,N_cond", [(3, 7), (13, 2)])
@pytest.mark.parametrize("dropout_prob", [None, 0.5])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_ConditionalSlimBlock_equivariance(
    N: int, N_cond: int, dropout_prob: float | None, norm_elementwise_affine: bool
) -> None:
    # ConditionalSlimBlock is SO(1, 3)-equivariant at eval time.
    layer = ConditionalSlimBlock(
        v_channels=V_CHANNELS,
        v_channels_cond=V_CHANNELS_COND,
        s_channels=S_CHANNELS,
        s_channels_cond=S_CHANNELS_COND,
        num_heads=2,
        dropout_prob=dropout_prob,
        norm_elementwise_affine=norm_elementwise_affine,
    )
    layer.eval()

    s = torch.randn(*BATCH_DIMS, N, S_CHANNELS)
    s_cond = torch.randn(*BATCH_DIMS, N_cond, S_CHANNELS_COND)

    check_equivariance(
        layer,
        batch_dims=[
            (*BATCH_DIMS, N, V_CHANNELS),
            (*BATCH_DIMS, N_cond, V_CHANNELS_COND),
        ],
        num_args=2,
        fn_kwargs=dict(scalars=s, scalars_cond=s_cond),
        vector_dim=-2,
        **TOLERANCES,
    )


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
@pytest.mark.parametrize("num_blocks,checkpoint_blocks", [(1, False), (2, True)])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_ConditionalLGATrSlim_equivariance(
    N: int,
    N_cond: int,
    in_v_channels: int,
    in_s_channels: int,
    out_v_channels: int,
    out_s_channels: int,
    v_channels_cond: int,
    s_channels_cond: int,
    num_blocks: int,
    checkpoint_blocks: bool,
    norm_elementwise_affine: bool,
) -> None:
    # ConditionalLGATrSlim (full network) preserves shapes and is SO(1, 3)-equivariant at eval
    # time. The layer is in eval mode, so dropout_prob would not change anything and is fixed here.
    layer = ConditionalLGATrSlim(
        in_v_channels=in_v_channels,
        v_channels_cond=v_channels_cond,
        out_v_channels=out_v_channels,
        hidden_v_channels=16,
        in_s_channels=in_s_channels,
        s_channels_cond=s_channels_cond,
        out_s_channels=out_s_channels,
        hidden_s_channels=8,
        num_blocks=num_blocks,
        num_heads=4,
        dropout_prob=0.5,
        checkpoint_blocks=checkpoint_blocks,
        norm_elementwise_affine=norm_elementwise_affine,
    )
    layer.eval()

    s = torch.randn(*BATCH_DIMS, N, in_s_channels)
    s_cond = torch.randn(*BATCH_DIMS, N_cond, s_channels_cond)
    v = torch.randn(*BATCH_DIMS, N, in_v_channels, 4)
    v_cond = torch.randn(*BATCH_DIMS, N_cond, v_channels_cond, 4)

    outputs_v, outputs_s = layer(vectors=v, vectors_cond=v_cond, scalars=s, scalars_cond=s_cond)
    assert outputs_v.shape == (*BATCH_DIMS, N, out_v_channels, 4)
    assert outputs_s.shape == (*BATCH_DIMS, N, out_s_channels)

    check_equivariance(
        layer,
        batch_dims=[
            (*BATCH_DIMS, N, in_v_channels),
            (*BATCH_DIMS, N_cond, v_channels_cond),
        ],
        num_args=2,
        fn_kwargs=dict(scalars=s, scalars_cond=s_cond),
        **TOLERANCES,
    )
