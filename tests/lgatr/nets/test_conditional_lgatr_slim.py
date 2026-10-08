import pytest
import torch

from lgatr.interface import from_lightcone, get_lightcone_frame, to_lightcone
from lgatr.layers.slim_layers import ConditionalSlimBlock, SlimCrossAttention
from lgatr.nets.conditional_slim import ConditionalLGATrSlim
from tests.helpers import BATCH_DIMS, STRICT_TOLERANCES, TOLERANCES, check_equivariance

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


def _outputs_and_grads(model, *inputs):
    outputs_v, outputs_s = model(*inputs)
    (outputs_v.square().sum() + outputs_s.square().sum()).backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    return outputs_v.detach(), outputs_s.detach(), grads


def test_ConditionalLGATrSlim_lightcone_matches_cartesian() -> None:
    # The light-cone coordinates are an exact change of basis: with the same weights, a network
    # fed the mapped vectors computes the same function and gradients, up to rounding.
    channels = dict(
        in_v_channels=2,
        v_channels_cond=2,
        out_v_channels=2,
        hidden_v_channels=8,
        in_s_channels=3,
        s_channels_cond=3,
        out_s_channels=3,
        hidden_s_channels=8,
        num_blocks=2,
        num_heads=2,
    )
    cartesian = ConditionalLGATrSlim(**channels).double().eval()
    for block in cartesian.blocks:
        # the "small" init of the q/kv projections hides most of the attention
        block.selfattention.linear_in.reset_parameters("default")
        block.crossattention.linear_in_q.reset_parameters("default")
        block.crossattention.linear_in_kv.reset_parameters("default")
    lightcone = ConditionalLGATrSlim(**channels, lightcone=True).double().eval()
    lightcone.load_state_dict(cartesian.state_dict())

    items, items_cond = 4, 3
    s = torch.randn(*BATCH_DIMS, items, channels["in_s_channels"], dtype=torch.float64)
    s_cond = torch.randn(*BATCH_DIMS, items_cond, channels["s_channels_cond"], dtype=torch.float64)
    v = torch.randn(*BATCH_DIMS, items, channels["in_v_channels"], 4, dtype=torch.float64)
    v_cond = torch.randn(
        *BATCH_DIMS, items_cond, channels["v_channels_cond"], 4, dtype=torch.float64
    )
    frame = get_lightcone_frame(v.sum(dim=(-3, -2)))[..., None, None, :, :]

    # the frame is orthogonal, so the loss is the same in both coordinates
    outputs_v, outputs_s, grads = _outputs_and_grads(cartesian, v, v_cond, s, s_cond)
    outputs_v_lc, outputs_s_lc, grads_lc = _outputs_and_grads(
        lightcone, to_lightcone(v, frame), to_lightcone(v_cond, frame), s, s_cond
    )
    torch.testing.assert_close(from_lightcone(outputs_v_lc, frame), outputs_v, **STRICT_TOLERANCES)
    torch.testing.assert_close(outputs_s_lc, outputs_s, **STRICT_TOLERANCES)
    torch.testing.assert_close(grads_lc, grads, **STRICT_TOLERANCES)
