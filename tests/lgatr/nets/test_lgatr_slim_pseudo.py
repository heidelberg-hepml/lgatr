import pytest
import torch

from lgatr.layers.slim_pseudo_layers import (
    PseudoDeterminant,
    SlimPseudoBlock,
    SlimPseudoDropout,
    SlimPseudoGLU,
    SlimPseudoLinear,
    SlimPseudoMLP,
    SlimPseudoRMSNorm,
    SlimPseudoSelfAttention,
)
from lgatr.nets.slim_pseudo import LGATrSlimPseudo
from tests.helpers import BATCH_DIMS, TOLERANCES, check_equivariance

# (in_v, out_v, in_s, out_s, in_p, out_p)
CHANNELS = [
    (5, 1, 4, 2, 3, 2),
    (2, 1, 2, 0, 0, 1),
    (3, 3, 1, 3, 1, 0),
    (1, 2, 2, 3, 0, 0),
]

GLU_CASES = [(*channels, "sigmoid") for channels in CHANNELS]

LINEAR_CASES = [(*channels, "default") for channels in CHANNELS] + [(*CHANNELS[0], "small")]

PARITY = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0]))


def parity(vectors: torch.Tensor, vector_dim: int) -> torch.Tensor:
    """Spatial inversion of four-vectors stored along ``vector_dim``."""
    vectors = vectors.movedim(vector_dim, -1)
    return torch.einsum("ij,...j->...i", PARITY.to(vectors.dtype), vectors).movedim(-1, vector_dim)


def check_parity_equivariance(
    function, vectors, scalars, pseudoscalars, vector_dim: int = -2, **tolerances
) -> None:
    """Compare ``f(P x)`` against ``P f(x)``: vectors and pseudoscalars flip, scalars do not.

    Layers use the internal ``(..., 4, channels)`` layout (``vector_dim=-2``), the network the
    public ``(..., channels, 4)`` layout (``vector_dim=-1``).
    """
    out_v, out_s, out_p = function(vectors, scalars=scalars, pseudoscalars=pseudoscalars)

    out_v_of_transformed, out_s_of_transformed, out_p_of_transformed = function(
        parity(vectors, vector_dim), scalars=scalars, pseudoscalars=-pseudoscalars
    )

    torch.testing.assert_close(parity(out_v, vector_dim), out_v_of_transformed, **tolerances)
    torch.testing.assert_close(out_s, out_s_of_transformed, **tolerances)
    torch.testing.assert_close(-out_p, out_p_of_transformed, **tolerances)


def test_PseudoDeterminant_parity() -> None:
    layer = PseudoDeterminant(in_v_channels=4, out_p_channels=2)

    vectors = torch.randn(*BATCH_DIMS, 4, 4)
    scalars = torch.randn(*BATCH_DIMS, 3)
    pseudoscalars = torch.randn(*BATCH_DIMS, 2)
    tr_vectors, tr_scalars, tr_pseudoscalars = layer(vectors, scalars, pseudoscalars)
    parity_pseudoscalars = layer(parity(vectors, -2), scalars, -pseudoscalars)[2]

    torch.testing.assert_close(tr_vectors, vectors, **TOLERANCES)
    torch.testing.assert_close(tr_scalars, scalars, **TOLERANCES)
    torch.testing.assert_close(parity_pseudoscalars, -tr_pseudoscalars, **TOLERANCES)


@pytest.mark.parametrize("dropout_prob", [0.1, 0.5])
def test_SlimPseudoDropout_equivariance(dropout_prob: float) -> None:
    layer = SlimPseudoDropout(dropout_prob)
    layer.eval()

    v = torch.randn(*BATCH_DIMS, 4, 6)
    s = torch.randn(*BATCH_DIMS, 5)
    p = torch.randn(*BATCH_DIMS, 3)

    layer.train()
    torch.manual_seed(0)
    train_1 = layer(v, scalars=s, pseudoscalars=p)
    torch.manual_seed(0)
    train_2 = layer(v, scalars=s, pseudoscalars=p)
    for out_1, out_2, x in zip(train_1, train_2, (v, s, p), strict=True):
        assert out_1.shape == x.shape
        torch.testing.assert_close(out_1, out_2, **TOLERANCES)

    layer.eval()
    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, 6),
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        vector_dim=-2,
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("split_norm", [False, True])
def test_SlimPseudoRMSNorm_equivariance(split_norm: bool) -> None:
    v_channels, s_channels, p_channels = 6, 4, 3
    layer = SlimPseudoRMSNorm(v_channels, s_channels, p_channels, split_norm=split_norm)

    v = torch.randn(*BATCH_DIMS, 4, v_channels)
    s = torch.randn(*BATCH_DIMS, s_channels)
    p = torch.randn(*BATCH_DIMS, p_channels)
    outputs_v, outputs_s, outputs_p = layer(v, scalars=s, pseudoscalars=p)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape
    assert outputs_p.shape == p.shape

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, v_channels),
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        vector_dim=-2,
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("in_v,out_v,in_s,out_s,in_p,out_p,initialization", LINEAR_CASES)
def test_SlimPseudoLinear_equivariance(
    in_v: int,
    out_v: int,
    in_s: int,
    out_s: int,
    in_p: int,
    out_p: int,
    initialization: str,
) -> None:
    layer = SlimPseudoLinear(
        in_v_channels=in_v,
        out_v_channels=out_v,
        in_s_channels=in_s,
        out_s_channels=out_s,
        in_p_channels=in_p,
        out_p_channels=out_p,
        initialization=initialization,
    )
    s = torch.randn(*BATCH_DIMS, in_s)
    p = torch.randn(*BATCH_DIMS, in_p)
    v = torch.randn(*BATCH_DIMS, 4, in_v)
    outputs_v, outputs_s, outputs_p = layer(v, s, p)
    assert outputs_v.shape == (*BATCH_DIMS, 4, out_v)
    assert outputs_s.shape == (*BATCH_DIMS, out_s)
    assert outputs_p.shape == (*BATCH_DIMS, out_p)

    check_parity_equivariance(layer, v, s, p, **TOLERANCES)
    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, in_v),
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        vector_dim=-2,
        **TOLERANCES,
    )


@pytest.mark.parametrize("in_v,out_v,in_s,out_s,in_p,out_p,nonlinearity", GLU_CASES)
def test_SlimPseudoGLU_equivariance(
    in_v: int, out_v: int, in_s: int, out_s: int, in_p: int, out_p: int, nonlinearity: str
) -> None:
    layer = SlimPseudoGLU(
        in_v_channels=in_v,
        out_v_channels=out_v,
        in_s_channels=in_s,
        out_s_channels=out_s,
        in_p_channels=in_p,
        out_p_channels=out_p,
        nonlinearity=nonlinearity,
    )
    s = torch.randn(*BATCH_DIMS, in_s)
    p = torch.randn(*BATCH_DIMS, in_p)
    v = torch.randn(*BATCH_DIMS, 4, in_v)
    outputs_v, outputs_s, outputs_p = layer(v, s, p)
    assert outputs_v.shape == (*BATCH_DIMS, 4, out_v)
    assert outputs_s.shape == (*BATCH_DIMS, out_s)
    assert outputs_p.shape == (*BATCH_DIMS, out_p)

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, in_v),
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        vector_dim=-2,
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("num_heads,attn_ratio", [(2, 1), (1, 2)])
def test_SlimPseudoSelfAttention_equivariance(num_heads: int, attn_ratio: int) -> None:
    v_channels, s_channels, p_channels = 24, 14, 5
    layer = SlimPseudoSelfAttention(
        v_channels=v_channels,
        s_channels=s_channels,
        p_channels=p_channels,
        num_heads=num_heads,
        attn_ratio=attn_ratio,
    )
    s = torch.randn(*BATCH_DIMS, s_channels)
    p = torch.randn(*BATCH_DIMS, p_channels)
    v = torch.randn(*BATCH_DIMS, 4, v_channels)
    outputs_v, outputs_s, outputs_p = layer(v, s, p)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape
    assert outputs_p.shape == p.shape

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, v_channels),
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        vector_dim=-2,
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("v_channels,s_channels,p_channels", [(32, 2, 4), (16, 4, 1)])
@pytest.mark.parametrize("mlp_ratio,num_layers", [(1, 2), (2, 2), (1, 3)])
def test_SlimPseudoMLP_equivariance(
    v_channels: int, s_channels: int, p_channels: int, mlp_ratio: int, num_layers: int
) -> None:
    layer = SlimPseudoMLP(
        v_channels=v_channels,
        s_channels=s_channels,
        p_channels=p_channels,
        mlp_ratio=mlp_ratio,
        num_layers=num_layers,
    )
    s = torch.randn(*BATCH_DIMS, s_channels)
    p = torch.randn(*BATCH_DIMS, p_channels)
    v = torch.randn(*BATCH_DIMS, 4, v_channels)

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, v_channels),
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        vector_dim=-2,
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize(
    "v_channels,s_channels,p_channels,num_heads", [(32, 2, 4, 1), (16, 4, 1, 4)]
)
@pytest.mark.parametrize("dropout_prob", [None, 0.5])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
@pytest.mark.parametrize("split_norm", [False, True])
def test_SlimPseudoBlock_equivariance(
    v_channels: int,
    s_channels: int,
    p_channels: int,
    num_heads: int,
    dropout_prob: float | None,
    norm_elementwise_affine: bool,
    split_norm: bool,
) -> None:
    layer = SlimPseudoBlock(
        v_channels=v_channels,
        s_channels=s_channels,
        p_channels=p_channels,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        norm_elementwise_affine=norm_elementwise_affine,
        split_norm=split_norm,
    )
    layer.eval()
    s = torch.randn(*BATCH_DIMS, s_channels)
    p = torch.randn(*BATCH_DIMS, p_channels)
    v = torch.randn(*BATCH_DIMS, 4, v_channels)

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, v_channels),
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        vector_dim=-2,
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, **TOLERANCES)


@pytest.mark.parametrize("in_v,out_v,in_s,out_s,in_p,out_p", CHANNELS)
@pytest.mark.parametrize(
    "hidden_v,hidden_s,hidden_p,num_heads",
    [(32, 2, 4, 1), (16, 4, 1, 4)],
)
@pytest.mark.parametrize("num_blocks,checkpoint_blocks", [(1, False), (2, True)])
def test_LGATrSlimPseudo_equivariance(
    in_v: int,
    in_s: int,
    in_p: int,
    out_v: int,
    out_s: int,
    out_p: int,
    hidden_v: int,
    hidden_s: int,
    hidden_p: int,
    num_heads: int,
    num_blocks: int,
    checkpoint_blocks: bool,
) -> None:
    layer = LGATrSlimPseudo(
        num_blocks=num_blocks,
        in_v_channels=in_v,
        out_v_channels=out_v,
        hidden_v_channels=hidden_v,
        in_s_channels=in_s,
        out_s_channels=out_s,
        hidden_s_channels=hidden_s,
        num_heads=num_heads,
        in_p_channels=in_p,
        out_p_channels=out_p,
        hidden_p_channels=hidden_p,
        dropout_prob=0.5,
        checkpoint_blocks=checkpoint_blocks,
    )
    layer.eval()
    s = torch.randn(*BATCH_DIMS, in_s)
    p = torch.randn(*BATCH_DIMS, in_p)
    v = torch.randn(*BATCH_DIMS, in_v, 4)
    outputs_v, outputs_s, outputs_p = layer(v, s, p)
    assert outputs_v.shape == (*BATCH_DIMS, out_v, 4)
    assert outputs_s.shape == (*BATCH_DIMS, out_s)
    assert outputs_p.shape == (*BATCH_DIMS, out_p)

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, in_v),
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, vector_dim=-1, **TOLERANCES)
