import pytest
import torch

from lgatr.layers.slim_layers import SlimLinear
from lgatr.layers.slim_pseudo_layers import (
    SlimPseudoBlock,
    SlimPseudoDropout,
    SlimPseudoGLU,
    SlimPseudoLinear,
    SlimPseudoMixing,
    SlimPseudoMLP,
    SlimPseudoRMSNorm,
    SlimPseudoSelfAttention,
    VectorToPseudoscalar,
    det4x4,
)
from lgatr.nets.slim import LGATrSlim
from lgatr.nets.slim_pseudo import LGATrSlimPseudo
from tests.helpers import BATCH_DIMS, TOLERANCES, check_equivariance
from tests.helpers.equivariance import RandomLorentzTransform

# (in_v, out_v, in_s, out_s, in_p, out_p), covering the zero-channel edges on every slot. The slim
# nets require scalars, so in_s is only zero on layers that are tested standalone.
CHANNELS = [
    (5, 1, 4, 2, 3, 2),
    (1, 4, 0, 2, 2, 3),
    (9, 3, 4, 0, 1, 1),
    (2, 7, 0, 0, 4, 2),
    (0, 1, 2, 3, 2, 1),
    (3, 0, 2, 3, 1, 4),
    (0, 0, 2, 3, 3, 2),
]
NONLINEARITIES = ["relu", "sigmoid", "tanh", "gelu", "silu"]

# Channels and nonlinearity cannot interact, so sweep them as a union rather than a product.
GLU_CASES = [(*channels, "sigmoid") for channels in CHANNELS]
GLU_CASES += [
    (*CHANNELS[0], nonlinearity) for nonlinearity in NONLINEARITIES if nonlinearity != "sigmoid"
]

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


def test_VectorToPseudoscalar_flips_under_parity() -> None:
    # The learned oriented 4-volume is parity-odd for an explicitly set weight.
    layer = VectorToPseudoscalar(in_v_channels=4, out_p_channels=2)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[0] = torch.eye(4)
        layer.weight[1] = 2.0 * torch.eye(4)

    vectors = torch.randn(*BATCH_DIMS, 4, 4)
    pseudoscalar = layer(vectors)
    parity_pseudoscalar = layer(parity(vectors, -2))

    torch.testing.assert_close(parity_pseudoscalar, -pseudoscalar, **TOLERANCES)


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


def test_SlimPseudoRMSNorm_split_norm_per_stream() -> None:
    v_channels, s_channels, p_channels = 6, 4, 3
    layer = SlimPseudoRMSNorm(v_channels, s_channels, p_channels, split_norm=True)
    v = torch.randn(*BATCH_DIMS, 4, v_channels)
    s = torch.randn(*BATCH_DIMS, s_channels)
    p = torch.randn(*BATCH_DIMS, p_channels)

    # each stream is normalized by its own rms, independently of the other two
    _, outputs_s, outputs_p = layer(v, s, p)
    s_norm = torch.rsqrt(s.square().mean(-1) + layer.epsilon)
    p_norm = torch.rsqrt(p.square().mean(-1) + layer.epsilon)
    torch.testing.assert_close(outputs_s, s * s_norm[..., None] * layer.weight_s, **TOLERANCES)
    torch.testing.assert_close(outputs_p, p * p_norm[..., None] * layer.weight_p, **TOLERANCES)


def test_SlimPseudoRMSNorm_split_norm_changes_output() -> None:
    # split_norm must be honoured by forward; it used to be a dead forward-only argument.
    v_channels, s_channels, p_channels = 6, 4, 3
    v = torch.randn(*BATCH_DIMS, 4, v_channels)
    s = 10.0 * torch.randn(*BATCH_DIMS, s_channels)
    p = torch.randn(*BATCH_DIMS, p_channels)

    shared = SlimPseudoRMSNorm(v_channels, s_channels, p_channels, split_norm=False)
    split = SlimPseudoRMSNorm(v_channels, s_channels, p_channels, split_norm=True)

    for shared_out, split_out in zip(shared(v, s, p), split(v, s, p), strict=True):
        assert not torch.allclose(shared_out, split_out, **TOLERANCES)


@pytest.mark.parametrize("v_channels,s_channels,p_channels", [(6, 4, 0), (0, 4, 3), (6, 0, 0)])
def test_SlimPseudoRMSNorm_split_norm_handles_empty_streams(
    v_channels: int, s_channels: int, p_channels: int
) -> None:
    # a zero-channel stream must pass through rather than divide by zero
    layer = SlimPseudoRMSNorm(v_channels, s_channels, p_channels, split_norm=True)
    v = torch.randn(*BATCH_DIMS, 4, v_channels)
    s = torch.randn(*BATCH_DIMS, s_channels)
    p = torch.randn(*BATCH_DIMS, p_channels)

    for output in layer(v, s, p):
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("compress", [False, True])
def test_det4x4_flips_under_parity(compress: bool) -> None:
    # asinh is odd, so compression preserves the parity-oddness of the determinant
    m = torch.randn(*BATCH_DIMS, 4, 4)
    parity_m = parity(m, -1)

    torch.testing.assert_close(
        det4x4(parity_m, compress=compress), -det4x4(m, compress=compress), **TOLERANCES
    )


def test_det4x4_compress_changes_output() -> None:
    m = torch.randn(*BATCH_DIMS, 4, 4)
    raw = det4x4(m, compress=False)
    assert not torch.allclose(raw, det4x4(m, compress=True), **TOLERANCES)
    torch.testing.assert_close(det4x4(m, compress=True), torch.asinh(raw), **TOLERANCES)


def test_SlimPseudoMixing_equivariance() -> None:
    # vectors -> pseudoscalars is parity-odd, squared pseudoscalars -> scalars parity-even
    v_channels, s_channels, p_channels = 6, 5, 3
    layer = SlimPseudoMixing(v_channels, s_channels, p_channels)
    v = torch.randn(*BATCH_DIMS, 4, v_channels)
    p = torch.randn(*BATCH_DIMS, p_channels)

    outputs_s, outputs_p = layer(v, p)
    assert outputs_s.shape == (*BATCH_DIMS, s_channels)
    assert outputs_p.shape == (*BATCH_DIMS, p_channels)

    parity_s, parity_p = layer(parity(v, -2), -p)
    torch.testing.assert_close(parity_s, outputs_s, **TOLERANCES)
    torch.testing.assert_close(parity_p, -outputs_p, **TOLERANCES)

    def fn(vectors, pseudoscalars):
        outputs_s, outputs_p = layer(vectors, pseudoscalars)
        return torch.cat([outputs_s, outputs_p], dim=-1)

    for _ in range(2):
        transform = RandomLorentzTransform(vector_dim=-2)
        torch.testing.assert_close(fn(transform(v), p), fn(v, p), **TOLERANCES)


def test_det_compress_reaches_mixing() -> None:
    kwargs = dict(
        num_blocks=2,
        in_v_channels=1,
        out_v_channels=0,
        hidden_v_channels=8,
        in_s_channels=4,
        out_s_channels=2,
        hidden_s_channels=8,
        num_heads=2,
        in_p_channels=2,
        out_p_channels=1,
        hidden_p_channels=4,
    )
    torch.manual_seed(0)
    compressed = LGATrSlimPseudo(**kwargs, det_compress=True)
    torch.manual_seed(0)
    raw = LGATrSlimPseudo(**kwargs, det_compress=False)
    for block in compressed.blocks:
        assert block.mixing.vector_to_p._det_compress
    for block in raw.blocks:
        assert not block.mixing.vector_to_p._det_compress


def test_LGATrSlimPseudo_one_mixing_per_block() -> None:
    # the determinant and p -> s maps only live in the per-block mixing layer
    num_blocks = 3
    layer = LGATrSlimPseudo(
        num_blocks=num_blocks,
        in_v_channels=1,
        out_v_channels=0,
        hidden_v_channels=8,
        in_s_channels=4,
        out_s_channels=2,
        hidden_s_channels=8,
        num_heads=2,
        in_p_channels=2,
        out_p_channels=1,
        hidden_p_channels=4,
    )
    mixings = [m for m in layer.modules() if isinstance(m, SlimPseudoMixing)]
    dets = [m for m in layer.modules() if isinstance(m, VectorToPseudoscalar)]
    assert len(mixings) == len(dets) == num_blocks


@pytest.mark.parametrize("in_v,out_v,in_s,out_s,in_p,out_p,initialization", LINEAR_CASES)
def test_SlimPseudoLinear_matches_SlimLinear(
    in_v: int, out_v: int, in_s: int, out_s: int, in_p: int, out_p: int, initialization: str
) -> None:
    # the vector and scalar paths are exactly SlimLinear
    layer = SlimPseudoLinear(in_v, out_v, in_s, out_s, in_p, out_p, initialization=initialization)
    slim = SlimLinear(in_v, out_v, in_s, out_s, initialization=initialization)
    slim.load_state_dict(layer.linear_vs.state_dict())

    v = torch.randn(*BATCH_DIMS, 4, in_v)
    s = torch.randn(*BATCH_DIMS, in_s)
    p = torch.randn(*BATCH_DIMS, in_p)
    outputs_v, outputs_s, _ = layer(v, s, p)
    slim_v, slim_s = slim(v, s)
    torch.testing.assert_close(outputs_v, slim_v)
    torch.testing.assert_close(outputs_s, slim_s)


def test_LGATrSlimPseudo_split_norm_equivariance() -> None:
    layer = LGATrSlimPseudo(
        num_blocks=2,
        in_v_channels=3,
        out_v_channels=2,
        hidden_v_channels=16,
        in_s_channels=4,
        out_s_channels=2,
        hidden_s_channels=8,
        num_heads=2,
        in_p_channels=2,
        out_p_channels=1,
        hidden_p_channels=4,
        split_norm=True,
    )
    layer.eval()
    assert layer.blocks[0].norm1.split_norm
    assert layer.blocks[0].norm2.split_norm
    assert layer.blocks[0].norm_mix.split_norm
    assert layer.blocks[0].attention.norm.split_norm

    v = torch.randn(*BATCH_DIMS, 3, 4)
    s = torch.randn(*BATCH_DIMS, 4)
    p = torch.randn(*BATCH_DIMS, 2)

    check_equivariance(
        layer, batch_dims=(*BATCH_DIMS, 3), fn_kwargs=dict(scalars=s, pseudoscalars=p), **TOLERANCES
    )
    check_parity_equivariance(layer, v, s, p, vector_dim=-1, **TOLERANCES)


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


@pytest.mark.parametrize("in_v,out_v,in_s,out_s,in_p,out_p", CHANNELS[:4])
def test_SlimPseudoLinear_initialization(
    in_v: int,
    out_v: int,
    in_s: int,
    out_s: int,
    in_p: int,
    out_p: int,
    var_tolerance: float = 10.0,
) -> None:
    # SlimPseudoLinear maps unit-variance inputs to roughly unit-variance outputs.
    layer = SlimPseudoLinear(
        in_v_channels=in_v,
        out_v_channels=out_v,
        in_s_channels=in_s,
        out_s_channels=out_s,
        in_p_channels=in_p,
        out_p_channels=out_p,
    )

    inputs_v = torch.randn(100, 4, in_v)
    inputs_s = torch.randn(100, in_s)
    inputs_p = torch.randn(100, in_p)
    outputs_v, outputs_s, outputs_p = layer(inputs_v, inputs_s, inputs_p)

    v_mean = outputs_v.detach().to(torch.float64).mean(dim=(0, -1))
    v_var = outputs_v.detach().to(torch.float64).var(dim=(0, -1))
    target_mean = torch.zeros_like(v_mean)
    target_var = torch.ones_like(v_var) / 3.0
    assert torch.all(v_mean > target_mean - 0.3)
    assert torch.all(v_mean < target_mean + 0.3)
    assert torch.all(v_var > target_var / var_tolerance)
    assert torch.all(v_var < target_var * var_tolerance)

    if out_s > 0 and in_s > 0:
        s_mean = outputs_s.detach().to(torch.float64).mean().item()
        s_var = outputs_s.detach().to(torch.float64).var().item()

        assert -1.0 < s_mean < 1.0
        assert 1.0 / var_tolerance < s_var < 1.0 * var_tolerance

    if out_p > 0:
        assert torch.isfinite(outputs_p).all()


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


@pytest.mark.parametrize("v_channels,s_channels,p_channels", [(32, 4, 3), (16, 8, 2)])
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
    "v_channels,s_channels,p_channels,num_heads", [(32, 4, 3, 1), (16, 8, 2, 4)]
)
@pytest.mark.parametrize("dropout_prob", [None, 0.5])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_SlimPseudoBlock_equivariance(
    v_channels: int,
    s_channels: int,
    p_channels: int,
    num_heads: int,
    dropout_prob: float | None,
    norm_elementwise_affine: bool,
) -> None:
    layer = SlimPseudoBlock(
        v_channels=v_channels,
        s_channels=s_channels,
        p_channels=p_channels,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        norm_elementwise_affine=norm_elementwise_affine,
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


@pytest.mark.parametrize(
    "in_v_channels,in_s_channels,in_p_channels,out_v_channels,out_s_channels,out_p_channels",
    [
        (4, 3, 2, 9, 2, 1),
        (2, 9, 1, 0, 3, 2),
        (3, 5, 4, 7, 0, 3),
        (8, 3, 2, 0, 0, 5),
    ],
)
@pytest.mark.parametrize(
    "hidden_v_channels,hidden_s_channels,hidden_p_channels,num_heads",
    [(32, 4, 3, 1), (16, 8, 2, 4)],
)
@pytest.mark.parametrize("num_blocks,checkpoint_blocks", [(1, False), (2, True)])
def test_LGATrSlimPseudo_equivariance(
    in_v_channels: int,
    in_s_channels: int,
    in_p_channels: int,
    out_v_channels: int,
    out_s_channels: int,
    out_p_channels: int,
    hidden_v_channels: int,
    hidden_s_channels: int,
    hidden_p_channels: int,
    num_heads: int,
    num_blocks: int,
    checkpoint_blocks: bool,
) -> None:
    # The full network preserves shapes and is SO(1, 3)-equivariant and parity-covariant at eval
    # time. Built through LGATrSlim to exercise the dispatch as well.
    layer = LGATrSlim(
        num_blocks=num_blocks,
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        hidden_v_channels=hidden_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        num_heads=num_heads,
        in_p_channels=in_p_channels,
        out_p_channels=out_p_channels,
        hidden_p_channels=hidden_p_channels,
        dropout_prob=0.5,
        checkpoint_blocks=checkpoint_blocks,
    )
    assert type(layer) is LGATrSlimPseudo
    layer.eval()
    s = torch.randn(*BATCH_DIMS, in_s_channels)
    p = torch.randn(*BATCH_DIMS, in_p_channels)
    v = torch.randn(*BATCH_DIMS, in_v_channels, 4)
    outputs_v, outputs_s, outputs_p = layer(v, s, p)
    assert outputs_v.shape == (*BATCH_DIMS, out_v_channels, 4)
    assert outputs_s.shape == (*BATCH_DIMS, out_s_channels)
    assert outputs_p.shape == (*BATCH_DIMS, out_p_channels)

    check_equivariance(
        layer,
        batch_dims=(*BATCH_DIMS, in_v_channels),
        fn_kwargs=dict(scalars=s, pseudoscalars=p),
        **TOLERANCES,
    )
    check_parity_equivariance(layer, v, s, p, vector_dim=-1, **TOLERANCES)


@pytest.mark.parametrize("out_v_channels", [0, 2])
def test_LGATrSlimPseudo_gradients_flow(out_v_channels: int) -> None:
    # Every trainable parameter receives a gradient; out_v_channels=0 is the tagging setting.
    layer = LGATrSlimPseudo(
        num_blocks=2,
        in_v_channels=3,
        out_v_channels=out_v_channels,
        hidden_v_channels=8,
        in_s_channels=4,
        out_s_channels=2,
        hidden_s_channels=8,
        num_heads=2,
        in_p_channels=2,
        out_p_channels=1,
        hidden_p_channels=4,
    )
    v = torch.randn(*BATCH_DIMS, 3, 4)
    s = torch.randn(*BATCH_DIMS, 4)
    p = torch.randn(*BATCH_DIMS, 2)

    outputs_v, outputs_s, outputs_p = layer(v, s, p)
    (outputs_v.sum() + outputs_s.sum() + outputs_p.sum()).backward()

    for name, param in layer.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_LGATrSlimPseudo_pseudoscalars_default_to_empty() -> None:
    # pseudoscalars may be omitted when the model expects no input pseudoscalar channels.
    layer = LGATrSlimPseudo(
        num_blocks=1,
        in_v_channels=3,
        out_v_channels=0,
        hidden_v_channels=8,
        in_s_channels=4,
        out_s_channels=2,
        hidden_s_channels=8,
        num_heads=2,
        in_p_channels=0,
        out_p_channels=1,
        hidden_p_channels=4,
    )
    layer.eval()
    v = torch.randn(*BATCH_DIMS, 3, 4)
    s = torch.randn(*BATCH_DIMS, 4)

    outputs_v, outputs_s, outputs_p = layer(v, s)
    assert outputs_v.shape == (*BATCH_DIMS, 0, 4)
    assert outputs_s.shape == (*BATCH_DIMS, 2)
    assert outputs_p.shape == (*BATCH_DIMS, 1)


def test_LGATrSlimPseudo_requires_scalars() -> None:
    layer = LGATrSlimPseudo(
        num_blocks=1,
        in_v_channels=3,
        out_v_channels=2,
        hidden_v_channels=8,
        in_s_channels=4,
        out_s_channels=2,
        hidden_s_channels=8,
        num_heads=2,
        in_p_channels=2,
        out_p_channels=1,
        hidden_p_channels=4,
    )
    with pytest.raises(ValueError):
        layer(torch.randn(*BATCH_DIMS, 3, 4), None, torch.randn(*BATCH_DIMS, 2))


def test_LGATrSlimPseudo_compiled() -> None:
    layer = LGATrSlimPseudo(
        num_blocks=1,
        in_v_channels=4,
        out_v_channels=9,
        hidden_v_channels=16,
        in_s_channels=3,
        out_s_channels=2,
        hidden_s_channels=8,
        num_heads=4,
        in_p_channels=2,
        out_p_channels=1,
        hidden_p_channels=3,
        compile=True,
        compile_kwargs={"dynamic": True},
    )
    layer.eval()
    s = torch.randn(*BATCH_DIMS, 3)
    p = torch.randn(*BATCH_DIMS, 2)
    v = torch.randn(*BATCH_DIMS, 4, 4)
    outputs_v, outputs_s, outputs_p = layer(v, s, p)
    assert outputs_v.shape == (*BATCH_DIMS, 9, 4)
    assert outputs_s.shape == (*BATCH_DIMS, 2)
    assert outputs_p.shape == (*BATCH_DIMS, 1)

    check_parity_equivariance(layer, v, s, p, vector_dim=-1, **TOLERANCES)
