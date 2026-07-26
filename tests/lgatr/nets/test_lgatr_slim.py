import pytest
import torch

from lgatr.nets.slim import LGATrSlim
from lgatr.nets.slim_layers import (
    SlimBlock,
    SlimDropout,
    SlimGLU,
    SlimLinear,
    SlimMLP,
    SlimRMSNorm,
    SlimSelfAttention,
)

from ...helpers.constants import BATCH_DIMS, COMPILE_SUPPORTED, TOLERANCES
from ...helpers.equivariance_noga import check_equivariance

CHANNELS = [
    (5, 1, 4, 2),
    (1, 4, 0, 2),
    (9, 3, 4, 0),
    (2, 7, 0, 0),
    (0, 1, 2, 3),
    (3, 0, 2, 3),
    (0, 0, 2, 3),
]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("dropout_prob", [0.0, 0.1, 0.5])
def test_SlimDropout_equivariance(batch_dims: list[int], dropout_prob: float) -> None:
    # Slim Dropout preserves shapes and is SO(1, 3)-equivariant at eval time.
    layer = SlimDropout(dropout_prob)

    v = torch.randn(*batch_dims[:-1], 4, batch_dims[-1])
    s = torch.randn(*batch_dims)

    # shape and determinism in train mode (same seed -> same output; exercises the actual
    # dropout path, which is skipped in eval mode)
    layer.train()
    torch.manual_seed(0)
    train_v1, train_s1 = layer(v, scalars=s)
    torch.manual_seed(0)
    train_v2, train_s2 = layer(v, scalars=s)
    assert train_v1.shape == v.shape
    assert train_s1.shape == s.shape
    torch.testing.assert_close(train_v1, train_v2, **TOLERANCES)
    torch.testing.assert_close(train_s1, train_s2, **TOLERANCES)

    # shape at eval time
    layer.eval()
    outputs_v, outputs_s = layer(v, scalars=s)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    # equivariance
    check_equivariance(
        layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), vector_dim=-2, **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("zero_channels", [None, "v", "s"])
def test_SlimRMSNorm_equivariance(batch_dims: list[int], zero_channels: str | None) -> None:
    # SlimRMSNorm preserves shapes and is SO(1, 3)-equivariant, including the zero-channel edge
    # cases where the affine weight is frozen (weight.numel() == 0).
    v_channels = 0 if zero_channels == "v" else batch_dims[-1]
    s_channels = 0 if zero_channels == "s" else batch_dims[-1]
    layer = SlimRMSNorm(v_channels, s_channels)
    if zero_channels == "v":
        assert not layer.weight_v.requires_grad
    if zero_channels == "s":
        assert not layer.weight_s.requires_grad

    # shape
    v = torch.randn(*batch_dims[:-1], 4, v_channels)
    s = torch.randn(*batch_dims[:-1], s_channels)
    outputs_v, outputs_s = layer(v, scalars=s)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    # equivariance
    check_equivariance(
        layer,
        batch_dims=[*batch_dims[:-1], v_channels],
        fn_kwargs=dict(scalars=s),
        vector_dim=-2,
        **TOLERANCES,
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("nonlinearity", ["relu", "sigmoid", "tanh", "gelu", "silu"])
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels", CHANNELS)
def test_SlimGLU_equivariance(
    batch_dims: list[int],
    nonlinearity: str,
    in_v_channels: int,
    out_v_channels: int,
    in_s_channels: int,
    out_s_channels: int,
) -> None:
    # SlimGLU produces the right output shapes and is SO(1, 3)-equivariant.
    layer = SlimGLU(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        nonlinearity=nonlinearity,
    )
    s = torch.randn(*batch_dims, in_s_channels)
    v = torch.randn(*batch_dims, 4, in_v_channels)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == v.shape[:-2] + (4, out_v_channels)
    assert outputs_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(
        layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), vector_dim=-2, **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels", CHANNELS)
@pytest.mark.parametrize("initialization", ["default", "small"])
def test_SlimLinear_equivariance(
    batch_dims: list[int],
    in_v_channels: int,
    out_v_channels: int,
    in_s_channels: int,
    out_s_channels: int,
    initialization: str,
) -> None:
    # Slim Linear produces the right output shapes and is SO(1, 3)-equivariant.
    layer = SlimLinear(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        initialization=initialization,
    )
    s = torch.randn(*batch_dims, in_s_channels)
    v = torch.randn(*batch_dims, 4, in_v_channels)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == v.shape[:-2] + (4, out_v_channels)
    assert outputs_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(
        layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), vector_dim=-2, **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", [(100,)])
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels", CHANNELS[:4])
def test_SlimLinear_initialization(
    batch_dims: tuple[int, ...],
    in_v_channels: int,
    out_v_channels: int,
    in_s_channels: int,
    out_s_channels: int,
    var_tolerance: float = 10.0,
) -> None:
    # Slim Linear maps unit-variance inputs to roughly unit-variance outputs.
    layer = SlimLinear(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
    )

    inputs_v = torch.randn(*batch_dims, 4, in_v_channels)
    inputs_s = torch.randn(*batch_dims, in_s_channels)
    outputs_v, outputs_s = layer(inputs_v, inputs_s)

    v_mean = outputs_v.cpu().detach().to(torch.float64).mean(dim=(0, -1))
    v_var = outputs_v.cpu().detach().to(torch.float64).var(dim=(0, -1))
    target_mean = torch.zeros_like(v_mean)
    target_var = torch.ones_like(v_var) / 3.0
    assert torch.all(v_mean > target_mean - 0.3)
    assert torch.all(v_mean < target_mean + 0.3)
    assert torch.all(v_var > target_var / var_tolerance)
    assert torch.all(v_var < target_var * var_tolerance)

    if out_s_channels > 0 and in_s_channels > 0:
        s_mean = outputs_s.cpu().detach().to(torch.float64).mean().item()
        s_var = outputs_s.cpu().detach().to(torch.float64).var().item()

        assert -1.0 < s_mean < 1.0
        assert 1.0 / var_tolerance < s_var < 1.0 * var_tolerance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("v_channels,s_channels", [(24, 14)])
@pytest.mark.parametrize("num_heads,attn_ratio", [(2, 1), (1, 2)])
def test_SlimSelfAttention_equivariance(
    batch_dims: list[int],
    v_channels: int,
    s_channels: int,
    num_heads: int,
    attn_ratio: int,
) -> None:
    # Slim SlimSelfAttention preserves shapes and is SO(1, 3)-equivariant.
    layer = SlimSelfAttention(
        v_channels=v_channels,
        s_channels=s_channels,
        num_heads=num_heads,
        attn_ratio=attn_ratio,
    )
    s = torch.randn(*batch_dims, s_channels)

    v = torch.randn(*batch_dims, 4, v_channels)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    batch_dims = batch_dims + [v_channels]
    check_equivariance(
        layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), vector_dim=-2, **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("v_channels,s_channels", [(32, 4), (16, 8)])
@pytest.mark.parametrize("mlp_ratio,num_layers", [(1, 2), (2, 2), (1, 3)])
def test_SlimMLP_equivariance(
    batch_dims: list[int],
    v_channels: int,
    s_channels: int,
    mlp_ratio: int,
    num_layers: int,
) -> None:
    # Slim MLP is SO(1, 3)-equivariant.
    layer = SlimMLP(
        v_channels=v_channels,
        s_channels=s_channels,
        mlp_ratio=mlp_ratio,
        num_layers=num_layers,
    )
    s = torch.randn(*batch_dims, s_channels)
    batch_dims = batch_dims + [v_channels]

    # equivariance
    check_equivariance(
        layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), vector_dim=-2, **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("v_channels,s_channels,num_heads", [(32, 4, 1), (16, 8, 4)])
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.5])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_SlimBlock_equivariance(
    batch_dims: list[int],
    v_channels: int,
    s_channels: int,
    num_heads: int,
    dropout_prob: float | None,
    norm_elementwise_affine: bool,
) -> None:
    # SlimBlock is SO(1, 3)-equivariant at eval time.
    layer = SlimBlock(
        v_channels=v_channels,
        s_channels=s_channels,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        norm_elementwise_affine=norm_elementwise_affine,
    )
    layer.eval()
    s = torch.randn(*batch_dims, s_channels)
    batch_dims = batch_dims + [v_channels]

    # equivariance
    check_equivariance(
        layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), vector_dim=-2, **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "in_v_channels,in_s_channels,out_v_channels,out_s_channels",
    [
        (4, 3, 9, 2),
        (2, 9, 0, 3),
        (3, 5, 7, 0),
        (8, 3, 0, 0),
    ],
)
@pytest.mark.parametrize("hidden_v_channels,hidden_s_channels,num_heads", [(32, 4, 1), (16, 8, 4)])
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.5])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_LGATrSlim_equivariance(
    batch_dims: list[int],
    in_v_channels: int,
    in_s_channels: int,
    out_v_channels: int,
    out_s_channels: int,
    hidden_v_channels: int,
    hidden_s_channels: int,
    num_heads: int,
    num_blocks: int,
    dropout_prob: float | None,
    checkpoint_blocks: bool,
    norm_elementwise_affine: bool,
) -> None:
    # LGATrSlim (full network) preserves shapes and is SO(1, 3)-equivariant at eval time.
    layer = LGATrSlim(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        hidden_v_channels=hidden_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        checkpoint_blocks=checkpoint_blocks,
        norm_elementwise_affine=norm_elementwise_affine,
    )
    layer.eval()
    s = torch.randn(*batch_dims, in_s_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert outputs_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.skipif(not COMPILE_SUPPORTED, reason="torch.compile is unavailable")
@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "in_v_channels,in_s_channels,out_v_channels,out_s_channels", [(4, 3, 9, 2)]
)
@pytest.mark.parametrize(
    "hidden_v_channels,hidden_s_channels,num_heads,num_blocks", [(16, 8, 4, 1)]
)
def test_LGATrSlim_equivariance_compiled(
    batch_dims: list[int],
    in_v_channels: int,
    in_s_channels: int,
    out_v_channels: int,
    out_s_channels: int,
    hidden_v_channels: int,
    hidden_s_channels: int,
    num_heads: int,
    num_blocks: int,
    compile: bool = True,
) -> None:
    # torch.compile-wrapped LGATrSlim still preserves shapes and SO(1, 3)-equivariance.
    layer = LGATrSlim(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        hidden_v_channels=hidden_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        compile=compile,
    )
    layer.eval()
    s = torch.randn(*batch_dims, in_s_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert outputs_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)
