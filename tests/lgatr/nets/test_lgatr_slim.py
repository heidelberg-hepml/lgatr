import pytest
import torch

from lgatr.nets.lgatr_slim import (
    MLP,
    Attention,
    Dropout,
    GatedLinearUnit,
    LGATrSlim,
    LGATrSlimBlock,
    Linear,
    RMSNorm,
    squared_norm,
)

from ...helpers.constants import BATCH_DIMS, TOLERANCES
from ...helpers.equivariance_noga import check_equivariance, check_invariance

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
def test_squared_norm_invariance(batch_dims):
    check_invariance(squared_norm, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("dropout_prob", [0.0, 0.1, 0.5])
def test_Dropout_equivariance(batch_dims, dropout_prob):
    layer = Dropout(dropout_prob)
    layer.eval()

    # shape
    v = torch.randn(*batch_dims, 4)
    s = torch.randn(*batch_dims)
    out_v, out_s = layer(v, scalars=s)
    assert out_v.shape == v.shape
    assert out_s.shape == s.shape

    # equivariance
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_RMSNorm_equivariance(batch_dims):
    layer = RMSNorm()

    # shape
    v = torch.randn(*batch_dims, 4)
    s = torch.randn(*batch_dims)
    out_v, out_s = layer(v, scalars=s)
    assert out_v.shape == v.shape
    assert out_s.shape == s.shape

    # equivariance
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("nonlinearity", ["relu", "sigmoid", "tanh", "gelu", "silu"])
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels", CHANNELS)
def test_GatedLinearUnit_equivariance(
    batch_dims, nonlinearity, in_v_channels, out_v_channels, in_s_channels, out_s_channels
):
    layer = GatedLinearUnit(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        nonlinearity=nonlinearity,
    )
    s = torch.randn(*batch_dims, in_s_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    out_v, out_s = layer(v, s)
    assert out_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert out_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels", CHANNELS)
@pytest.mark.parametrize("initialization", ["default", "small"])
def test_Linear_equivariance(
    batch_dims,
    in_v_channels,
    out_v_channels,
    in_s_channels,
    out_s_channels,
    initialization,
):
    layer = Linear(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        initialization=initialization,
    )
    s = torch.randn(*batch_dims, in_s_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    out_v, out_s = layer(v, s)
    assert out_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert out_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(100,)])
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels", CHANNELS[:4])
def test_Linear_initialization(
    batch_dims,
    in_v_channels,
    out_v_channels,
    in_s_channels,
    out_s_channels,
    var_tolerance=10.0,
):
    # Test that inputs with variance 1 are roughly mapped to outputs with variance 1
    layer = Linear(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
    )

    inputs_v = torch.randn(*batch_dims, in_v_channels, 4)
    inputs_s = torch.randn(*batch_dims, in_s_channels)
    outputs_v, outputs_s = layer(inputs_v, inputs_s)

    v_mean = outputs_v.cpu().detach().to(torch.float64).mean(dim=(0, 1))
    v_var = outputs_v.cpu().detach().to(torch.float64).var(dim=(0, 1))
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
@pytest.mark.parametrize("N", [3, 13])
@pytest.mark.parametrize("v_channels,s_channels", [(24, 14)])
@pytest.mark.parametrize("num_heads,attn_ratio", [(2, 1), (1, 2)])
def test_Attention_equivariance(
    batch_dims,
    N,
    v_channels,
    s_channels,
    num_heads,
    attn_ratio,
):
    layer = Attention(
        v_channels=v_channels,
        s_channels=s_channels,
        num_heads=num_heads,
        attn_ratio=attn_ratio,
    )
    s = torch.randn(*batch_dims, N, s_channels)

    v = torch.randn(*batch_dims, N, v_channels, 4)
    out_v, out_s = layer(v, s)
    assert out_v.shape == v.shape
    assert out_s.shape == s.shape

    batch_dims = batch_dims + [N, v_channels]
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("v_channels,s_channels", [(32, 4), (16, 8)])
@pytest.mark.parametrize("mlp_ratio,num_layers", [(1, 2), (2, 2), (1, 3)])
def test_MLP_equivariance(batch_dims, v_channels, s_channels, mlp_ratio, num_layers):
    layer = MLP(
        v_channels=v_channels,
        s_channels=s_channels,
        mlp_ratio=mlp_ratio,
        num_layers=num_layers,
    )
    s = torch.randn(*batch_dims, s_channels)
    batch_dims = batch_dims + [v_channels]

    # equivariance
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("v_channels,s_channels,num_heads", [(32, 4, 1), (16, 8, 4)])
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.5])
def test_LGATrSlimBlock_equivariance(
    batch_dims,
    v_channels,
    s_channels,
    num_heads,
    dropout_prob,
):
    layer = LGATrSlimBlock(
        v_channels=v_channels,
        s_channels=s_channels,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
    )
    layer.eval()
    s = torch.randn(*batch_dims, s_channels)
    batch_dims = batch_dims + [v_channels]

    # equivariance
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


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
def test_LGATrSlim_equivariance(
    batch_dims,
    in_v_channels,
    in_s_channels,
    out_v_channels,
    out_s_channels,
    hidden_v_channels,
    hidden_s_channels,
    num_heads,
    num_blocks,
    dropout_prob,
    checkpoint_blocks,
):
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
    )
    layer.eval()
    s = torch.randn(*batch_dims, in_s_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    out_v, out_s = layer(v, s)
    assert out_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert out_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "in_v_channels,in_s_channels,out_v_channels,out_s_channels", [(4, 3, 9, 2)]
)
@pytest.mark.parametrize(
    "hidden_v_channels,hidden_s_channels,num_heads,num_blocks", [(16, 8, 4, 1)]
)
def test_LGATrSlim_equivariance_compiled(
    batch_dims,
    in_v_channels,
    in_s_channels,
    out_v_channels,
    out_s_channels,
    hidden_v_channels,
    hidden_s_channels,
    num_heads,
    num_blocks,
    compile=True,
):
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
    out_v, out_s = layer(v, s)
    assert out_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert out_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)
