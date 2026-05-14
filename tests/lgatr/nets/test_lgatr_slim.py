import pytest
import torch

from lgatr.nets.lgatr_slim import (
    MLP,
    Dropout,
    GatedLinearUnit,
    LGATrSlim,
    LGATrSlimBlock,
    Linear,
    RMSNorm,
    SelfAttention,
)

from ...helpers.constants import BATCH_DIMS, TOLERANCES
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
def test_Dropout_equivariance(batch_dims: list[int], dropout_prob: float) -> None:
    # Slim Dropout preserves shapes and is SO(1, 3)-equivariant at eval time.
    layer = Dropout(dropout_prob)
    layer.eval()

    # shape
    v = torch.randn(*batch_dims, 4)
    s = torch.randn(*batch_dims)
    outputs_v, outputs_s = layer(v, scalars=s)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    # equivariance
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_RMSNorm_equivariance(batch_dims: list[int]) -> None:
    # RMSNorm preserves shapes and is SO(1, 3)-equivariant.
    layer = RMSNorm(batch_dims[-1], batch_dims[-1])

    # shape
    v = torch.randn(*batch_dims, 4)
    s = torch.randn(*batch_dims)
    outputs_v, outputs_s = layer(v, scalars=s)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    # equivariance
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("nonlinearity", ["relu", "sigmoid", "tanh", "gelu", "silu"])
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels", CHANNELS)
def test_GatedLinearUnit_equivariance(
    batch_dims: list[int],
    nonlinearity: str,
    in_v_channels: int,
    out_v_channels: int,
    in_s_channels: int,
    out_s_channels: int,
) -> None:
    # GatedLinearUnit produces the right output shapes and is SO(1, 3)-equivariant.
    layer = GatedLinearUnit(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        nonlinearity=nonlinearity,
    )
    s = torch.randn(*batch_dims, in_s_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert outputs_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels", CHANNELS)
@pytest.mark.parametrize("initialization", ["default", "small"])
def test_Linear_equivariance(
    batch_dims: list[int],
    in_v_channels: int,
    out_v_channels: int,
    in_s_channels: int,
    out_s_channels: int,
    initialization: str,
) -> None:
    # Slim Linear produces the right output shapes and is SO(1, 3)-equivariant.
    layer = Linear(
        in_v_channels=in_v_channels,
        out_v_channels=out_v_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        initialization=initialization,
    )
    s = torch.randn(*batch_dims, in_s_channels)
    v = torch.randn(*batch_dims, in_v_channels, 4)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == v.shape[:-2] + (out_v_channels, 4)
    assert outputs_s.shape == s.shape[:-1] + (out_s_channels,)

    # equivariance
    batch_dims = batch_dims + [in_v_channels]
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(100,)])
@pytest.mark.parametrize("in_v_channels,out_v_channels,in_s_channels,out_s_channels", CHANNELS[:4])
def test_Linear_initialization(
    batch_dims: tuple[int, ...],
    in_v_channels: int,
    out_v_channels: int,
    in_s_channels: int,
    out_s_channels: int,
    var_tolerance: float = 10.0,
) -> None:
    # Slim Linear maps unit-variance inputs to roughly unit-variance outputs.
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
@pytest.mark.parametrize("v_channels,s_channels", [(24, 14)])
@pytest.mark.parametrize("num_heads,attn_ratio", [(2, 1), (1, 2)])
def test_SelfAttention_equivariance(
    batch_dims: list[int],
    v_channels: int,
    s_channels: int,
    num_heads: int,
    attn_ratio: int,
) -> None:
    # Slim SelfAttention preserves shapes and is SO(1, 3)-equivariant.
    layer = SelfAttention(
        v_channels=v_channels,
        s_channels=s_channels,
        num_heads=num_heads,
        attn_ratio=attn_ratio,
    )
    s = torch.randn(*batch_dims, s_channels)

    v = torch.randn(*batch_dims, v_channels, 4)
    outputs_v, outputs_s = layer(v, s)
    assert outputs_v.shape == v.shape
    assert outputs_s.shape == s.shape

    batch_dims = batch_dims + [v_channels]
    check_equivariance(layer, batch_dims=batch_dims, fn_kwargs=dict(scalars=s), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("v_channels,s_channels", [(32, 4), (16, 8)])
@pytest.mark.parametrize("mlp_ratio,num_layers", [(1, 2), (2, 2), (1, 3)])
def test_MLP_equivariance(
    batch_dims: list[int],
    v_channels: int,
    s_channels: int,
    mlp_ratio: int,
    num_layers: int,
) -> None:
    # Slim MLP is SO(1, 3)-equivariant.
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
    batch_dims: list[int],
    v_channels: int,
    s_channels: int,
    num_heads: int,
    dropout_prob: float | None,
) -> None:
    # LGATrSlimBlock is SO(1, 3)-equivariant at eval time.
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
