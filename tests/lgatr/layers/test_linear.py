import pytest
import torch

from lgatr.layers.linear import EquiLinear
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", [(100,)])
@pytest.mark.parametrize("in_mv_channels, out_mv_channels", [(200, 5), (16, 16), (5, 200)])
@pytest.mark.parametrize("in_s_channels, out_s_channels", [(0, 0), (0, 100), (100, 0), (32, 32)])
@pytest.mark.parametrize(
    "initialization", ["default", "small", "unit_scalar", "almost_unit_scalar"]
)
@pytest.mark.parametrize("use_fully_connected_subgroup", [True, False])
def test_linear_layer_initialization(
    initialization: str,
    batch_dims: tuple[int, ...],
    in_mv_channels: int,
    out_mv_channels: int,
    in_s_channels: int,
    out_s_channels: int,
    use_fully_connected_subgroup: bool,
    var_tolerance: float = 10.0,
) -> None:
    # EquiLinear maps unit-variance inputs to roughly unit-variance outputs across channel sizes.
    primitives = PrimitivesConfig(use_fully_connected_subgroup=use_fully_connected_subgroup)

    # Create layer
    try:
        layer = EquiLinear(
            in_mv_channels,
            out_mv_channels,
            primitives,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            initialization=initialization,
        )
    # Some initialization schemes are not implemented when data is all-scalar. That's fine.
    except NotImplementedError as exc:
        print(exc)
        return

    # Inputs
    inputs_mv = torch.randn(*batch_dims, in_mv_channels, 16)
    inputs_s = torch.randn(*batch_dims, in_s_channels) if in_s_channels else None

    # Compute outputs
    outputs_mv, outputs_s = layer(inputs_mv, scalars=inputs_s)

    # Compute mean and variance of MV outputs
    mv_mean = outputs_mv[...].cpu().detach().to(torch.float64).mean(dim=(0, 1))
    mv_var = outputs_mv[...].cpu().detach().to(torch.float64).var(dim=(0, 1))

    print("Output multivector means and std by components:")
    for i, (mean_, var_) in enumerate(zip(mv_mean, mv_var, strict=False)):
        print(f"  Component {i}: mean = {mean_:.2f}, std = {var_**0.5:.2f}")

    # Check that the mean and variance agree with expectations
    if initialization == "default":
        target_mean = torch.zeros_like(mv_mean)
        target_var = torch.ones_like(mv_var) / 3.0  # Factor 3 comes from heuristics
    elif initialization == "small":
        target_mean = torch.zeros_like(mv_mean)
        target_var = 0.01 * torch.ones_like(mv_var) / 3.0
    elif initialization == "unit_scalar":
        target_mean = torch.zeros_like(mv_mean)
        target_mean[0] = 1.0
        if primitives.use_fully_connected_subgroup:
            target_mean[-1] = 1.0
        target_var = 0.01 * torch.ones_like(mv_var) / 3.0
    elif initialization == "almost_unit_scalar":
        target_mean = torch.zeros_like(mv_mean)
        target_mean[0] = 1.0
        if primitives.use_fully_connected_subgroup:
            target_mean[-1] = 1.0
        target_var = 0.25 * torch.ones_like(mv_var) / 3.0
    else:
        raise ValueError(initialization)

    assert torch.all(mv_mean > target_mean - 0.3)
    assert torch.all(mv_mean < target_mean + 0.3)
    assert torch.all(mv_var > target_var / var_tolerance)
    assert torch.all(mv_var < target_var * var_tolerance)

    # Same for scalar outputs
    if out_s_channels:
        s_mean = outputs_s[...].cpu().detach().to(torch.float64).mean().item()
        s_var = outputs_s[...].cpu().detach().to(torch.float64).var().item()

        print(f"Output scalar: mean = {s_mean:.2f}, std = {s_var**0.5:.2f}")
        assert -0.3 < s_mean < 0.3
        if initialization in {"default", "unit_scalar", "almost_unit_scalar"}:
            assert 1.0 / 3.0 / var_tolerance < s_var < 1.0 / 3.0 * var_tolerance
        else:
            assert 0.01 / 3.0 / var_tolerance < s_var < 0.01 / 3.0 * var_tolerance


@pytest.mark.parametrize("rescaling", [0.0, -2.0, 100.0])
@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_mv_channels", [9, 1])
@pytest.mark.parametrize("out_mv_channels", [7, 1])
@pytest.mark.parametrize("in_s_channels", [0, 3])
@pytest.mark.parametrize("out_s_channels", [0, 4])
def test_linear_layer_linearity(
    batch_dims: list[int],
    in_mv_channels: int,
    out_mv_channels: int,
    in_s_channels: int,
    out_s_channels: int,
    rescaling: float,
) -> None:
    # EquiLinear (no bias) is linear: f(x + c*y) == f(x) + c*f(y).
    layer = EquiLinear(
        in_mv_channels,
        out_mv_channels,
        PrimitivesConfig(),
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        bias=False,
    )

    # Inputs
    x_mv = torch.randn(*batch_dims, in_mv_channels, 16)
    y_mv = torch.randn(*batch_dims, in_mv_channels, 16)
    xy_mv = x_mv + rescaling * y_mv

    if in_s_channels:
        x_s = torch.randn(*batch_dims, in_s_channels)
        y_s = torch.randn(*batch_dims, in_s_channels)
        xy_s = x_s + rescaling * y_s
    else:
        x_s, y_s, xy_s = None, None, None

    # Compute outputs
    o_xy_mv, o_xy_s = layer(xy_mv, scalars=xy_s)
    o_x_mv, o_x_s = layer(x_mv, scalars=x_s)
    o_y_mv, o_y_s = layer(y_mv, scalars=y_s)

    # Check equality
    torch.testing.assert_close(o_xy_mv, o_x_mv + rescaling * o_y_mv, **TOLERANCES)

    if out_s_channels:
        torch.testing.assert_close(o_xy_s, o_x_s + rescaling * o_y_s, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_mv_channels", [9, 1])
@pytest.mark.parametrize("out_mv_channels", [7, 1])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("in_s_channels", [0, 3])
@pytest.mark.parametrize("out_s_channels", [0, 4])
@pytest.mark.parametrize("use_fully_connected_subgroup", [True, False])
def test_linear_layer_equivariance(
    batch_dims: list[int],
    in_mv_channels: int,
    out_mv_channels: int,
    in_s_channels: int,
    out_s_channels: int,
    bias: bool,
    use_fully_connected_subgroup: bool,
) -> None:
    # EquiLinear is Pin-equivariant for the full Lorentz group and the proper-orthochronous subgroup.
    primitives = PrimitivesConfig(use_fully_connected_subgroup=use_fully_connected_subgroup)

    layer = EquiLinear(
        in_mv_channels,
        out_mv_channels,
        primitives,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        bias=bias,
    )
    data_dims = tuple(list(batch_dims) + [in_mv_channels])
    scalars = torch.randn(*batch_dims, in_s_channels) if in_s_channels else None
    check_pin_equivariance(
        layer, 1, fn_kwargs=dict(scalars=scalars), batch_dims=data_dims, **TOLERANCES
    )
