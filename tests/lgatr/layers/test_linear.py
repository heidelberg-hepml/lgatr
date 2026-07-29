import pytest
import torch

from lgatr.layers.linear import EquiLinear
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance

# Fan-in-heavy, fan-out-heavy and square channel counts, covering the 0-scalar-in and
# 0-scalar-out edges. The unit-scalar initializations need a scalar stream, so the row without
# scalar inputs is paired only with the schemes that support it.
INIT_CHANNELS = [
    (200, 5, 100, 0, "default"),
    (200, 5, 100, 0, "small"),
    (200, 5, 100, 0, "unit_scalar"),
    (200, 5, 100, 0, "almost_unit_scalar"),
    (16, 16, 32, 32, "default"),
    (16, 16, 32, 32, "small"),
    (16, 16, 32, 32, "unit_scalar"),
    (16, 16, 32, 32, "almost_unit_scalar"),
    (5, 200, 0, 100, "default"),
    (5, 200, 0, 100, "small"),
    (16, 16, 0, 0, "default"),
    (16, 16, 0, 0, "small"),
]
CHANNELS = [(9, 7, 3, 4), (1, 1, 0, 0), (9, 1, 3, 0), (1, 7, 0, 4)]


@pytest.mark.parametrize("in_mv,out_mv,in_s,out_s,initialization", INIT_CHANNELS)
@pytest.mark.parametrize("subgroup", [True, False])
def test_linear_layer_initialization(
    in_mv: int,
    out_mv: int,
    in_s: int,
    out_s: int,
    initialization: str,
    subgroup: bool,
    var_tolerance: float = 10.0,
) -> None:
    # EquiLinear maps unit-variance inputs to roughly unit-variance outputs across channel sizes.
    primitives = PrimitivesConfig(subgroup=subgroup)
    layer = EquiLinear(
        in_mv,
        out_mv,
        primitives,
        in_s_channels=in_s,
        out_s_channels=out_s,
        initialization=initialization,
    )

    inputs_mv = torch.randn(100, in_mv, 16)
    inputs_s = torch.randn(100, in_s) if in_s else None
    outputs_mv, outputs_s = layer(inputs_mv, scalars=inputs_s)

    mv_mean = outputs_mv.detach().to(torch.float64).mean(dim=(0, 1))
    mv_var = outputs_mv.detach().to(torch.float64).var(dim=(0, 1))

    # Check that the mean and variance agree with expectations
    target_mean = torch.zeros_like(mv_mean)
    if initialization == "default":
        target_var = torch.ones_like(mv_var) / 3.0  # Factor 3 comes from heuristics
    elif initialization == "small":
        target_var = 0.01 * torch.ones_like(mv_var) / 3.0
    elif initialization == "unit_scalar":
        target_var = 0.01 * torch.ones_like(mv_var) / 3.0
    else:
        target_var = 0.25 * torch.ones_like(mv_var) / 3.0
    if initialization in {"unit_scalar", "almost_unit_scalar"}:
        target_mean[0] = 1.0
        if subgroup:
            target_mean[-1] = 1.0

    assert torch.all(mv_mean > target_mean - 0.3)
    assert torch.all(mv_mean < target_mean + 0.3)
    assert torch.all(mv_var > target_var / var_tolerance)
    assert torch.all(mv_var < target_var * var_tolerance)

    # Same for scalar outputs
    if out_s:
        s_mean = outputs_s.detach().to(torch.float64).mean().item()
        s_var = outputs_s.detach().to(torch.float64).var().item()

        assert -0.3 < s_mean < 0.3
        expected_s_var = 0.01 / 3.0 if initialization == "small" else 1.0 / 3.0
        assert expected_s_var / var_tolerance < s_var < expected_s_var * var_tolerance


@pytest.mark.parametrize("rescaling", [-2.0, 100.0])
@pytest.mark.parametrize("in_mv,out_mv,in_s,out_s", CHANNELS)
def test_linear_layer_linearity(
    in_mv: int, out_mv: int, in_s: int, out_s: int, rescaling: float
) -> None:
    # EquiLinear (no bias) is linear: f(x + c*y) == f(x) + c*f(y).
    layer = EquiLinear(
        in_mv,
        out_mv,
        PrimitivesConfig(),
        in_s_channels=in_s,
        out_s_channels=out_s,
        bias=False,
    )

    x_mv = torch.randn(*BATCH_DIMS, in_mv, 16)
    y_mv = torch.randn(*BATCH_DIMS, in_mv, 16)
    if in_s:
        x_s = torch.randn(*BATCH_DIMS, in_s)
        y_s = torch.randn(*BATCH_DIMS, in_s)
        xy_s = x_s + rescaling * y_s
    else:
        x_s, y_s, xy_s = None, None, None

    o_xy_mv, o_xy_s = layer(x_mv + rescaling * y_mv, scalars=xy_s)
    o_x_mv, o_x_s = layer(x_mv, scalars=x_s)
    o_y_mv, o_y_s = layer(y_mv, scalars=y_s)

    torch.testing.assert_close(o_xy_mv, o_x_mv + rescaling * o_y_mv, **TOLERANCES)
    if out_s:
        torch.testing.assert_close(o_xy_s, o_x_s + rescaling * o_y_s, **TOLERANCES)


@pytest.mark.parametrize("in_mv,out_mv,in_s,out_s", CHANNELS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("subgroup,spin", [(True, True), (False, False)])
def test_linear_layer_equivariance(
    in_mv: int, out_mv: int, in_s: int, out_s: int, bias: bool, subgroup: bool, spin: bool
) -> None:
    # EquiLinear is equivariant under the group it is built for: Spin for the proper-orthochronous
    # subgroup, the full Pin group (including reflections) otherwise.
    layer = EquiLinear(
        in_mv,
        out_mv,
        PrimitivesConfig(subgroup=subgroup),
        in_s_channels=in_s,
        out_s_channels=out_s,
        bias=bias,
    )
    data_dims = (*BATCH_DIMS, in_mv)
    scalars = torch.randn(*BATCH_DIMS, in_s) if in_s else None
    check_pin_equivariance(
        layer, 1, fn_kwargs=dict(scalars=scalars), batch_dims=data_dims, spin=spin, **TOLERANCES
    )
