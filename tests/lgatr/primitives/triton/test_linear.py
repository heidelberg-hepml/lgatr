"""Triton-backend tests for linear primitives."""

import pytest
import torch

from lgatr.primitives.config import gatr_config
from lgatr.primitives.linear import equi_linear
from lgatr.primitives.triton import _HAVE_TRITON

_GPU_TRITON_REQUIRED = pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAVE_TRITON),
    reason="requires CUDA + triton",
)
_TRITON_DTYPE_TOLERANCES = {
    torch.float32: dict(atol=1e-3, rtol=1e-4),
    torch.float16: dict(atol=5e-3, rtol=5e-3),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}


def test_triton_module_imports_on_cpu() -> None:
    # Importing the triton subpackage must succeed even on CPU-only machines.
    from lgatr.primitives.triton import _HAVE_TRITON as flag  # noqa: F401


@_GPU_TRITON_REQUIRED
@pytest.mark.parametrize("use_fully_connected_subgroup", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 5, 8, 11),  # small, tile-aligned
        (37, 64, 128),  # larger; non-power-of-two batch exercises the masking path
    ],
)
def test_equi_linear_triton_dense_equivalence(
    shape: tuple[int, ...], use_fully_connected_subgroup: bool, dtype: torch.dtype
) -> None:
    # The Triton path agrees with the dense path within dtype-appropriate tolerances.
    gatr_config.use_fully_connected_subgroup = use_fully_connected_subgroup
    *batch, in_c, out_c = shape
    nb = gatr_config.num_pin_linear_basis_elements
    x = torch.randn(*batch, in_c, 16, device="cuda", dtype=dtype)
    coeffs = torch.randn(out_c, in_c, nb, device="cuda", dtype=dtype)
    try:
        gatr_config.triton = False
        out_dense = equi_linear(x, coeffs)
        gatr_config.triton = True
        out_triton = equi_linear(x, coeffs)
    finally:
        gatr_config.triton = False
        gatr_config.use_fully_connected_subgroup = True
    torch.testing.assert_close(out_triton, out_dense, **_TRITON_DTYPE_TOLERANCES[dtype])


@_GPU_TRITON_REQUIRED
@pytest.mark.parametrize("use_fully_connected_subgroup", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_equi_linear_triton_grad(use_fully_connected_subgroup: bool, dtype: torch.dtype) -> None:
    # Triton fwd+bwd produces gradients that match autograd through the dense path.
    gatr_config.use_fully_connected_subgroup = use_fully_connected_subgroup
    nb = gatr_config.num_pin_linear_basis_elements
    in_c, out_c = 8, 11
    x = torch.randn(4, in_c, 16, device="cuda", dtype=dtype)
    coeffs = torch.randn(out_c, in_c, nb, device="cuda", dtype=dtype)
    go = torch.randn(4, out_c, 16, device="cuda", dtype=dtype)
    try:
        gatr_config.triton = True
        x_t = x.clone().requires_grad_(True)
        c_t = coeffs.clone().requires_grad_(True)
        equi_linear(x_t, c_t).backward(go)

        gatr_config.triton = False
        x_d = x.clone().requires_grad_(True)
        c_d = coeffs.clone().requires_grad_(True)
        equi_linear(x_d, c_d).backward(go)
    finally:
        gatr_config.triton = False
        gatr_config.use_fully_connected_subgroup = True
    torch.testing.assert_close(x_t.grad, x_d.grad, **_TRITON_DTYPE_TOLERANCES[dtype])
    torch.testing.assert_close(c_t.grad, c_d.grad, **_TRITON_DTYPE_TOLERANCES[dtype])
