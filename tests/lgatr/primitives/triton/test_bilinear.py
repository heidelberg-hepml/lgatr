"""Triton-backend tests for bilinear primitives."""

import pytest
import torch

from lgatr.primitives.bilinear import geometric_product
from lgatr.primitives.config import PrimitivesConfig
from lgatr.primitives.triton import _HAVE_TRITON
from tests.helpers import BATCH_DIMS, TOLERANCES

_GPU_TRITON_REQUIRED = pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAVE_TRITON),
    reason="requires CUDA + triton",
)
_TRITON_DTYPE_TOLERANCES = {
    torch.float32: dict(atol=1e-3, rtol=1e-4),
    torch.float16: dict(atol=5e-3, rtol=5e-3),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}


@_GPU_TRITON_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_triton_dense_equivalence(
    batch_dims: list[int], dtype: torch.dtype
) -> None:
    # The Triton path agrees with the dense path within dtype-appropriate tolerances.
    x = torch.randn(*batch_dims, 16, device="cuda", dtype=dtype)
    y = torch.randn(*batch_dims, 16, device="cuda", dtype=dtype)
    out_dense = geometric_product(x, y, config=PrimitivesConfig(triton=False))
    out_triton = geometric_product(x, y, config=PrimitivesConfig(triton=True))
    torch.testing.assert_close(out_triton, out_dense, **_TRITON_DTYPE_TOLERANCES[dtype])


@_GPU_TRITON_REQUIRED
@pytest.mark.parametrize(
    "x_batch,y_batch",
    [
        ((3, 16), (16,)),
        ((16,), (3, 16)),
        ((4, 1, 16), (3, 16)),
        ((2, 1, 5, 16), (3, 1, 16)),
    ],
)
def test_geometric_product_triton_dense_equivalence_broadcasting(
    x_batch: tuple[int, ...], y_batch: tuple[int, ...]
) -> None:
    # The Triton path preserves the broadcasting semantics of the dense path.
    x = torch.randn(*x_batch, device="cuda", dtype=torch.float32)
    y = torch.randn(*y_batch, device="cuda", dtype=torch.float32)
    out_dense = geometric_product(x, y, config=PrimitivesConfig(triton=False))
    out_triton = geometric_product(x, y, config=PrimitivesConfig(triton=True))
    torch.testing.assert_close(out_triton, out_dense, **TOLERANCES)


@_GPU_TRITON_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_geometric_product_triton_grad(dtype: torch.dtype) -> None:
    # Triton fwd+bwd produces gradients matching autograd through the dense path.
    x = torch.randn(4, 8, 16, device="cuda", dtype=dtype)
    y = torch.randn(4, 8, 16, device="cuda", dtype=dtype)
    go = torch.randn(4, 8, 16, device="cuda", dtype=dtype)

    x_t = x.clone().requires_grad_(True)
    y_t = y.clone().requires_grad_(True)
    geometric_product(x_t, y_t, config=PrimitivesConfig(triton=True)).backward(go)

    x_d = x.clone().requires_grad_(True)
    y_d = y.clone().requires_grad_(True)
    geometric_product(x_d, y_d, config=PrimitivesConfig(triton=False)).backward(go)

    torch.testing.assert_close(x_t.grad, x_d.grad, **_TRITON_DTYPE_TOLERANCES[dtype])
    torch.testing.assert_close(y_t.grad, y_d.grad, **_TRITON_DTYPE_TOLERANCES[dtype])
