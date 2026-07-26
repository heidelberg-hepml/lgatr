"""Unit tests of bilinear primitives."""

import operator

import pytest
import torch

from lgatr.primitives.bilinear import geometric_product
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, TOLERANCES, check_against_clifford, check_pin_equivariance


@pytest.mark.parametrize("sparse_gp", [False, True])
def test_geometric_product_correctness(sparse_gp: bool) -> None:
    # Both the dense and the sparse path match the clifford-library reference implementation.
    config = PrimitivesConfig(sparse_gp=sparse_gp)
    check_against_clifford(
        lambda x, y: geometric_product(x, y, config=config),
        operator.mul,
        BATCH_DIMS,
        num_args=2,
        **TOLERANCES,
    )


@pytest.mark.parametrize("sparse_gp", [False, True])
def test_geometric_product_equivariance(sparse_gp: bool) -> None:
    # geometric_product is Pin-equivariant in both arguments, on both paths.
    config = PrimitivesConfig(sparse_gp=sparse_gp)
    check_pin_equivariance(
        geometric_product,
        2,
        fn_kwargs=dict(config=config),
        batch_dims=[BATCH_DIMS] * 2,
        spin=False,
        **TOLERANCES,
    )


@pytest.mark.parametrize(
    "x_batch,y_batch",
    [
        ((3, 16), (16,)),
        ((16,), (3, 16)),
        ((4, 1, 16), (3, 16)),
        ((2, 1, 5, 16), (3, 1, 16)),
    ],
)
def test_geometric_product_sparse_dense_equivalence_broadcasting(
    x_batch: tuple[int, ...], y_batch: tuple[int, ...]
) -> None:
    # The sparse path must preserve the broadcasting semantics of the dense path, including in
    # the backward pass (where the sparse path un-broadcasts gradients back to input shapes).
    x = torch.randn(*x_batch)
    y = torch.randn(*y_batch)
    x_dense, y_dense = x.clone().requires_grad_(), y.clone().requires_grad_()
    x_sparse, y_sparse = x.clone().requires_grad_(), y.clone().requires_grad_()

    out_dense = geometric_product(x_dense, y_dense, config=PrimitivesConfig(sparse_gp=False))
    out_sparse = geometric_product(x_sparse, y_sparse, config=PrimitivesConfig(sparse_gp=True))
    torch.testing.assert_close(out_sparse, out_dense, **TOLERANCES)

    out_dense.sum().backward()
    out_sparse.sum().backward()
    assert x_sparse.grad.shape == x.shape
    assert y_sparse.grad.shape == y.shape
    torch.testing.assert_close(x_sparse.grad, x_dense.grad, **TOLERANCES)
    torch.testing.assert_close(y_sparse.grad, y_dense.grad, **TOLERANCES)
