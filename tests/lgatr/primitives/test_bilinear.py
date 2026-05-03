"""Unit tests of bilinear primitives."""

import pytest
import torch

from lgatr.primitives.bilinear import geometric_product
from lgatr.primitives.config import gatr_config
from tests.helpers import (
    BATCH_DIMS,
    TOLERANCES,
    check_consistence_with_geometric_product,
    check_pin_equivariance,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_correctness(batch_dims: list[int]) -> None:
    # geometric_product matches the clifford-library reference implementation.
    check_consistence_with_geometric_product(geometric_product, batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_equivariance(batch_dims: list[int]) -> None:
    # geometric_product is Pin-equivariant in both arguments.
    check_pin_equivariance(geometric_product, 2, batch_dims=[batch_dims] * 2, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_sparse_dense_equivalence(batch_dims: list[int]) -> None:
    # The sparse path agrees with the dense path within TOLERANCES on shared inputs.
    x = torch.randn(*batch_dims, 16)
    y = torch.randn(*batch_dims, 16)
    try:
        gatr_config.sparse = False
        out_dense = geometric_product(x, y)
        gatr_config.sparse = True
        out_sparse = geometric_product(x, y)
    finally:
        gatr_config.sparse = False
    torch.testing.assert_close(out_sparse, out_dense, **TOLERANCES)


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
    # The sparse path must preserve the broadcasting semantics of the dense path.
    x = torch.randn(*x_batch)
    y = torch.randn(*y_batch)
    try:
        gatr_config.sparse = False
        out_dense = geometric_product(x, y)
        gatr_config.sparse = True
        out_sparse = geometric_product(x, y)
    finally:
        gatr_config.sparse = False
    torch.testing.assert_close(out_sparse, out_dense, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_sparse_correctness(batch_dims: list[int]) -> None:
    # The sparse path matches the clifford-library reference implementation.
    gatr_config.sparse = True
    try:
        check_consistence_with_geometric_product(geometric_product, batch_dims, **TOLERANCES)
    finally:
        gatr_config.sparse = False


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_sparse_equivariance(batch_dims: list[int]) -> None:
    # The sparse path is Pin-equivariant in both arguments.
    gatr_config.sparse = True
    try:
        check_pin_equivariance(geometric_product, 2, batch_dims=[batch_dims] * 2, **TOLERANCES)
    finally:
        gatr_config.sparse = False
