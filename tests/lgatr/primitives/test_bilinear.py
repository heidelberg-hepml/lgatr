"""Unit tests of bilinear primitives."""

import pytest
import torch

from lgatr.primitives.bilinear import geometric_product
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import (
    BATCH_DIMS,
    TOLERANCES,
    check_consistence_with_geometric_product,
    check_pin_equivariance,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_correctness(batch_dims: list[int]) -> None:
    # geometric_product matches the clifford-library reference implementation.
    config = PrimitivesConfig()
    check_consistence_with_geometric_product(
        lambda x, y: geometric_product(x, y, config=config), batch_dims, **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_equivariance(batch_dims: list[int]) -> None:
    # geometric_product is Pin-equivariant in both arguments.
    config = PrimitivesConfig()
    check_pin_equivariance(
        geometric_product,
        2,
        fn_kwargs=dict(config=config),
        batch_dims=[batch_dims] * 2,
        **TOLERANCES,
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_sparse_dense_equivalence(batch_dims: list[int]) -> None:
    # The sparse path agrees with the dense path within TOLERANCES on shared inputs.
    x = torch.randn(*batch_dims, 16)
    y = torch.randn(*batch_dims, 16)
    out_dense = geometric_product(x, y, config=PrimitivesConfig(sparse=False))
    out_sparse = geometric_product(x, y, config=PrimitivesConfig(sparse=True))
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
    out_dense = geometric_product(x, y, config=PrimitivesConfig(sparse=False))
    out_sparse = geometric_product(x, y, config=PrimitivesConfig(sparse=True))
    torch.testing.assert_close(out_sparse, out_dense, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_sparse_correctness(batch_dims: list[int]) -> None:
    # The sparse path matches the clifford-library reference implementation.
    config = PrimitivesConfig(sparse=True)
    check_consistence_with_geometric_product(
        lambda x, y: geometric_product(x, y, config=config), batch_dims, **TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_sparse_equivariance(batch_dims: list[int]) -> None:
    # The sparse path is Pin-equivariant in both arguments.
    config = PrimitivesConfig(sparse=True)
    check_pin_equivariance(
        geometric_product,
        2,
        fn_kwargs=dict(config=config),
        batch_dims=[batch_dims] * 2,
        **TOLERANCES,
    )
