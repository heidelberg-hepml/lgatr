from typing import Iterator, Tuple

import opt_einsum
import pytest
import torch

from lgatr.utils.einsum import cached_einsum

_DIM = 5


@pytest.fixture(name="einsum_eq")
def einsum_eq_fixture() -> str:
    """Provides a non-trivial einsum equation."""
    # Example equation from https://optimized-einsum.readthedocs.io/en/stable/index.html
    return "pi,qj,ijkl,rk,sl->pqrs"


@pytest.fixture(name="example_operands")
def example_operands_fixture() -> Tuple[torch.Tensor, ...]:
    """Provides tensors for a non-trivial einsum equation."""
    # Example tensors from https://optimized-einsum.readthedocs.io/en/stable/index.html
    I = torch.rand(_DIM, _DIM, _DIM, _DIM)
    C = torch.rand(_DIM, _DIM)
    return (C, C, I, C, C)


def test_opt_einsum_shape() -> None:
    """Verifies contract_path can be called with shapes.

    Note that we test upstream functionality here, which should not be necessary.
    However, with the latest release (3.3.0), this failed.

    NB: Test does not fail on every example (e.g., does not fail on `einsum_eq_fixture`).
    We use an equation from a failing test (test_pin).
    """
    einsum_eq = "i j k, ... j, ... k -> ... i"
    sizes = (_DIM, _DIM, _DIM)
    operands = tuple(torch.rand(sizes) for _ in sizes)
    op_shape = tuple(x.size() for x in operands)
    assert (
        opt_einsum.contract_path(einsum_eq, *operands, optimize="optimal")[0]
        == opt_einsum.contract_path(
            einsum_eq, *op_shape, optimize="optimal", shapes=True
        )[0]
    )


def test_e2e_cached_path(einsum_eq: str, example_operands: Tuple[torch.Tensor]) -> None:
    """Checks that torch.einsum and cached_einsum deliver the same values on a non-trivial einsum
    equation."""
    expected_result = torch.einsum(einsum_eq, *example_operands)
    result_with_uncached_path = cached_einsum(einsum_eq, *example_operands)
    result_with_cached_path = cached_einsum(einsum_eq, *example_operands)

    torch.testing.assert_close(expected_result, result_with_uncached_path)
    torch.testing.assert_close(expected_result, result_with_cached_path)
