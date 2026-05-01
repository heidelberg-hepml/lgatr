import pytest
import torch

from lgatr.utils.einsum import custom_einsum

_DIM = 5


@pytest.fixture(name="einsum_eq")
def einsum_eq_fixture() -> str:
    """Provides a non-trivial einsum equation."""
    # Example equation from https://optimized-einsum.readthedocs.io/en/stable/index.html
    return "pi,qj,ijkl,rk,sl->pqrs"


@pytest.fixture(name="example_operands")
def example_operands_fixture() -> tuple[torch.Tensor, ...]:
    """Provides tensors for a non-trivial einsum equation."""
    # Example tensors from https://optimized-einsum.readthedocs.io/en/stable/index.html
    A = torch.rand(_DIM, _DIM, _DIM, _DIM)
    C = torch.rand(_DIM, _DIM)
    return (C, C, A, C, C)


def test_custom_einsum_matches_torch(
    einsum_eq: str, example_operands: tuple[torch.Tensor, ...]
) -> None:
    """Checks that custom_einsum with a hardcoded contraction path matches torch.einsum."""
    # Path captured once via opt_einsum.contract_path(eq, *shapes, optimize="optimal", shapes=True)
    # for the fixture's shapes; flattened from [(0,2),(0,3),(0,2),(0,1)].
    hardcoded_path = [0, 2, 0, 3, 0, 2, 0, 1]

    expected_result = torch.einsum(einsum_eq, *example_operands)
    result = custom_einsum(einsum_eq, *example_operands, path=hardcoded_path)

    torch.testing.assert_close(expected_result, result)
