"""This module provides efficiency improvements over torch's einsum through caching."""

import functools
from collections.abc import Sequence

import opt_einsum
import torch


def custom_einsum(equation: str, *operands: torch.Tensor, path: list[int]) -> torch.Tensor:
    """Computes einsum with a custom contraction order."""

    # Justification: For the sake of performance, we need direct access to torch's private methods.

    return torch._VF.einsum(equation, operands, path=path)


def cached_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """Computes einsum with a cached optimal contraction.

    Inspired by upstream
    https://github.com/pytorch/pytorch/blob/v1.13.0/torch/functional.py#L381.
    """
    op_shape = tuple(op.shape for op in operands)
    path = _get_cached_path_for_equation_and_shapes(equation=equation, op_shape=op_shape)

    return custom_einsum(equation, *operands, path=path)


@functools.cache
def _get_cached_path_for_equation_and_shapes(
    equation: str, op_shape: Sequence[torch.Tensor]
) -> list[int]:
    """Provides caching of optimal path."""
    tupled_path = opt_einsum.contract_path(equation, *op_shape, optimize="optimal", shapes=True)[0]

    return [item for pair in tupled_path for item in pair]
