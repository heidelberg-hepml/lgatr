"""Utility function to test callables for consistency with clifford-algebra references."""

from collections.abc import Callable
from math import prod

import clifford
import torch

from .clifford import LAYOUT, mv_list_to_tensor


def _sample_list_of_mv(batch_dims: tuple | list) -> list[clifford.MultiVector]:
    """Sample a list of multivectors of total length ``prod(batch_dims)``."""
    total_batchsize = 1 if not batch_dims else prod(batch_dims)
    xs = clifford.randomMV(layout=LAYOUT, n=total_batchsize)
    if total_batchsize == 1:  # Dealing with inconsistency of clifford.randomMV
        xs = [xs]
    return xs


def check_against_clifford(
    function: Callable,
    reference: Callable,
    batch_dims: tuple | list = (1,),
    num_args: int = 1,
    **tolerances,
) -> None:
    """Check that ``function`` matches a clifford-library ``reference`` on random multivectors.

    Parameters
    ----------
    function
        Function under test, taking ``num_args`` multivector tensors of shape ``(..., 16)``.
    reference
        Equivalent operation on ``clifford.MultiVector`` objects, e.g. ``operator.mul``.
    batch_dims
        Batch shape of the multivector inputs.
    num_args
        Number of multivector arguments.
    **tolerances
        Optional tolerance keyword arguments forwarded to :func:`torch.testing.assert_close`.
    """
    inputs = [_sample_list_of_mv(batch_dims) for _ in range(num_args)]

    expected = [reference(*mvs) for mvs in zip(*inputs, strict=True)]
    expected = mv_list_to_tensor(expected, batch_dims)
    outputs = function(*[mv_list_to_tensor(mvs, batch_dims) for mvs in inputs])

    torch.testing.assert_close(outputs, expected, **tolerances)
