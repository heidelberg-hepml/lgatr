"""Utility functions to test callables for equivariance with respect to Pin(1, 3)."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .clifford import SlowRandomPinTransform


def get_first_output(outputs: Any) -> torch.Tensor:
    """Return the first output of a tuple, or ``outputs`` itself if not a tuple.

    Convenient for equivariance checks: primitives usually return a single multivector tensor
    while layers return two tensors. Wrapping the call site with this lets either be tested
    uniformly.
    """
    if isinstance(outputs, tuple):
        return outputs[0]

    return outputs


def check_pin_equivariance(
    function: Callable,
    num_multivector_args: int = 1,
    fn_kwargs: dict | None = None,
    batch_dims: tuple | list = (1,),
    spin: bool = True,
    rng: np.random.Generator | None = None,
    num_checks: int = 2,
    **kwargs,
) -> None:
    """Check whether a callable is equivariant w.r.t. the Pin(1, 3) or Spin(1, 3) group.

    The callable can have an arbitrary number of multivector inputs.

    Parameters
    ----------
    function
        Function to be tested for equivariance. The first ``num_multivector_args`` positional
        arguments must accept multivector tensors of shape ``(..., 16)`` and will be transformed
        as part of the equivariance test.
    num_multivector_args
        Number of multivector arguments that ``function`` accepts.
    fn_kwargs
        Keyword arguments passed to ``function``.
    batch_dims
        Batch shapes for the multivector inputs. Expects a list of tuples when
        ``num_multivector_args > 1``.
    spin
        If True, tests Spin equivariance; if False, tests Pin equivariance.
    rng
        Numpy RNG used to draw the inputs and transformations.
    num_checks
        Number of random draws used for the equivariance test.
    **kwargs
        Optional tolerance keyword arguments forwarded to :func:`torch.testing.assert_close`
        (e.g. ``atol``, ``rtol``).
    """
    # Default arguments
    if fn_kwargs is None:
        fn_kwargs = {}

    # Propagate numpy random state to torch
    if rng is not None:
        torch.manual_seed(rng.integers(100000))

    if num_multivector_args == 1:
        batch_dims = [batch_dims]
    assert num_multivector_args == len(batch_dims)

    # Loop over multiple checks
    for _ in range(num_checks):
        # Generate function inputs and Pin(3,0,1) transformations
        # Generate function inputs
        inputs = [torch.randn(*batch_dim, 16) for batch_dim in batch_dims]

        transform = SlowRandomPinTransform(rng=rng, spin=spin)

        # First function, then transformation
        outputs = get_first_output(function(*inputs, **fn_kwargs))
        transformed_outputs = transform(outputs)

        # First transformation, then function
        transformed_inputs = [transform(inputss) for inputss in inputs]
        outputs_of_transformed = get_first_output(function(*transformed_inputs, **fn_kwargs))

        # Check equality
        torch.testing.assert_close(transformed_outputs, outputs_of_transformed, **kwargs)


def check_pin_invariance(
    function: Callable,
    num_multivector_args: int = 1,
    fn_kwargs: dict | None = None,
    batch_dims: tuple | list = (1,),
    spin: bool = True,
    rng: np.random.Generator | None = None,
    num_checks: int = 2,
    **kwargs,
) -> None:
    """Check whether a callable is invariant w.r.t. the Pin(1, 3) or Spin(1, 3) group.

    Parameters
    ----------
    function
        Function to be tested for invariance. The first ``num_multivector_args`` positional
        arguments must accept multivector tensors of shape ``(..., 16)`` and will be transformed
        as part of the invariance test.
    num_multivector_args
        Number of multivector arguments that ``function`` accepts.
    fn_kwargs
        Keyword arguments passed to ``function``.
    batch_dims
        Batch shapes for the multivector inputs. Expects a list of tuples when
        ``num_multivector_args > 1``.
    spin
        If True, tests Spin invariance; if False, tests Pin invariance. Since Spin is a subgroup
        of Pin, confirming Pin invariance is usually enough.
    rng
        Numpy RNG used to draw the inputs and transformations.
    num_checks
        Number of random draws used for the invariance test.
    **kwargs
        Optional tolerance keyword arguments forwarded to :func:`torch.testing.assert_close`
        (e.g. ``atol``, ``rtol``).
    """
    # Default arguments
    if fn_kwargs is None:
        fn_kwargs = {}

    # Propagate numpy random state to torch
    if rng is not None:
        torch.manual_seed(rng.integers(100000))

    if num_multivector_args == 1:
        batch_dims = [batch_dims]
    assert num_multivector_args == len(batch_dims)

    # Loop over multiple checks
    for _ in range(num_checks):
        # Generate function inputs
        inputs = [torch.randn(*batch_dim, 16) for batch_dim in batch_dims]

        # Transform inputs with Pin(1,3)
        transform = SlowRandomPinTransform(rng=rng, spin=spin)
        transformed_inputs = [transform(inputss) for inputss in inputs]

        # Evaluate function on original and transformed inputs
        outputs = get_first_output(function(*inputs, **fn_kwargs))
        outputs_of_transformed = get_first_output(function(*transformed_inputs, **fn_kwargs))

        # Check equality
        torch.testing.assert_close(outputs, outputs_of_transformed, **kwargs)
