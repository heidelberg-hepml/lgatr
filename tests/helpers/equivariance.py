"""Utility functions to test callables for equivariance and invariance.

Two groups are covered: Pin(1, 3) acting on multivectors of shape ``(..., 16)``, and SO(1, 3)
acting on plain Lorentz vectors (used by the ``*Slim`` models, which have no geometric algebra).
"""

from collections.abc import Callable
from typing import Any

import torch

from .clifford import RandomPinTransform
from .rand_lorentz import rand_lorentz


class RandomLorentzTransform:
    """Random SO(1, 3) transform on Lorentz-vector tensors.

    ``vector_dim`` says where the 4 lives: ``-1`` for ``(..., 4)``, ``-2`` for ``(..., 4, channels)``
    (in which case the last entry of ``batch_dims`` is the channel count).
    """

    def __init__(self, vector_dim: int = -1) -> None:
        assert vector_dim in (-1, -2)
        self._vector_dim = vector_dim
        self._matrix = rand_lorentz(())

    def sample(self, batch_dims: tuple[int, ...] | list[int]) -> torch.Tensor:
        """Draw random Lorentz-vector inputs with the 4 at ``vector_dim``."""
        if self._vector_dim == -1:
            return torch.randn(*batch_dims, 4)

        return torch.randn(*batch_dims[:-1], 4, batch_dims[-1])

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._vector_dim == -1:
            return torch.einsum("ij,...j->...i", self._matrix, inputs)

        return torch.einsum("ij,...jc->...ic", self._matrix, inputs)


def _first(outputs: Any) -> torch.Tensor:
    """Primitives return a single tensor, layers return a tuple; test the first tensor either way."""
    return outputs[0] if isinstance(outputs, tuple) else outputs


def _check(
    function: Callable,
    transforms: list,
    num_args: int,
    batch_dims: tuple | list,
    fn_kwargs: dict | None,
    invariant: bool,
    **tolerances,
) -> None:
    """Compare ``f(g x)`` against ``g f(x)`` (or ``f(x)`` when ``invariant``) for each transform."""
    if fn_kwargs is None:
        fn_kwargs = {}
    if num_args == 1:
        batch_dims = [batch_dims]
    assert num_args == len(batch_dims)

    for transform in transforms:
        inputs = [transform.sample(dims) for dims in batch_dims]

        outputs = _first(function(*inputs, **fn_kwargs))
        expected = outputs if invariant else transform(outputs)

        transformed_inputs = [transform(x) for x in inputs]
        outputs_of_transformed = _first(function(*transformed_inputs, **fn_kwargs))

        torch.testing.assert_close(expected, outputs_of_transformed, **tolerances)


def _pin_transforms(spin: bool, num_checks: int) -> list[RandomPinTransform]:
    """Transforms for one Pin/Spin check: Spin uses even elements, Pin alternates odd and even."""
    return [RandomPinTransform(odd=not spin and i % 2 == 0) for i in range(num_checks)]


def check_pin_equivariance(
    function: Callable,
    num_multivector_args: int = 1,
    fn_kwargs: dict | None = None,
    batch_dims: tuple | list = (1,),
    spin: bool = True,
    num_checks: int = 2,
    **tolerances,
) -> None:
    """Check whether a callable is equivariant w.r.t. the Pin(1, 3) or Spin(1, 3) group.

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
        If True, tests Spin equivariance; if False, tests Pin equivariance, which additionally
        applies reflections. Models built with ``PrimitivesConfig(subgroup=True)`` are equivariant
        under Spin only.
    num_checks
        Number of random draws used for the equivariance test.
    **tolerances
        Optional tolerance keyword arguments forwarded to :func:`torch.testing.assert_close`
        (e.g. ``atol``, ``rtol``).
    """
    _check(
        function,
        _pin_transforms(spin, num_checks),
        num_multivector_args,
        batch_dims,
        fn_kwargs,
        invariant=False,
        **tolerances,
    )


def check_pin_invariance(
    function: Callable,
    num_multivector_args: int = 1,
    fn_kwargs: dict | None = None,
    batch_dims: tuple | list = (1,),
    spin: bool = True,
    num_checks: int = 2,
    **tolerances,
) -> None:
    """Check whether a callable is invariant w.r.t. the Pin(1, 3) or Spin(1, 3) group.

    Takes the same arguments as :func:`check_pin_equivariance`, but compares the outputs of the
    transformed inputs against the untransformed outputs.
    """
    _check(
        function,
        _pin_transforms(spin, num_checks),
        num_multivector_args,
        batch_dims,
        fn_kwargs,
        invariant=True,
        **tolerances,
    )


def check_equivariance(
    function: Callable,
    fn_kwargs: dict | None = None,
    batch_dims: tuple | list = (1,),
    num_args: int = 1,
    num_checks: int = 2,
    vector_dim: int = -1,
    **tolerances,
) -> None:
    """Check whether a callable is SO(1, 3)-equivariant on Lorentz-vector inputs.

    Takes the same arguments as :func:`check_pin_equivariance`, plus ``vector_dim`` to say whether
    the inputs are shaped ``(..., 4)`` or ``(..., 4, channels)``.
    """
    _check(
        function,
        [RandomLorentzTransform(vector_dim) for _ in range(num_checks)],
        num_args,
        batch_dims,
        fn_kwargs,
        invariant=False,
        **tolerances,
    )
