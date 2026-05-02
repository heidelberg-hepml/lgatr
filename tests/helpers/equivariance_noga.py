"""Utility functions to test callables for equivariance with respect to SO(1, 3) (no GA)."""

from collections.abc import Callable

import torch
from lloca.utils.rand_transforms import rand_lorentz


def check_invariance(
    function: Callable,
    fn_kwargs: dict | None = None,
    batch_dims: tuple | list = (1,),
    num_args: int = 1,
    num_checks: int = 2,
    **kwargs,
) -> None:
    """Check whether a callable is SO(1, 3)-invariant on Lorentz-vector inputs of shape ``(..., 4)``."""
    if fn_kwargs is None:
        fn_kwargs = {}
    if num_args == 1:
        batch_dims = [batch_dims]

    for _ in range(num_checks):
        inputs = [torch.randn(*batch_dims[i], 4) for i in range(num_args)]
        trafo = rand_lorentz((1,) * len(batch_dims))

        outputs = function(*inputs, **fn_kwargs)[0]

        inputs_transformed = [torch.einsum("...ij,...j->...i", trafo, inp) for inp in inputs]
        outputs_of_transformed = function(*inputs_transformed, **fn_kwargs)[0]

        torch.testing.assert_close(outputs, outputs_of_transformed, **kwargs)


def check_equivariance(
    function: Callable,
    fn_kwargs: dict | None = None,
    batch_dims: tuple | list = (1,),
    num_args: int = 1,
    num_checks: int = 2,
    **kwargs,
) -> None:
    """Check whether a callable is SO(1, 3)-equivariant on Lorentz-vector inputs of shape ``(..., 4)``."""
    if fn_kwargs is None:
        fn_kwargs = {}
    if num_args == 1:
        batch_dims = [batch_dims]

    for _ in range(num_checks):
        inputs = [torch.randn(*batch_dims[i], 4) for i in range(num_args)]
        trafo = rand_lorentz((1,) * len(batch_dims))

        outputs = function(*inputs, **fn_kwargs)[0]
        outputs_transformed = torch.einsum("...ij,...j->...i", trafo, outputs)

        inputs_transformed = [torch.einsum("...ij,...j->...i", trafo, inp) for inp in inputs]
        outputs_of_transformed = function(*inputs_transformed, **fn_kwargs)[0]

        torch.testing.assert_close(outputs_transformed, outputs_of_transformed, **kwargs)
