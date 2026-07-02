"""Utility functions to test callables for equivariance with respect to SO(1, 3) (no GA)."""

from collections.abc import Callable

import torch

from .rand_lorentz import rand_lorentz


def _rand_vectors(batch_dims: tuple | list, vector_dim: int) -> torch.Tensor:
    """Random vector inputs: ``(..., 4)`` for vector_dim=-1, ``(..., 4, channels)`` for -2
    (the last entry of ``batch_dims`` is then the channel count)."""
    if vector_dim == -1:
        return torch.randn(*batch_dims, 4)
    assert vector_dim == -2
    return torch.randn(*batch_dims[:-1], 4, batch_dims[-1])


def _apply_lorentz(trafo: torch.Tensor, tensor: torch.Tensor, vector_dim: int) -> torch.Tensor:
    if vector_dim == -1:
        return torch.einsum("...ij,...j->...i", trafo, tensor)
    return torch.einsum("...ij,...jc->...ic", trafo, tensor)


def check_invariance(
    function: Callable,
    fn_kwargs: dict | None = None,
    batch_dims: tuple | list = (1,),
    num_args: int = 1,
    num_checks: int = 2,
    vector_dim: int = -1,
    **kwargs,
) -> None:
    """Check whether a callable is SO(1, 3)-invariant on Lorentz-vector inputs."""
    if fn_kwargs is None:
        fn_kwargs = {}
    if num_args == 1:
        batch_dims = [batch_dims]

    for _ in range(num_checks):
        inputs = [_rand_vectors(batch_dims[i], vector_dim) for i in range(num_args)]
        trafo = rand_lorentz((1,) * len(batch_dims))

        outputs = function(*inputs, **fn_kwargs)[0]

        inputs_transformed = [_apply_lorentz(trafo, inp, vector_dim) for inp in inputs]
        outputs_of_transformed = function(*inputs_transformed, **fn_kwargs)[0]

        torch.testing.assert_close(outputs, outputs_of_transformed, **kwargs)


def check_equivariance(
    function: Callable,
    fn_kwargs: dict | None = None,
    batch_dims: tuple | list = (1,),
    num_args: int = 1,
    num_checks: int = 2,
    vector_dim: int = -1,
    **kwargs,
) -> None:
    """Check whether a callable is SO(1, 3)-equivariant on Lorentz-vector inputs."""
    if fn_kwargs is None:
        fn_kwargs = {}
    if num_args == 1:
        batch_dims = [batch_dims]

    for _ in range(num_checks):
        inputs = [_rand_vectors(batch_dims[i], vector_dim) for i in range(num_args)]
        trafo = rand_lorentz((1,) * len(batch_dims))

        outputs = function(*inputs, **fn_kwargs)[0]
        outputs_transformed = _apply_lorentz(trafo, outputs, vector_dim)

        inputs_transformed = [_apply_lorentz(trafo, inp, vector_dim) for inp in inputs]
        outputs_of_transformed = function(*inputs_transformed, **fn_kwargs)[0]

        torch.testing.assert_close(outputs_transformed, outputs_of_transformed, **kwargs)
