"""Utility functions to test callables for equivariance with respect to SO(1,3)."""

import torch
from lloca.utils.rand_transforms import rand_lorentz


def check_invariance(
    function,
    fn_kwargs=None,
    batch_dims=(1,),
    num_args=1,
    num_checks=3,
    **kwargs,
):
    if fn_kwargs is None:
        fn_kwargs = {}

    for _ in range(num_checks):
        inputs = [torch.randn(*batch_dims, 4) for _ in range(num_args)]
        trafo = rand_lorentz((1,) * len(batch_dims))

        outputs = function(*inputs, **fn_kwargs)[0]

        inputs_transformed = [torch.einsum("...ij,...j->...i", trafo, inp) for inp in inputs]
        outputs_of_transformed = function(*inputs_transformed, **fn_kwargs)[0]

        torch.testing.assert_close(outputs, outputs_of_transformed, **kwargs)


def check_equivariance(
    function,
    fn_kwargs=None,
    batch_dims=(1,),
    num_args=1,
    num_checks=3,
    **kwargs,
):
    if fn_kwargs is None:
        fn_kwargs = {}

    for _ in range(num_checks):
        inputs = [torch.randn(*batch_dims, 4) for _ in range(num_args)]
        trafo = rand_lorentz((1,) * len(batch_dims))

        outputs = function(*inputs, **fn_kwargs)[0]
        outputs_transformed = torch.einsum("...ij,...j->...i", trafo, outputs)

        inputs_transformed = [torch.einsum("...ij,...j->...i", trafo, inp) for inp in inputs]
        outputs_of_transformed = function(*inputs_transformed, **fn_kwargs)[0]

        torch.testing.assert_close(outputs_transformed, outputs_of_transformed, **kwargs)
