"""Helpers for the light-cone versus Cartesian parity tests."""

import torch

from lgatr.interface import get_lightcone_frame


def random_lightcone_frame(x: torch.Tensor) -> torch.Tensor:
    """Random float64 light-cone frames along the first dim of ``x``, broadcast over the others."""
    frame = get_lightcone_frame(torch.randn(x.shape[0], 4, dtype=torch.float64))
    return frame.view(x.shape[0], *[1] * (x.dim() - 2), 4, 4)


def outputs_and_input_grads(fn, *inputs):
    """Outputs of ``fn`` and the gradients of their squared sum with respect to the inputs."""
    inputs = [x.clone().requires_grad_() for x in inputs]
    outputs = fn(*inputs)
    outputs.square().sum().backward()
    return outputs.detach(), *(x.grad for x in inputs)
