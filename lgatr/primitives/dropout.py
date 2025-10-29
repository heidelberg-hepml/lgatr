"""Grade dropout."""

import torch

from .linear import grade_project


def grade_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """Multivector dropout, dropping out grades independently.

    Parameters
    ----------
    x : torch.Tensor
        Input data with shape (..., 16).
    p : float
        Dropout probability (assumed the same for each grade).
    training : bool
        Switches between train-time and test-time behaviour.

    Returns
    -------
    outputs : torch.Tensor
        Inputs with dropout applied, shape (..., 16).
    """

    # Project to grades
    x = grade_project(x)

    # Apply standard 1D dropout
    # For whatever reason, that only works with a single batch dimension, so let's reshape a bit
    h = x.view(-1, 5, 16)
    h = torch.nn.functional.dropout1d(h, p=p, training=training, inplace=False)
    h = h.view(x.shape)

    # Combine grades again
    h = torch.sum(h, dim=-2)

    return h
