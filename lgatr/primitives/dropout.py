"""Grade dropout."""

import torch

from .linear import grade_project


def grade_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """Multivector dropout that drops grades independently.

    Parameters
    ----------
    x
        Input data of shape ``(..., 16)``.
    p
        Dropout probability (the same for each grade).
    training
        Switches between train-time and test-time behavior.

    Returns
    -------
    outputs
        Inputs with dropout applied, shape ``(..., 16)``.
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
