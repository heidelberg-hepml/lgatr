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

    if not training or p == 0.0:
        return x

    x = grade_project(x)
    # dropout1d only accepts a single batch dim; ``reshape`` (not ``view``) tolerates non-
    # contiguous inputs from the broadcast in ``grade_project``.
    h = x.reshape(-1, 5, 16)
    h = torch.nn.functional.dropout1d(h, p=p, training=True, inplace=False)
    return h.reshape(x.shape).sum(dim=-2)
