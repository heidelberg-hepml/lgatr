"""Equivariant dropout layer."""

import torch
from torch import nn

from ..primitives import grade_dropout


class GradeDropout(nn.Module):
    """Dropout on multivectors.

    Parameters
    ----------
    p : float
        Dropout probability.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self._dropout_prob = p

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Applies dropout.

        Parameters
        ----------
        multivectors : torch.Tensor
            Multivector inputs  with shape (..., 16).
        scalars : torch.Tensor
            Scalar inputs with shape (...).

        Returns
        -------
        outputs_mv : torch.Tensor
            Multivector inputs with dropout applied, shape (..., 16).
        output_scalars : torch.Tensor
            Scalar inputs with dropout applied, shape (...).
        """

        out_mv = grade_dropout(multivectors, p=self._dropout_prob, training=self.training)
        out_s = torch.nn.functional.dropout(scalars, p=self._dropout_prob, training=self.training)

        return out_mv, out_s
