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
        self, multivectors: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass. Applies dropout.

        Parameters
        ----------
        multivectors : torch.Tensor
            Multivector inputs  with shape (..., 16).
        scalars : None or torch.Tensor
            Optional scalar inputs with shape (...). If None, scalar dropout is
            skipped and outputs_s is None.

        Returns
        -------
        outputs_mv : torch.Tensor
            Multivector inputs with dropout applied, shape (..., 16).
        output_scalars : None or torch.Tensor
            Scalar inputs with dropout applied, shape (...), or None if scalars is None.
        """

        out_mv = grade_dropout(multivectors, p=self._dropout_prob, training=self.training)
        if scalars is None:
            out_s = None
        else:
            out_s = torch.nn.functional.dropout(
                scalars, p=self._dropout_prob, training=self.training
            )

        return out_mv, out_s
