"""Equivariant dropout layer."""

import torch
from torch import nn

from ..primitives import grade_dropout


class GradeDropout(nn.Module):
    """Grade-wise dropout on multivectors (and an optional scalar stream).

    Parameters
    ----------
    p
        Dropout probability.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self._dropout_prob = p

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply dropout to multivectors and (optionally) scalars.

        Parameters
        ----------
        multivectors
            Multivector inputs of shape ``(..., 16)``.
        scalars
            Optional scalar inputs of shape ``(...)``. If None, scalar dropout is skipped and
            ``outputs_s`` is None.

        Returns
        -------
        outputs_mv
            Multivectors after dropout, shape ``(..., 16)``.
        outputs_s
            Scalars after dropout, shape ``(...)``, or None if ``scalars`` is None.
        """

        out_mv = grade_dropout(multivectors, p=self._dropout_prob, training=self.training)
        if scalars is None:
            out_s = None
        else:
            out_s = torch.nn.functional.dropout(
                scalars, p=self._dropout_prob, training=self.training
            )

        return out_mv, out_s
