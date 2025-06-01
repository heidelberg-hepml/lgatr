"""Gated nonlinearity on multivectors."""
from typing import Tuple

import torch
from torch import nn

from ...primitives.nonlinearities import gated_gelu, gated_relu, gated_sigmoid


class ScalarGatedNonlinearity(nn.Module):
    """Gated nonlinearity on multivectors.

    Given multivector input x, computes ``f(x_0) * x``, where f can either be ReLU, sigmoid, or GeLU.

    Auxiliary scalar inputs are simply processed with ReLU, sigmoid, or GeLU, without gating.

    Parameters
    ----------
    nonlinearity : {"relu", "sigmoid", "gelu"}
        Non-linearity type
    """

    def __init__(self, nonlinearity: str = "relu") -> None:
        super().__init__()

        gated_fn_dict = dict(relu=gated_relu, gelu=gated_gelu, sigmoid=gated_sigmoid)
        scalar_fn_dict = dict(
            relu=nn.functional.relu,
            gelu=nn.functional.gelu,
            sigmoid=nn.functional.sigmoid,
        )
        try:
            self.gated_nonlinearity = gated_fn_dict[nonlinearity]
            self.scalar_nonlinearity = scalar_fn_dict[nonlinearity]
        except KeyError as exc:
            raise ValueError(
                f"Unknown nonlinearity {nonlinearity} for options {list(gated_fn_dict.keys())}"
            ) from exc

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes ``f(x_0) * x`` for multivector x, where f is GELU, ReLU, or sigmoid.

        f is chosen depending on self.gated_nonlinearity and self.scalar_nonlinearity.

        Parameters
        ----------
        multivectors : torch.Tensor
            Input multivectors with shape (..., 16)
        scalars : None or torch.Tensor
            Input scalars with shape (...)

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors with shape (..., 16)
        output_scalars : torch.Tensor
            Output scalars with shape (...)
        """

        gates = multivectors[..., [0]]
        outputs_mv = self.gated_nonlinearity(multivectors, gates=gates)
        outputs_s = self.scalar_nonlinearity(scalars)

        return outputs_mv, outputs_s
