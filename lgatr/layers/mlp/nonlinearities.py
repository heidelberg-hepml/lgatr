"""Gated nonlinearity on multivectors."""

import torch
from torch import nn

from ...utils.misc import get_nonlinearity


class ScalarGatedNonlinearity(nn.Module):
    """Gated nonlinearity on multivectors.

    Given multivector input ``x``, computes ``f(x_0) * x``, where ``f`` is one of ReLU, sigmoid,
    GeLU, or SiLU. Auxiliary scalar inputs are processed with the same ``f`` directly (without
    gating).

    Parameters
    ----------
    nonlinearity
        Non-linearity type. One of ``"relu"``, ``"sigmoid"``, ``"tanh"``, ``"gelu"``,
        ``"silu"``.
    """

    def __init__(self, nonlinearity: str = "gelu") -> None:
        super().__init__()
        self.nonlinearity = get_nonlinearity(nonlinearity)

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the gated nonlinearity.

        Parameters
        ----------
        multivectors
            Input multivectors of shape ``(..., 16)``.
        scalars
            Optional input scalars of shape ``(...)``. If None, ``outputs_s`` is None.

        Returns
        -------
        outputs_mv
            Output multivectors of shape ``(..., 16)``.
        outputs_s
            Output scalars of shape ``(...)``, or None if ``scalars`` is None.
        """

        gates = multivectors[..., 0:1]
        outputs_mv = self.nonlinearity(gates) * multivectors
        outputs_s = self.nonlinearity(scalars) if scalars is not None else None

        return outputs_mv, outputs_s
