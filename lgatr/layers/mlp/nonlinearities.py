"""Gated nonlinearity on multivectors."""

import torch
from torch import nn

from ...primitives.nonlinearities import (
    gated_gelu,
    gated_relu,
    gated_sigmoid,
    gated_silu,
)


class ScalarGatedNonlinearity(nn.Module):
    """Gated nonlinearity on multivectors.

    Given multivector input ``x``, computes ``f(x_0) * x``, where ``f`` is one of ReLU, sigmoid,
    GeLU, or SiLU. Auxiliary scalar inputs are processed with the same ``f`` directly (without
    gating).

    Parameters
    ----------
    nonlinearity
        Non-linearity type. One of ``"relu"``, ``"sigmoid"``, ``"gelu"``, ``"silu"``.
    """

    def __init__(self, nonlinearity: str = "relu") -> None:
        super().__init__()

        gated_fn_dict = dict(
            relu=gated_relu, gelu=gated_gelu, sigmoid=gated_sigmoid, silu=gated_silu
        )
        scalar_fn_dict = dict(
            relu=nn.functional.relu,
            gelu=nn.functional.gelu,
            sigmoid=nn.functional.sigmoid,
            silu=nn.functional.silu,
        )
        try:
            self.gated_nonlinearity = gated_fn_dict[nonlinearity]
            self.scalar_nonlinearity = scalar_fn_dict[nonlinearity]
        except KeyError as exc:
            raise ValueError(
                f"Unknown nonlinearity {nonlinearity} for options {list(gated_fn_dict.keys())}"
            ) from exc

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
        outputs_mv = self.gated_nonlinearity(multivectors, gates=gates)
        outputs_s = self.scalar_nonlinearity(scalars) if scalars is not None else None

        return outputs_mv, outputs_s
