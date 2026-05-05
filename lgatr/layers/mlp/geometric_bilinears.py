"""Geometric product on multivectors."""

import torch
from torch import nn

from ...primitives import geometric_product
from ...primitives.config import PrimitivesConfig
from ..layer_norm import EquiLayerNorm
from ..linear import EquiLinear


class GeometricBilinear(nn.Module):
    """Pin-equivariant bilinear map that constructs new geometric features via geometric products.

    Parameters
    ----------
    in_mv_channels
        Input multivector channels.
    out_mv_channels
        Output multivector channels.
    primitives
        LGATr primitives configuration.
    hidden_mv_channels
        Hidden multivector channels. If None, uses ``out_mv_channels``.
    in_s_channels
        Input scalar channels. Use 0 for no scalar inputs.
    out_s_channels
        Output scalar channels. Use 0 for no scalar outputs.
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        primitives: PrimitivesConfig,
        hidden_mv_channels: int | None = None,
        in_s_channels: int = 0,
        out_s_channels: int = 0,
    ) -> None:
        super().__init__()
        self.primitives = primitives

        # Default options
        if hidden_mv_channels is None:
            hidden_mv_channels = out_mv_channels

        # Linear projections for GP
        self.linear_left = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            primitives,
            in_s_channels=in_s_channels,
            out_s_channels=0,
        )
        self.linear_right = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            primitives,
            in_s_channels=in_s_channels,
            out_s_channels=0,
            initialization="almost_unit_scalar",
        )

        # Output linear projection
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            primitives,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
        )
        self.norm = EquiLayerNorm()

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        multivectors
            Input multivectors of shape ``(..., in_mv_channels, 16)``.
        scalars
            Optional input scalars of shape ``(..., in_s_channels)``. If None, the scalar stream
            is bypassed.

        Returns
        -------
        outputs_mv
            Output multivectors of shape ``(..., out_mv_channels, 16)``.
        outputs_s
            Output scalars of shape ``(..., out_s_channels)``, or None if ``out_s_channels == 0``.
        """

        # GP
        left, _ = self.linear_left(multivectors, scalars=scalars)
        right, _ = self.linear_right(multivectors, scalars=scalars)
        gp_outputs = geometric_product(left, right, config=self.primitives)
        if not self.primitives.use_bivector:
            gp_outputs[..., 5:11] = 0.0

        # Output linear
        outputs_mv, outputs_s = self.linear_out(gp_outputs, scalars=scalars)

        outputs_mv, outputs_s = self.norm(outputs_mv, outputs_s)
        return outputs_mv, outputs_s
