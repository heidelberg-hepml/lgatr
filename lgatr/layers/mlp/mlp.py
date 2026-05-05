"""MLP with geometric product."""

import torch
from torch import nn

from ...primitives.config import PrimitivesConfig
from ..dropout import GradeDropout
from ..linear import EquiLinear
from .config import MLPConfig
from .geometric_bilinears import GeometricBilinear
from .nonlinearities import ScalarGatedNonlinearity


class GeoMLP(nn.Module):
    """MLP with a geometric product as the first nonlinear mixing step.

    Similar to a regular MLP, except the first linear layer is replaced by a geometric bilinear
    (the geometric product). Inputs of shape ``(..., channels, 16)`` map to outputs of shape
    ``(..., channels, 16)``; hidden layers have shape
    ``(..., increase_hidden_channels * channels, 16)``.

    Parameters
    ----------
    config
        MLP configuration.
    primitives
        LGATr primitives configuration.
    """

    def __init__(
        self,
        config: MLPConfig,
        primitives: PrimitivesConfig,
    ) -> None:
        super().__init__()

        # Store settings
        self.config = config
        self.primitives = primitives

        assert config.mv_channels is not None
        s_channels = config.s_channels

        mv_channels_list = [config.mv_channels]
        mv_channels_list.extend(
            [config.increase_hidden_channels * config.mv_channels] * config.num_hidden_layers
        )
        mv_channels_list.append(config.mv_channels)
        s_channels_list = [s_channels]
        s_channels_list.extend(
            [config.increase_hidden_channels * s_channels] * config.num_hidden_layers
        )
        s_channels_list.append(s_channels)

        layers: list[nn.Module] = []

        if config.num_hidden_layers >= 0:
            kwargs = dict(
                in_mv_channels=mv_channels_list[0],
                out_mv_channels=mv_channels_list[1],
                in_s_channels=s_channels_list[0],
                out_s_channels=s_channels_list[1],
            )
            if primitives.use_geometric_product:
                layers.append(GeometricBilinear(primitives=primitives, **kwargs))
            else:
                layers.append(ScalarGatedNonlinearity(config.activation))
                layers.append(EquiLinear(primitives=primitives, **kwargs))
            if config.dropout_prob is not None:
                layers.append(GradeDropout(config.dropout_prob))

            for in_, out, in_s, out_s in zip(
                mv_channels_list[1:-1],
                mv_channels_list[2:],
                s_channels_list[1:-1],
                s_channels_list[2:],
                strict=False,
            ):
                layers.append(ScalarGatedNonlinearity(config.activation))
                layers.append(
                    EquiLinear(in_, out, primitives, in_s_channels=in_s, out_s_channels=out_s)
                )
                if config.dropout_prob is not None:
                    layers.append(GradeDropout(config.dropout_prob))

        self.layers = nn.ModuleList(layers)

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        multivectors
            Input multivectors of shape ``(..., mv_channels, 16)``.
        scalars
            Optional input scalars of shape ``(..., s_channels)``. If None, the scalar stream is
            bypassed and ``outputs_s`` is None.

        Returns
        -------
        outputs_mv
            Output multivectors of shape ``(..., mv_channels, 16)``.
        outputs_s
            Output scalars of shape ``(..., s_channels)``, or None if ``scalars`` is None.
        """

        h_mv, h_s = multivectors, scalars

        for layer in self.layers:
            h_mv, h_s = layer(h_mv, scalars=h_s)

        return h_mv, h_s
