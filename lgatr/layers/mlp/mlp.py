"""MLP with geometric product."""

from typing import List, Tuple, Union

import torch
from torch import nn

from ...primitives.config import gatr_config
from ..dropout import GradeDropout
from ..linear import EquiLinear
from .config import MLPConfig
from .geometric_bilinears import GeometricBilinear
from .nonlinearities import ScalarGatedNonlinearity


class GeoMLP(nn.Module):
    """MLP with geometric product.

    This is a core component of GATr's transformer blocks. It is similar to a regular MLP, except
    that it uses geometric bilinears (the geometric product) in place of the first linear layer.

    Assumes input has shape (..., channels, 16), output has shape (..., channels, 16),
    will create hidden layers with shape (..., increase_hidden_channels*channels, 16).

    Parameters
    ----------
    config: MLPConfig
        Configuration object
    """

    def __init__(
        self,
        config: MLPConfig,
    ) -> None:
        super().__init__()

        # Store settings
        self.config = config

        assert config.mv_channels is not None
        s_channels = None if config.s_channels is None else config.s_channels

        mv_channels_list = [config.mv_channels]
        mv_channels_list.extend(
            [config.increase_hidden_channels * config.mv_channels]
            * config.num_hidden_layers
        )
        mv_channels_list.append(config.mv_channels)
        if s_channels is not None:
            s_channels_list = [s_channels]
            s_channels_list.extend(
                [config.increase_hidden_channels * s_channels]
                * config.num_hidden_layers
            )
            s_channels_list.append(s_channels)
        else:
            s_channels = [None] * (len(mv_channels_list))

        layers: List[nn.Module] = []

        if config.num_hidden_layers >= 0:
            kwargs = dict(
                in_mv_channels=mv_channels_list[0],
                out_mv_channels=mv_channels_list[1],
                in_s_channels=s_channels_list[0],
                out_s_channels=s_channels_list[1],
            )
            if gatr_config.use_geometric_product:
                layers.append(GeometricBilinear(**kwargs))
            else:
                layers.append(ScalarGatedNonlinearity(config.activation))
                layers.append(EquiLinear(**kwargs))
            if config.dropout_prob is not None:
                layers.append(GradeDropout(config.dropout_prob))

            for in_, out, in_s, out_s in zip(
                mv_channels_list[1:-1],
                mv_channels_list[2:],
                s_channels_list[1:-1],
                s_channels_list[2:],
            ):
                layers.append(ScalarGatedNonlinearity(config.activation))
                layers.append(
                    EquiLinear(in_, out, in_s_channels=in_s, out_s_channels=out_s)
                )
                if config.dropout_prob is not None:
                    layers.append(GradeDropout(config.dropout_prob))

        self.layers = nn.ModuleList(layers)

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor
            Input multivectors with shape (..., mv_channels, 16).
        scalars : None or torch.Tensor
            Optional input scalars with shape (..., s_channels).

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors  with shape (..., mv_channels, 16).
        outputs_s : None or torch.Tensor
            Output scalars with shape (..., s_channels).
        """

        mv, s = multivectors, scalars

        for layer in self.layers:
            mv, s = layer(mv, scalars=s)

        return mv, s
