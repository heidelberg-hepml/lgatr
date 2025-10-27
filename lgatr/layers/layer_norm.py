"""Equivariant normalization layers."""

import torch
from torch import nn

from ..primitives import equi_layer_norm


class EquiLayerNorm(nn.Module):
    """Layer normalization.

    Rescales input such that ``mean_channels |inputs|^2 = 1``, where the norm is the GA norm and the
    mean goes over the channel dimensions.

    In addition, the layer performs a regular LayerNorm operation on auxiliary scalar inputs.

    Parameters
    ----------
    mv_channel_dim : int
        Channel dimension index for multivector inputs. Defaults to the second-last entry (last are
        the multivector components).
    epsilon : float
        Small numerical factor to avoid instabilities. We use a reasonably large number to balance
        issues that arise from some multivector components not contributing to the norm.
    """

    def __init__(self, mv_channel_dim=-2, epsilon: float = 0.01):
        super().__init__()
        self.mv_channel_dim = mv_channel_dim
        self.epsilon = epsilon

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Computes equivariant LayerNorm for multivectors.

        Parameters
        ----------
        multivectors : torch.Tensor
            Multivector inputs with shape (..., 16).
        scalars : torch.Tensor
            Scalar inputs with shape (...).

        Returns
        -------
        outputs_mv : torch.Tensor
            Normalized multivectors with shape (..., 16).
        output_scalars : torch.Tensor
            Normalized scalars with shape (...).
        """

        outputs_mv = equi_layer_norm(
            multivectors, channel_dim=self.mv_channel_dim, epsilon=self.epsilon
        )
        normalized_shape = scalars.shape[-1:]
        outputs_s = torch.nn.functional.layer_norm(scalars, normalized_shape=normalized_shape)

        return outputs_mv, outputs_s
