"""Equivariant normalization layers."""

import torch
from torch import nn

from ..primitives import equi_layer_norm


class EquiLayerNorm(nn.Module):
    """Layer normalization for multivectors (and an optional scalar stream).

    Rescales the multivector input such that ``mean_channels |x|^2 = 1``, where the norm is the GA
    norm and the mean is taken over the channel dimension. The scalar stream, if present, is
    passed through a regular :func:`torch.nn.functional.layer_norm`.

    Parameters
    ----------
    mv_channel_dim
        Channel-dimension index for multivector inputs. Defaults to the second-to-last entry (the
        last is the multivector component dimension).
    epsilon
        Small numerical offset to avoid instabilities. The default is intentionally larger than
        usual to balance the fact that some multivector components do not contribute to the norm.
    """

    def __init__(self, mv_channel_dim: int = -2, epsilon: float = 0.01) -> None:
        super().__init__()
        self.mv_channel_dim = mv_channel_dim
        self.epsilon = epsilon

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply equivariant LayerNorm.

        Parameters
        ----------
        multivectors
            Multivector inputs of shape ``(..., 16)``.
        scalars
            Optional scalar inputs of shape ``(...)``. If None, no scalar normalization is
            performed and ``outputs_s`` is None.

        Returns
        -------
        outputs_mv
            Normalized multivectors of shape ``(..., 16)``.
        outputs_s
            Normalized scalars of shape ``(...)``, or None if ``scalars`` is None.
        """

        outputs_mv = equi_layer_norm(
            multivectors, channel_dim=self.mv_channel_dim, epsilon=self.epsilon
        )
        if scalars is None:
            outputs_s = None
        else:
            outputs_s = torch.nn.functional.layer_norm(scalars, normalized_shape=scalars.shape[-1:])

        return outputs_mv, outputs_s
