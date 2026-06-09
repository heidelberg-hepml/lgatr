"""Equivariant normalization layers."""

import torch
from torch import nn

from ..primitives import equi_layer_norm

# Maps each of the 16 multivector components to its grade (grade dimensions [1, 4, 6, 4, 1]).
_GRADE_INDEX = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4]


class EquiLayerNorm(nn.Module):
    """Layer normalization for multivectors (and an optional scalar stream).

    Rescales the multivector input such that ``mean_channels |x|^2 = 1``, where the norm is the GA
    norm and the mean is taken over the channel dimension. The scalar stream, if present, is
    passed through a regular :func:`torch.nn.functional.layer_norm`.

    With ``elementwise_affine=True`` a learnable gain is applied after normalization: a per-grade
    scalar per multivector channel (shape ``(mv_channels, 5)``, broadcast over the components of
    each grade) and a per-channel scalar for the scalar stream (shape ``(s_channels,)``). Scaling
    each grade independently preserves Pin-equivariance.

    Parameters
    ----------
    mv_channels
        Number of multivector channels. Only used to size the affine gain when
        ``elementwise_affine=True``.
    s_channels
        Number of scalar channels. Only used to size the affine gain when
        ``elementwise_affine=True``.
    mv_channel_dim
        Channel-dimension index for multivector inputs. Defaults to the second-to-last entry (the
        last is the multivector component dimension).
    epsilon
        Small numerical offset to avoid instabilities. The default is intentionally larger than
        usual to balance the fact that some multivector components do not contribute to the norm.
    elementwise_affine
        Whether to learn a per-channel-per-grade multivector gain and a per-channel scalar gain.
    """

    def __init__(
        self,
        mv_channels: int = 0,
        s_channels: int = 0,
        mv_channel_dim: int = -2,
        epsilon: float = 0.01,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__()
        self.mv_channel_dim = mv_channel_dim
        self.epsilon = epsilon
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.register_buffer("grade_index", torch.tensor(_GRADE_INDEX), persistent=False)
            self.weight_mv = nn.Parameter(torch.ones(mv_channels, 5))
            self.weight_s = nn.Parameter(torch.ones(s_channels))
            # zero-size params get grads only sometimes under compile, breaking DDP
            for weight in (self.weight_mv, self.weight_s):
                if weight.numel() == 0:
                    weight.requires_grad_(False)
        else:
            self.register_parameter("weight_mv", None)
            self.register_parameter("weight_s", None)

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply equivariant LayerNorm.

        Parameters
        ----------
        multivectors
            Multivector inputs of shape ``(..., 16)``.
        scalars
            Optional scalar inputs of shape ``(..., s_channels)``. If None, no scalar normalization
            is performed and ``outputs_s`` is None.

        Returns
        -------
        outputs_mv
            Normalized multivectors of shape ``(..., 16)``.
        outputs_s
            Normalized scalars of shape ``(..., s_channels)``, or None if ``scalars`` is None.
        """

        outputs_mv = equi_layer_norm(
            multivectors, channel_dim=self.mv_channel_dim, epsilon=self.epsilon
        )
        if scalars is None:
            outputs_s = None
        else:
            outputs_s = torch.nn.functional.layer_norm(scalars, normalized_shape=scalars.shape[-1:])

        if self.elementwise_affine:
            weight16 = self.weight_mv[:, self.grade_index]  # (mv_channels, 16)
            outputs_mv = outputs_mv * weight16.to(outputs_mv.dtype)
            if outputs_s is not None:
                outputs_s = outputs_s * self.weight_s.to(outputs_s.dtype)

        return outputs_mv, outputs_s
