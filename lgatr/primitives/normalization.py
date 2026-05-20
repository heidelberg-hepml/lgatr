"""Multivector normalization."""

import torch

from ..utils.autocast import minimum_autocast_precision
from .invariants import abs_squared_norm


@minimum_autocast_precision(torch.float32)
def equi_layer_norm(
    x: torch.Tensor,
    channel_dim: int = -2,
    gain: float = 1.0,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Equivariant LayerNorm for multivectors.

    Rescales the input such that ``mean_channels |x|^2 = 1``, where the norm is the GA norm and
    the mean is taken over the channel dimension.

    Using a factor ``gain > 1`` makes up for the fact that the GA norm overestimates the actual
    standard deviation of the input data.

    Parameters
    ----------
    x
        Input multivectors of shape ``(..., 16)``.
    channel_dim
        Channel-dimension index. Defaults to the second-to-last entry (the last is the multivector
        component dimension).
    gain
        Target output scale.
    epsilon
        Small numerical offset to avoid instabilities. The default is intentionally larger than
        usual to balance the fact that some multivector components do not contribute to the norm.

    Returns
    -------
    outputs
        Normalized multivectors of shape ``(..., 16)``.
    """

    # Compute mean_channels |inputs|^2
    abs_squared_norms = abs_squared_norm(x)
    abs_squared_norms = torch.mean(abs_squared_norms, dim=channel_dim, keepdim=True)

    # Insure against low-norm tensors (which can arise even when `x.var(dim=-1)` is high b/c some
    # entries don't contribute to the inner product / GP norm!)
    abs_squared_norms = torch.clamp(abs_squared_norms, epsilon)

    # ``gain * rsqrt(...)`` collapses to a small (..., 1, 16) tensor first, so the final
    # broadcast multiply touches ``x`` only once (rather than ``gain * x * rsqrt`` which
    # would allocate an intermediate the size of ``x``).
    return x * (gain * torch.rsqrt(abs_squared_norms))
