"""Configuration dataclass for the geometric MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...utils.config import cast_config


@dataclass
class MLPConfig:
    """Geometric-MLP configuration.

    Parameters
    ----------
    mv_channels
        Number of input multivector channels. Set automatically by the parent network.
    s_channels
        Number of input scalar channels. Use 0 for no scalar stream. Set automatically by the
        parent network.
    dropout_prob
        Dropout probability. Set automatically by the parent network.
    nonlinearity
        Which (gated) activation function to use. One of ``"relu"``, ``"sigmoid"``, ``"tanh"``,
        ``"gelu"``, ``"silu"``.
    mlp_ratio
        Factor by which to increase the number of hidden channels (both multivectors and scalars).
    num_layers_mlp
        Total number of layers, including input and output layers (must be ``>= 1``).
    """

    mv_channels: int | None = None
    s_channels: int = 0
    dropout_prob: float | None = None
    nonlinearity: str = "gelu"
    mlp_ratio: int = 4
    num_layers_mlp: int = 2

    @classmethod
    def cast(cls, config: Any) -> MLPConfig:
        """Cast an :class:`MLPConfig` or mapping to an :class:`MLPConfig`."""
        return cast_config(cls, config)
