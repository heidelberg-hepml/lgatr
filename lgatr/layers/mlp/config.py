"""Configuration dataclass for the geometric MLP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class MLPConfig:
    """Geometric-MLP configuration.

    Parameters
    ----------
    activation
        Which (gated) activation function to use. One of ``"relu"``, ``"sigmoid"``, ``"gelu"``,
        ``"silu"``.
    increase_hidden_channels
        Factor by which to increase the number of hidden channels (both multivectors and scalars).
        Vanilla transformers use 4; we use 2 for backward compatibility.
    num_hidden_layers
        Number of hidden layers to create.

    Parameters auto-set by LGATr
    ----------------------------
    mv_channels
        Number of input multivector channels.
    s_channels
        Number of input scalar channels. Use 0 for no scalar stream.
    dropout_prob
        Dropout probability.
    """

    mv_channels: int | None = None
    s_channels: int = 0
    dropout_prob: float | None = None
    activation: str = "gelu"
    increase_hidden_channels: int = 4
    num_hidden_layers: int = 1

    @classmethod
    def cast(cls, config: Any) -> MLPConfig:
        """Cast an arbitrary object to an :class:`MLPConfig`."""
        if isinstance(config, MLPConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")
