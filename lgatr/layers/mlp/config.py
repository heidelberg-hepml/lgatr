from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class MLPConfig:
    """Geometric MLP configuration.

    Parameters
    ----------
    activation : {"relu", "sigmoid", "gelu"}
        Which (gated) activation function to use.
    increase_hidden_channels : int
        Factor by which to increase the number of hidden channels (both multivectors and scalars).
        Vanilla transformers use 4, we use 2 for backward compatibility.
    num_hidden_layers : int
        Number of hidden layers to create.

    Parameters auto-set by LGATr
    ----------------------------
    mv_channels : int
        Number of input multivector channels.
    s_channels : int
        Number of input scalar channels.
    dropout_prob : float or None
        Dropout probability
    """

    mv_channels: Optional[int] = None
    s_channels: Optional[int] = None
    dropout_prob: Optional[float] = None
    activation: str = "gelu"
    increase_hidden_channels: int = 2
    num_hidden_layers: int = 1

    @classmethod
    def cast(cls, config: Any) -> MLPConfig:
        """Casts an object as MLPConfig."""
        if isinstance(config, MLPConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")
