from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class MLPConfig:
    """Geometric MLP configuration.

    Parameters
    ----------
    mv_channels : int
        Number of input multivector channels.
    s_channels : int
        Number of input scalar channels.
    activation : {"relu", "sigmoid", "gelu"}
        Which (gated) activation function to use
    increase_hidden_channels : int
        Factor by which to increase the number of hidden channels (both multivectors and scalars)
    num_hidden_layers : int
        Number of hidden layers to create
    dropout_prob : float or None
        Dropout probability
    """

    mv_channels: Optional[int] = None
    s_channels: Optional[int] = None
    activation: str = "gelu"
    increase_hidden_channels: int = 2
    num_hidden_layers: int = 1
    dropout_prob: Optional[float] = None

    def __post_init__(self):
        """Type checking / conversion."""
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in [
            "null",
            "none",
        ]:
            self.dropout_prob = None

    @classmethod
    def cast(cls, config: Any) -> MLPConfig:
        """Casts an object as MLPConfig."""
        if isinstance(config, MLPConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")
