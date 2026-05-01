from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class SelfAttentionConfig:
    """Configuration for self-attention.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    multi_query: bool
        Whether to do multi-query attention, default is False.
        Multi-query attention decreases memory consumption and parameter count
        by using a single set of keys and values for all heads.
    increase_hidden_channels : int
        Factor by which to increase the number of hidden channels (both multivectors and scalars).
        Vanilla transformers use 1, we use 2 for backward compatibility.
    head_scale: bool
        Whether to use HeadScaleMHA following the NormFormer, see https://arxiv.org/pdf/2110.09456.
        Before combining the heads, each head is scaled by a learnable parameter.


    Parameters auto-set by LGATr
    ----------------------------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    in_s_channels : int
        Input scalar channels. Use 0 for no scalar inputs.
    out_s_channels : int
        Output scalar channels. Use 0 for no scalar outputs.
    additional_qk_mv_channels : int
        Whether additional multivector features for the keys and queries will be provided.
    additional_qk_s_channels : int
        Whether additional scalar features for the keys and queries will be provided.
    output_init : str
        Initialization scheme for final linear layer
    dropout_prob : float or None
        Dropout probability
    """

    in_mv_channels: int | None = None
    out_mv_channels: int | None = None
    in_s_channels: int = 0
    out_s_channels: int = 0
    additional_qk_mv_channels: int = 0
    additional_qk_s_channels: int = 0
    output_init: str = "default"
    dropout_prob: float | None = None
    num_heads: int = 8
    multi_query: bool = False
    increase_hidden_channels: int = 1
    head_scale: bool = False

    @property
    def hidden_mv_channels(self) -> int | None:
        """Returns the number of hidden multivector channels."""

        return max(self.increase_hidden_channels * self.in_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> int:
        """Returns the number of hidden scalar channels (0 if no scalar stream)."""

        if self.in_s_channels == 0:
            return 0

        return max(self.increase_hidden_channels * self.in_s_channels // self.num_heads, 4)

    @classmethod
    def cast(cls, config: Any) -> SelfAttentionConfig:
        """Casts an object as SelfAttentionConfig."""
        if isinstance(config, SelfAttentionConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")


@dataclass
class CrossAttentionConfig:
    """Configuration for cross-attention.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    multi_query: bool
        Whether to do multi-query attention, default is False.
        Multi-query attention decreases memory consumption and parameter count
        by using a single set of keys and values for all heads.
    increase_hidden_channels : int
        Factor by which to increase the number of hidden channels (both multivectors and scalars).
        Vanilla transformers use 1, we use 2 for backward compatibility.
    head_scale: bool
        Whether to use HeadScaleMHA following the NormFormer, see https://arxiv.org/pdf/2110.09456.
        Before combining the heads, each head is scaled by a learnable parameter.

    Parameters auto-set by LGATr
    ----------------------------
    in_q_mv_channels : int
        Number of input query multivector channels.
    in_kv_mv_channels : int
        Number of input key/value multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    in_q_s_channels : int
        Input query scalar channels. Use 0 for no scalar inputs.
    in_kv_s_channels : int
        Input key/value scalar channels. Use 0 for no scalar inputs.
    out_s_channels : int
        Output scalar channels. Use 0 for no scalar outputs.
    additional_q_mv_channels : int
        Whether additional multivector features for the queries will be provided.
    additional_q_s_channels : int
        Whether additional scalar features for the queries will be provided.
    additional_k_mv_channels : int
        Whether additional multivector features for the keys will be provided.
    additional_k_s_channels : int
        Whether additional scalar features for the keys will be provided.
    output_init : str
        Initialization scheme for final linear layer
    dropout_prob : float or None
        Dropout probability
    """

    in_q_mv_channels: int | None = None
    in_kv_mv_channels: int | None = None
    out_mv_channels: int | None = None
    out_s_channels: int = 0
    in_q_s_channels: int = 0
    in_kv_s_channels: int = 0
    additional_q_mv_channels: int = 0
    additional_q_s_channels: int = 0
    additional_k_mv_channels: int = 0
    additional_k_s_channels: int = 0
    output_init: str = "default"
    dropout_prob: float | None = None
    num_heads: int = 8
    multi_query: bool = False
    increase_hidden_channels: int = 1
    head_scale: bool = False

    @property
    def hidden_mv_channels(self) -> int | None:
        """Returns the number of hidden multivector channels."""

        return max(self.increase_hidden_channels * self.in_q_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> int:
        """Returns the number of hidden scalar channels (0 if no scalar stream)."""

        if self.in_q_s_channels == 0:
            assert self.in_kv_s_channels == 0
            return 0

        return max(self.increase_hidden_channels * self.in_q_s_channels // self.num_heads, 4)

    @classmethod
    def cast(cls, config: Any) -> CrossAttentionConfig:
        """Casts an object as CrossAttentionConfig."""
        if isinstance(config, CrossAttentionConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")
