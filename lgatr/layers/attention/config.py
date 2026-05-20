"""Configuration dataclasses for self- and cross-attention layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class SelfAttentionConfig:
    """Configuration for self-attention.

    Parameters
    ----------
    num_heads
        Number of attention heads.
    multi_query
        Whether to do multi-query attention. Multi-query attention decreases memory consumption
        and parameter count by using a single set of keys and values for all heads.
    increase_hidden_channels
        Factor by which to increase the number of hidden channels (both multivectors and scalars).
        Vanilla transformers use 1; we use 2 for backward compatibility.
    head_scale
        Whether to use HeadScaleMHA following the NormFormer
        (https://arxiv.org/pdf/2110.09456). Each head is scaled by a learnable parameter before
        the heads are combined.

    Parameters auto-set by LGATr
    ----------------------------
    in_mv_channels
        Number of input multivector channels.
    out_mv_channels
        Number of output multivector channels.
    in_s_channels
        Input scalar channels. Use 0 for no scalar inputs.
    out_s_channels
        Output scalar channels. Use 0 for no scalar outputs.
    additional_qk_mv_channels
        Whether additional multivector features for the keys and queries will be provided.
    additional_qk_s_channels
        Whether additional scalar features for the keys and queries will be provided.
    output_init
        Initialization scheme for final linear layer.
    dropout_prob
        Dropout probability.
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
        """Number of hidden multivector channels."""
        return max(self.increase_hidden_channels * self.in_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> int:
        """Number of hidden scalar channels (0 if no scalar stream)."""
        if self.in_s_channels == 0:
            return 0

        return max(self.increase_hidden_channels * self.in_s_channels // self.num_heads, 4)

    @classmethod
    def cast(cls, config: Any) -> SelfAttentionConfig:
        """Cast an arbitrary object to a :class:`SelfAttentionConfig`."""
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
    num_heads
        Number of attention heads.
    multi_query
        Whether to do multi-query attention. Multi-query attention decreases memory consumption
        and parameter count by using a single set of keys and values for all heads.
    increase_hidden_channels
        Factor by which to increase the number of hidden channels (both multivectors and scalars).
        Vanilla transformers use 1; we use 2 for backward compatibility.
    head_scale
        Whether to use HeadScaleMHA following the NormFormer
        (https://arxiv.org/pdf/2110.09456). Each head is scaled by a learnable parameter before
        the heads are combined.

    Parameters auto-set by LGATr
    ----------------------------
    q_mv_channels
        Number of input query multivector channels.
    kv_mv_channels
        Number of input key/value multivector channels.
    out_mv_channels
        Number of output multivector channels.
    q_s_channels
        Input query scalar channels. Use 0 for no scalar inputs.
    kv_s_channels
        Input key/value scalar channels. Use 0 for no scalar inputs.
    out_s_channels
        Output scalar channels. Use 0 for no scalar outputs.
    additional_q_mv_channels
        Whether additional multivector features for the queries will be provided.
    additional_q_s_channels
        Whether additional scalar features for the queries will be provided.
    additional_k_mv_channels
        Whether additional multivector features for the keys will be provided.
    additional_k_s_channels
        Whether additional scalar features for the keys will be provided.
    output_init
        Initialization scheme for final linear layer.
    dropout_prob
        Dropout probability.
    """

    q_mv_channels: int | None = None
    kv_mv_channels: int | None = None
    out_mv_channels: int | None = None
    out_s_channels: int = 0
    q_s_channels: int = 0
    kv_s_channels: int = 0
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
        """Number of hidden multivector channels."""
        return max(self.increase_hidden_channels * self.q_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> int:
        """Number of hidden scalar channels (0 if no scalar stream)."""
        if self.q_s_channels == 0:
            assert self.kv_s_channels == 0
            return 0

        return max(self.increase_hidden_channels * self.q_s_channels // self.num_heads, 4)

    @classmethod
    def cast(cls, config: Any) -> CrossAttentionConfig:
        """Cast an arbitrary object to a :class:`CrossAttentionConfig`."""
        if isinstance(config, CrossAttentionConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")
