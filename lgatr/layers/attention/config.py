"""Configuration dataclasses for self- and cross-attention layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...utils.config import cast_config


@dataclass
class SelfAttentionConfig:
    """Configuration for self-attention.

    Parameters
    ----------
    in_mv_channels
        Number of input multivector channels. Set automatically by the parent network.
    out_mv_channels
        Number of output multivector channels. Set automatically by the parent network.
    in_s_channels
        Input scalar channels. Use 0 for no scalar inputs. Set automatically by the parent network.
    out_s_channels
        Output scalar channels. Use 0 for no scalar outputs. Set automatically by the parent
        network.
    additional_qk_mv_channels
        Whether additional multivector features for the keys and queries will be provided. Set
        automatically by the parent network.
    additional_qk_s_channels
        Whether additional scalar features for the keys and queries will be provided. Set
        automatically by the parent network.
    output_init
        Initialization scheme for final linear layer. Set automatically by the parent network.
    dropout_prob
        Dropout probability. Set automatically by the parent network.
    num_heads
        Number of attention heads.
    multi_query
        Whether to do multi-query attention. Multi-query attention decreases memory consumption
        and parameter count by using a single set of keys and values for all heads.
    attn_ratio
        Factor by which to increase the number of hidden channels (both multivectors and scalars).
        Vanilla transformers use 1; for multi-query 2 is more natural.
    head_scale
        Whether to use HeadScaleMHA following the NormFormer
        (https://arxiv.org/pdf/2110.09456). Each head is scaled by a learnable parameter before
        the heads are combined.
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
    attn_ratio: int = 1
    head_scale: bool = False

    @property
    def hidden_mv_channels(self) -> int:
        """Number of hidden multivector channels."""
        return max(self.attn_ratio * self.in_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> int:
        """Number of hidden scalar channels (0 if no scalar stream)."""
        if self.in_s_channels == 0:
            return 0

        return max(self.attn_ratio * self.in_s_channels // self.num_heads, 4)

    @classmethod
    def cast(cls, config: Any) -> SelfAttentionConfig:
        """Cast a :class:`SelfAttentionConfig` or mapping to a :class:`SelfAttentionConfig`."""
        return cast_config(cls, config)


@dataclass
class CrossAttentionConfig:
    """Configuration for cross-attention.

    Parameters
    ----------
    q_mv_channels
        Number of input query multivector channels. Set automatically by the parent network.
    kv_mv_channels
        Number of input key/value multivector channels. Set automatically by the parent network.
    out_mv_channels
        Number of output multivector channels. Set automatically by the parent network.
    out_s_channels
        Output scalar channels. Use 0 for no scalar outputs. Set automatically by the parent
        network.
    q_s_channels
        Input query scalar channels. Use 0 for no scalar inputs. Set automatically by the parent
        network.
    kv_s_channels
        Input key/value scalar channels. Use 0 for no scalar inputs. Set automatically by the
        parent network.
    output_init
        Initialization scheme for final linear layer. Set automatically by the parent network.
    dropout_prob
        Dropout probability. Set automatically by the parent network.
    num_heads
        Number of attention heads.
    multi_query
        Whether to do multi-query attention. Multi-query attention decreases memory consumption
        and parameter count by using a single set of keys and values for all heads.
    attn_ratio
        Factor by which to increase the number of hidden channels (both multivectors and scalars).
        Vanilla transformers use 1; for multi-query 2 is more natural.
    head_scale
        Whether to use HeadScaleMHA following the NormFormer
        (https://arxiv.org/pdf/2110.09456). Each head is scaled by a learnable parameter before
        the heads are combined.
    """

    q_mv_channels: int | None = None
    kv_mv_channels: int | None = None
    out_mv_channels: int | None = None
    out_s_channels: int = 0
    q_s_channels: int = 0
    kv_s_channels: int = 0
    output_init: str = "default"
    dropout_prob: float | None = None
    num_heads: int = 8
    multi_query: bool = False
    attn_ratio: int = 1
    head_scale: bool = False

    @property
    def hidden_mv_channels(self) -> int:
        """Number of hidden multivector channels."""
        return max(self.attn_ratio * self.q_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> int:
        """Number of hidden scalar channels (0 if no scalar stream)."""
        if self.q_s_channels == 0:
            assert self.kv_s_channels == 0
            return 0

        return max(self.attn_ratio * self.q_s_channels // self.num_heads, 4)

    @classmethod
    def cast(cls, config: Any) -> CrossAttentionConfig:
        """Cast a :class:`CrossAttentionConfig` or mapping to a :class:`CrossAttentionConfig`."""
        return cast_config(cls, config)
