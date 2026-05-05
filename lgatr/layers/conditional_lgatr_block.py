"""L-GATr decoder block."""

from dataclasses import replace

import torch
from torch import nn

from ..utils.misc import residual_add
from .attention import (
    CrossAttention,
    CrossAttentionConfig,
    SelfAttention,
    SelfAttentionConfig,
)
from .layer_norm import EquiLayerNorm
from .mlp.config import MLPConfig
from .mlp.mlp import GeoMLP


class ConditionalLGATrBlock(nn.Module):
    """L-GATr decoder block.

    Inputs are first processed by LayerNorm, multi-head geometric self-attention, and a residual
    connection. Then the conditions are mixed in via cross-attention with the same overhead as
    self-attention. Finally the data goes through another LayerNorm, a two-layer geometric MLP
    with GeLU activations, and another residual connection.

    Parameters
    ----------
    mv_channels
        Number of input and output multivector channels.
    s_channels
        Number of input and output scalar channels. Use 0 for no scalar stream.
    mv_channels_cond
        Number of condition multivector channels.
    s_channels_cond
        Number of condition scalar channels. Use 0 for no scalar condition stream.
    attention
        Self-attention configuration.
    crossattention
        Cross-attention configuration.
    mlp
        MLP configuration.
    dropout_prob
        Dropout probability.
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        mv_channels_cond: int,
        s_channels_cond: int,
        attention: SelfAttentionConfig,
        crossattention: CrossAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: float | None = None,
    ) -> None:
        super().__init__()

        # Normalization layer (stateless, so we can use the same layer for both normalization instances)
        self.norm = EquiLayerNorm()

        # Self-attention layer
        attention = replace(
            attention,
            in_mv_channels=mv_channels,
            out_mv_channels=mv_channels,
            in_s_channels=s_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.attention = SelfAttention(attention)

        # Cross-attention layer
        crossattention = replace(
            crossattention,
            q_mv_channels=mv_channels,
            q_s_channels=s_channels,
            kv_mv_channels=mv_channels_cond,
            kv_s_channels=s_channels_cond,
            out_mv_channels=mv_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.crossattention = CrossAttention(crossattention)

        # MLP block
        mlp = replace(
            mlp,
            mv_channels=mv_channels,
            s_channels=s_channels,
            dropout_prob=dropout_prob,
        )
        self.mlp = GeoMLP(mlp)

    def forward(
        self,
        multivectors: torch.Tensor,
        multivectors_cond: torch.Tensor,
        scalars: torch.Tensor | None = None,
        scalars_cond: torch.Tensor | None = None,
        attn_kwargs: dict | None = None,
        crossattn_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the decoder block.

        Parameters
        ----------
        multivectors
            Input multivectors of shape ``(..., items, mv_channels, 16)``.
        multivectors_cond
            Condition multivectors of shape ``(..., items_cond, mv_channels_cond, 16)``.
        scalars
            Optional input scalars of shape ``(..., items, s_channels)``. If None, the scalar
            stream is bypassed and ``outputs_s`` is None.
        scalars_cond
            Optional condition scalars of shape ``(..., items_cond, s_channels_cond)``.
        attn_kwargs
            Optional keyword arguments forwarded to self-attention (e.g. attention masks).
        crossattn_kwargs
            Optional keyword arguments forwarded to cross-attention (e.g. attention masks).

        Returns
        -------
        outputs_mv
            Output multivectors of shape ``(..., items, mv_channels, 16)``.
        outputs_s
            Output scalars of shape ``(..., items, s_channels)``, or None.
        """
        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        crossattn_kwargs = crossattn_kwargs if crossattn_kwargs is not None else {}

        # Self-attention block: pre layer norm
        h_mv, h_s = self.norm(multivectors, scalars=scalars)

        # Self-attention block: self attention
        h_mv, h_s = self.attention(
            h_mv,
            scalars=h_s,
            **attn_kwargs,
        )

        # Self-attention block: skip connection
        multivectors = multivectors + h_mv
        scalars = residual_add(scalars, h_s)

        # Cross-attention block: pre layer norm
        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        mv_cond, s_cond = self.norm(multivectors_cond, scalars=scalars_cond)

        # Cross-attention block: cross attention
        h_mv, h_s = self.crossattention(
            multivectors_q=h_mv,
            multivectors_kv=mv_cond,
            scalars_q=h_s,
            scalars_kv=s_cond,
            **crossattn_kwargs,
        )

        # Cross-attention block: skip connection
        outputs_mv = multivectors + h_mv
        outputs_s = residual_add(scalars, h_s)

        # MLP block: pre layer norm
        h_mv, h_s = self.norm(outputs_mv, scalars=outputs_s)

        # MLP block: MLP
        h_mv, h_s = self.mlp(h_mv, scalars=h_s)

        # MLP block: skip connection
        outputs_mv = outputs_mv + h_mv
        outputs_s = residual_add(outputs_s, h_s)

        return outputs_mv, outputs_s
