"""L-GATr decoder block."""

from dataclasses import replace

import torch
from torch import nn

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

    Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
    self-attention, and a residual connection. Then the conditions are included with
    cross-attention using the same overhead as in the self-attention part.
    Then the data is processed by a block consisting of another LayerNorm,
    an item-wise two-layer geometric MLP with GeLU activations, and another
    residual connection.

    Parameters
    ----------
    mv_channels : int
        Number of input and output multivector channels
    s_channels: int
        Number of input and output scalar channels
    condition_mv_channels: int
        Number of condition multivector channels
    condition_s_channels: int
        Number of condition scalar channels
    attention: SelfAttentionConfig
        Attention configuration
    crossattention: CrossAttentionConfig
        Cross-attention configuration
    mlp: MLPConfig
        MLP configuration
    dropout_prob : float or None
        Dropout probability
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        condition_mv_channels: int,
        condition_s_channels: int,
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
            in_q_mv_channels=mv_channels,
            in_q_s_channels=s_channels,
            in_kv_mv_channels=condition_mv_channels,
            in_kv_s_channels=condition_s_channels,
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
        multivectors_condition: torch.Tensor,
        scalars: torch.Tensor = None,
        scalars_condition: torch.Tensor = None,
        attn_kwargs=None,
        crossattn_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer decoder block.

        Parameters
        ----------
        multivectors : torch.Tensor
            Input multivectors  with shape (..., items, mv_channels, 16).
        scalars : torch.Tensor
            Input scalars with shape (..., items, s_channels).
        multivectors_condition : torch.Tensor
            Input condition multivectors with shape (..., items, mv_channels, 16).
        scalars_condition : torch.Tensor
            Input condition scalars with shape (..., items, s_channels).
        attn_kwargs: None or torch.Tensor or AttentionBias
            Optional attention mask.
        crossattn_kwargs: None or torch.Tensor or AttentionBias
            Optional attention mask for the condition.

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors with shape (..., items, mv_channels, 16).
        output_scalars : torch.Tensor
            Output scalars with shape (..., items, s_channels).
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
        scalars = scalars + h_s

        # Cross-attention block: pre layer norm
        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        c_mv, c_s = self.norm(multivectors_condition, scalars=scalars_condition)

        # Cross-attention block: cross attention
        h_mv, h_s = self.crossattention(
            multivectors_q=h_mv,
            multivectors_kv=c_mv,
            scalars_q=h_s,
            scalars_kv=c_s,
            **crossattn_kwargs,
        )

        # Cross-attention block: skip connection
        outputs_mv = multivectors + h_mv
        outputs_s = scalars + h_s

        # MLP block: pre layer norm
        h_mv, h_s = self.norm(outputs_mv, scalars=outputs_s)

        # MLP block: MLP
        h_mv, h_s = self.mlp(h_mv, scalars=h_s)

        # MLP block: skip connection
        outputs_mv = outputs_mv + h_mv
        outputs_s = outputs_s + h_s

        return outputs_mv, outputs_s
