"""L-GATr encoder block."""
from dataclasses import replace
from typing import Optional, Tuple

import torch
from torch import nn

from .attention import SelfAttention, SelfAttentionConfig
from .layer_norm import EquiLayerNorm
from .mlp.config import MLPConfig
from .mlp.mlp import GeoMLP


class LGATrBlock(nn.Module):
    """L-GATr encoder block.

    Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
    self-attention, and a residual connection. Then the data is processed by a block consisting of
    another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and another
    residual connection.

    Parameters
    ----------
    mv_channels : int
        Number of input and output multivector channels
    s_channels: int
        Number of input and output scalar channels
    attention: SelfAttentionConfig
        Attention configuration
    mlp: MLPConfig
        MLP configuration
    dropout_prob : float or None
        Dropout probability
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
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
        scalars: torch.Tensor,
        additional_qk_features_mv=None,
        additional_qk_features_s=None,
        **attn_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer encoder block.

        Parameters
        ----------
        multivectors : torch.Tensor
            Input multivectors with shape (..., items, mv_channels, 16).
        scalars : torch.Tensor
            Input scalars with shape (..., items, s_channels).
        additional_qk_features_mv : None or torch.Tensor
            Additional multivector Q/K features with shape (..., items, add_qk_mv_channels, 16).
        additional_qk_features_s : None or torch.Tensor
            Additional scalar Q/K features with shape (..., items, add_qk_s_channels, 16).
        **attn_kwargs
            Optional keyword arguments passed to attention.

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors with shape (..., items, mv_channels, 16).
        output_scalars : torch.Tensor
            Output scalars  with shape (..., items, s_channels).
        """

        # Attention block: pre layer norm
        h_mv, h_s = self.norm(multivectors, scalars=scalars)

        # Attention block: self attention
        h_mv, h_s = self.attention(
            h_mv,
            scalars=h_s,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            **attn_kwargs,
        )

        # Attention block: skip connection
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
