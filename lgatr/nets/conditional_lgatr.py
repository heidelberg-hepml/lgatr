"""Equivariant conditional transformer for multivector data."""

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers import (
    CrossAttentionConfig,
    SelfAttentionConfig,
    ConditionalLGATrBlock,
    EquiLinear,
)
from ..layers.mlp.config import MLPConfig


class ConditionalLGATr(nn.Module):
    """Conditional L-GATr network.
    Assumes that the condition is already preprocessed, e.g. with a non-conditional `LGATr` network.

    It combines num_blocks conditional L-GATr transformer blocks, each consisting of geometric self-attention
    layers, geometric cross-attention layers, a geometric MLP, residual connections, and normalization layers.
    In addition, there are initial and final equivariant linear layers.

    Parameters
    ----------
    num_blocks : int
        Number of transformer blocks.
    in_mv_channels : int
        Number of input multivector channels.
    condition_mv_channels : int
        Number of condition multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    condition_s_channels : None or int
        If not None, sets the number of scalar condition channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    attention: Dict
        Data for SelfAttentionConfig.
    crossattention: Dict
        Data for CrossAttentionConfig.
    mlp: Dict
        Data for MLPConfig.
    dropout_prob : float or None
        Dropout probability.
    checkpoint_blocks : bool
        Whether to use checkpointing for the transformer blocks to save memory.
    """

    def __init__(
        self,
        num_blocks: int,
        in_mv_channels: int,
        condition_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        condition_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        attention: SelfAttentionConfig,
        crossattention: CrossAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
        checkpoint_blocks: bool = False,
    ) -> None:
        super().__init__()

        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )

        attention = SelfAttentionConfig.cast(attention)
        crossattention = CrossAttentionConfig.cast(crossattention)
        mlp = MLPConfig.cast(mlp)

        self.blocks = nn.ModuleList(
            [
                ConditionalLGATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    condition_mv_channels=condition_mv_channels,
                    condition_s_channels=condition_s_channels,
                    attention=attention,
                    crossattention=crossattention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._checkpoint_blocks = checkpoint_blocks

    def forward(
        self,
        multivectors: torch.Tensor,
        multivectors_condition: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        scalars_condition: Optional[torch.Tensor] = None,
        attn_kwargs={},
        crossattn_kwargs={},
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor
            Input multivectors with shape (..., items, in_mv_channels, 16).
        multivectors_condition : torch.Tensor
            Input condition multivectors with shape (..., items, in_mv_channels, 16).
        scalars : None or torch.Tensor
            Optional input scalars with shape (..., items, in_s_channels).
        scalars_condition : None or torch.Tensor
            Optional input scalars  with shape (..., items, in_s_channels).
        attn_kwargs: None or torch.Tensor or AttentionBias
            Optional attention arguments.
        crossattn_kwargs: None or torch.Tensor or AttentionBias
            Optional attention arguments for the condition.

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors  with shape (..., items, out_mv_channels, 16).
        outputs_s : None or torch.Tensor
            Output scalars with shape (..., items, out_s_channels). None if out_s_channels=None.
        """

        # Decode condition into main track with
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    multivectors_condition=multivectors_condition,
                    scalars_condition=scalars_condition,
                    attn_kwargs=attn_kwargs,
                    crossattn_kwargs=crossattn_kwargs,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    multivectors_condition=multivectors_condition,
                    scalars_condition=scalars_condition,
                    attn_kwargs=attn_kwargs,
                    crossattn_kwargs=crossattn_kwargs,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s
