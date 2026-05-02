"""Equivariant conditional transformer for multivector data."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers import (
    ConditionalLGATrBlock,
    CrossAttentionConfig,
    EquiLinear,
    SelfAttentionConfig,
)
from ..layers.mlp.config import MLPConfig


class ConditionalLGATr(nn.Module):
    """Conditional L-GATr network.

    Combines ``num_blocks`` :class:`~lgatr.layers.conditional_lgatr_block.ConditionalLGATrBlock`
    modules (geometric self-attention, cross-attention, geometric MLP, residual connections,
    normalization) with initial and final equivariant linear layers. The condition is expected to
    be already preprocessed (e.g. by a non-conditional :class:`LGATr` network).

    Parameters
    ----------
    num_blocks
        Number of transformer blocks.
    in_mv_channels
        Number of input multivector channels.
    condition_mv_channels
        Number of condition multivector channels.
    out_mv_channels
        Number of output multivector channels.
    hidden_mv_channels
        Number of hidden multivector channels.
    in_s_channels
        Number of scalar input channels. Use 0 for no scalar inputs.
    condition_s_channels
        Number of scalar condition channels. Use 0 for no scalar condition stream.
    out_s_channels
        Number of scalar output channels. Use 0 for no scalar outputs.
    hidden_s_channels
        Number of scalar hidden channels. Use 0 for no scalar stream in the hidden layers.
    attention
        Self-attention configuration.
    crossattention
        Cross-attention configuration.
    mlp
        MLP configuration.
    dropout_prob
        Dropout probability.
    checkpoint_blocks
        Whether to use gradient checkpointing for the transformer blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        in_mv_channels: int,
        condition_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: int,
        condition_s_channels: int,
        out_s_channels: int,
        hidden_s_channels: int,
        attention: SelfAttentionConfig,
        crossattention: CrossAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: float | None = None,
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
        scalars: torch.Tensor | None = None,
        scalars_condition: torch.Tensor | None = None,
        attn_kwargs: dict | None = None,
        crossattn_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        multivectors
            Input multivectors of shape ``(..., items, in_mv_channels, 16)``.
        multivectors_condition
            Condition multivectors of shape ``(..., items_condition, condition_mv_channels, 16)``.
        scalars
            Optional input scalars of shape ``(..., items, in_s_channels)``.
        scalars_condition
            Optional condition scalars of shape ``(..., items_condition, condition_s_channels)``.
        attn_kwargs
            Optional keyword arguments forwarded to self-attention.
        crossattn_kwargs
            Optional keyword arguments forwarded to cross-attention.

        Returns
        -------
        outputs_mv
            Output multivectors of shape ``(..., items, out_mv_channels, 16)``.
        outputs_s
            Output scalars of shape ``(..., items, out_s_channels)``, or None if
            ``out_s_channels == 0``.
        """
        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        crossattn_kwargs = crossattn_kwargs if crossattn_kwargs is not None else {}

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
