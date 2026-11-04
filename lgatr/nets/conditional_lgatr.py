"""Equivariant conditional transformer for multivector data."""

from collections.abc import Mapping
from dataclasses import replace

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
from ..primitives.compile import warmup_after_apply
from ..primitives.config import PrimitivesConfig


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
    mv_channels_cond
        Number of condition multivector channels.
    out_mv_channels
        Number of output multivector channels.
    hidden_mv_channels
        Number of hidden multivector channels.
    in_s_channels
        Number of scalar input channels. Use 0 for no scalar inputs.
    s_channels_cond
        Number of scalar condition channels. Use 0 for no scalar condition stream.
    out_s_channels
        Number of scalar output channels. Use 0 for no scalar outputs.
    hidden_s_channels
        Number of scalar hidden channels.
    attention
        Self-attention configuration.
    crossattention
        Cross-attention configuration.
    mlp
        MLP configuration.
    primitives
        LGATr primitives configuration. Accepts a :class:`PrimitivesConfig` instance, a dict,
        or ``None`` (uses defaults).
    dropout_prob
        Dropout probability.
    norm_elementwise_affine
        Whether the block :class:`EquiLayerNorm` instances learn an affine gain.
    checkpoint_blocks
        Whether to use gradient checkpointing for the transformer blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        in_mv_channels: int,
        mv_channels_cond: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: int,
        s_channels_cond: int,
        out_s_channels: int,
        hidden_s_channels: int,
        attention: SelfAttentionConfig | Mapping,
        crossattention: CrossAttentionConfig | Mapping,
        mlp: MLPConfig | Mapping,
        primitives: PrimitivesConfig | Mapping | None = None,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
        checkpoint_blocks: bool = False,
    ) -> None:
        super().__init__()
        primitives = PrimitivesConfig() if primitives is None else PrimitivesConfig.cast(primitives)
        self.primitives = primitives

        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            primitives,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )

        # ConditionalLGATr has no reinsert_* channels, so there are never additional qk features.
        attention = replace(
            SelfAttentionConfig.cast(attention),
            additional_qk_mv_channels=0,
            additional_qk_s_channels=0,
        )
        crossattention = CrossAttentionConfig.cast(crossattention)
        mlp = MLPConfig.cast(mlp)

        self.blocks = nn.ModuleList(
            [
                ConditionalLGATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    mv_channels_cond=mv_channels_cond,
                    s_channels_cond=s_channels_cond,
                    attention=attention,
                    crossattention=crossattention,
                    mlp=mlp,
                    primitives=primitives,
                    dropout_prob=dropout_prob,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            primitives,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._checkpoint_blocks = checkpoint_blocks

    def _apply(self, fn, *args, **kwargs):
        """Warm primitive caches after every ``.to()`` / ``.cuda()`` / ``.float()`` / etc."""
        super()._apply(fn, *args, **kwargs)
        warmup_after_apply(self)
        return self

    def forward(
        self,
        multivectors: torch.Tensor,
        multivectors_cond: torch.Tensor,
        scalars: torch.Tensor | None = None,
        scalars_cond: torch.Tensor | None = None,
        attn_kwargs: dict | None = None,
        crossattn_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        multivectors
            Input multivectors of shape ``(..., items, in_mv_channels, 16)``.
        multivectors_cond
            Condition multivectors of shape ``(..., items_cond, mv_channels_cond, 16)``.
        scalars
            Optional input scalars of shape ``(..., items, in_s_channels)``.
        scalars_cond
            Optional condition scalars of shape ``(..., items_cond, s_channels_cond)``.
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

        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    multivectors_cond=multivectors_cond,
                    scalars_cond=scalars_cond,
                    attn_kwargs=attn_kwargs,
                    crossattn_kwargs=crossattn_kwargs,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    multivectors_cond=multivectors_cond,
                    scalars_cond=scalars_cond,
                    attn_kwargs=attn_kwargs,
                    crossattn_kwargs=crossattn_kwargs,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s
