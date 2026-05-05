"""Equivariant conditional transformer for multivector data."""

from collections.abc import Mapping

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
from ..primitives.config import PrimitivesConfig
from ..utils.compile import compile_model, warmup_after_apply


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
        Number of scalar hidden channels. Use 0 for no scalar stream in the hidden layers.
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
    checkpoint_blocks
        Whether to use gradient checkpointing for the transformer blocks.
    compile
        Whether to wrap the model with :func:`torch.compile`. Primitive caches are warmed
        automatically whenever the model is moved or cast (``.to()``, ``.cuda()``, ``.float()``,
        etc.), so the captured graph is free of host-to-device copies.
    **compile_kwargs
        Forwarded to :func:`lgatr.utils.compile.compile_model` when ``compile=True``;
        see there for the supported keys (``compile_mode``, ``compile_dynamic``,
        ``compile_fullgraph``) and their defaults.
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
        attention: SelfAttentionConfig,
        crossattention: CrossAttentionConfig,
        mlp: MLPConfig,
        primitives: PrimitivesConfig | Mapping | None = None,
        dropout_prob: float | None = None,
        checkpoint_blocks: bool = False,
        compile: bool = False,
        **compile_kwargs,
    ) -> None:
        super().__init__()
        primitives = PrimitivesConfig.cast(primitives)
        self.primitives = primitives

        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            primitives,
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
                    mv_channels_cond=mv_channels_cond,
                    s_channels_cond=s_channels_cond,
                    attention=attention,
                    crossattention=crossattention,
                    mlp=mlp,
                    primitives=primitives,
                    dropout_prob=dropout_prob,
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

        if compile:
            compile_model(self, **compile_kwargs)

    def _apply(self, fn, recurse=True):
        """Warm primitive caches after every ``.to()`` / ``.cuda()`` / ``.float()`` / etc."""
        super()._apply(fn, recurse=recurse)
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
