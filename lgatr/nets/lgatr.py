"""Equivariant transformer for multivector data."""

from dataclasses import replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers.attention.config import SelfAttentionConfig
from ..layers.lgatr_block import LGATrBlock
from ..layers.linear import EquiLinear
from ..layers.mlp.config import MLPConfig
from ..utils.compile import compile_model, warmup_after_apply


class LGATr(nn.Module):
    """L-GATr network.

    Combines ``num_blocks`` :class:`~lgatr.layers.lgatr_block.LGATrBlock` modules (geometric
    self-attention, geometric MLP, residual connections, normalization) with initial and final
    equivariant linear layers.

    Inputs have shape ``(..., items, in_mv_channels, 16)``; outputs have shape
    ``(..., items, out_mv_channels, 16)``; hidden representations have shape
    ``(..., items, hidden_mv_channels, 16)`` (and similar for the optional scalar stream).

    Parameters
    ----------
    num_blocks
        Number of transformer blocks.
    in_mv_channels
        Number of input multivector channels.
    out_mv_channels
        Number of output multivector channels.
    hidden_mv_channels
        Number of hidden multivector channels.
    in_s_channels
        Number of scalar input channels. Use 0 for no scalar inputs.
    out_s_channels
        Number of scalar output channels. Use 0 for no scalar outputs.
    hidden_s_channels
        Number of scalar hidden channels. Use 0 for no scalar stream in the hidden layers.
    attention
        Self-attention configuration (see :class:`~lgatr.layers.attention.config.SelfAttentionConfig`).
    mlp
        MLP configuration (see :class:`~lgatr.layers.mlp.config.MLPConfig`).
    reinsert_mv_channels
        If not None, specifies multivector channels that will be reinserted in every attention layer.
    reinsert_s_channels
        If not None, specifies scalar channels that will be reinserted in every attention layer.
    dropout_prob
        Dropout probability.
    checkpoint_blocks
        Whether to use gradient checkpointing for the blocks. Saves memory at the cost of speed.
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
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        hidden_s_channels: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        reinsert_mv_channels: tuple[int] | None = None,
        reinsert_s_channels: tuple[int] | None = None,
        dropout_prob: float | None = None,
        checkpoint_blocks: bool = False,
        compile: bool = False,
        **compile_kwargs,
    ) -> None:
        super().__init__()
        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = replace(
            SelfAttentionConfig.cast(attention),
            additional_qk_mv_channels=(
                0 if reinsert_mv_channels is None else len(reinsert_mv_channels)
            ),
            additional_qk_s_channels=0 if reinsert_s_channels is None else len(reinsert_s_channels),
        )
        mlp = MLPConfig.cast(mlp)
        self.blocks = nn.ModuleList(
            [
                LGATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    attention=attention,
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
        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels
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
        scalars: torch.Tensor | None = None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        multivectors
            Input multivectors of shape ``(..., items, in_mv_channels, 16)``.
        scalars
            Optional input scalars of shape ``(..., items, in_s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_mv
            Output multivectors of shape ``(..., items, out_mv_channels, 16)``.
        outputs_s
            Output scalars of shape ``(..., items, out_s_channels)``, or None if
            ``out_s_channels == 0``.
        """

        # Channels that will be re-inserted in any query / key computation
        (
            additional_qk_features_mv,
            additional_qk_features_s,
        ) = self._construct_reinserted_channels(multivectors, scalars)

        # Pass through the blocks
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    **attn_kwargs,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    **attn_kwargs,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s

    def _construct_reinserted_channels(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Construct input features that will be reinserted in every attention layer.
        This can be useful to enhance the sensitivity to specific input features, similar to a residual connection.
        """

        if self._reinsert_mv_channels is None:
            additional_qk_features_mv = None
        else:
            additional_qk_features_mv = multivectors[..., self._reinsert_mv_channels, :]

        if self._reinsert_s_channels is None:
            additional_qk_features_s = None
        else:
            assert scalars is not None
            additional_qk_features_s = scalars[..., self._reinsert_s_channels]

        return additional_qk_features_mv, additional_qk_features_s
