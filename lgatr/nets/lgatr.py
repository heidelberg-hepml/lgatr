"""Equivariant transformer for multivector data."""

from collections.abc import Mapping
from dataclasses import replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers.attention.config import SelfAttentionConfig
from ..layers.lgatr_block import LGATrBlock
from ..layers.linear import EquiLinear
from ..layers.mlp.config import MLPConfig
from ..primitives.config import PrimitivesConfig
from ..utils.autocast import naive_amp
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
    primitives
        LGATr primitives configuration. Accepts a :class:`PrimitivesConfig` instance, a dict,
        or ``None`` (uses defaults).
    reinsert_mv_channels
        If not None, specifies multivector channels that will be reinserted in every attention layer.
    reinsert_s_channels
        If not None, specifies scalar channels that will be reinserted in every attention layer.
    dropout_prob
        Dropout probability.
    norm_elementwise_affine
        Whether the block :class:`EquiLayerNorm` instances learn an affine gain.
    checkpoint_blocks
        Whether to use gradient checkpointing for the blocks. Saves memory at the cost of speed.
    naive_amp
        Whether to bypass the fp32 precision islands so the whole forward runs in the surrounding
        autocast dtype (e.g. bf16). When ``False`` (default), under autocast the multivector stream
        and metric contractions stay fp32 while the scalar GEMMs run in bf16.
    compile
        Whether to wrap the model with :func:`torch.compile`. Primitive caches are warmed
        automatically whenever the model is moved or cast (``.to()``, ``.cuda()``, ``.float()``,
        etc.), so the captured graph is free of host-to-device copies.
    compile_kwargs
        Dict forwarded verbatim to :func:`torch.compile` (via
        :func:`lgatr.utils.compile.compile_model`) when ``compile=True`` (e.g. ``mode``,
        ``dynamic``, ``fullgraph``). Omitted keys fall back to torch's own defaults.
    activation_memory_budget
        Fraction in ``[0, 1]`` forwarded to :func:`lgatr.utils.compile.compile_model` when
        ``compile=True``. ``None`` (the default) leaves torch's global setting untouched. Setting
        ``1.0`` recomputes only cheap pointwise/reduction ops in the backward pass (torch default);
        lower values let the partitioner also recompute compute-intensive ops, ranked by
        memory-saved-per-FLOP, trading backward FLOPs for a smaller activation-memory peak. Smaller
        values (down to ~0.3) reduce the activation-memory peak at a modest backward-compute cost.
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
        primitives: PrimitivesConfig | Mapping | None = None,
        reinsert_mv_channels: tuple[int, ...] | None = None,
        reinsert_s_channels: tuple[int, ...] | None = None,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
        checkpoint_blocks: bool = False,
        naive_amp: bool = False,
        compile: bool = False,
        compile_kwargs: Mapping | None = None,
        activation_memory_budget: float | None = None,
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
        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels
        self._checkpoint_blocks = checkpoint_blocks
        self._naive_amp = naive_amp

        if compile:
            compile_model(
                self,
                compile_kwargs=compile_kwargs,
                activation_memory_budget=activation_memory_budget,
            )

    def _apply(self, fn, *args, **kwargs):
        """Warm primitive caches after every ``.to()`` / ``.cuda()`` / ``.float()`` / etc."""
        super()._apply(fn, *args, **kwargs)
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
        with naive_amp(self._naive_amp):
            return self._forward(multivectors, scalars, **attn_kwargs)

    def _forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor | None = None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
