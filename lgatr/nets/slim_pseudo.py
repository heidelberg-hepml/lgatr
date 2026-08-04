"""Equivariant transformer for vector, scalar, and pseudoscalar data."""

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers.slim_layers import _require_scalars
from ..layers.slim_pseudo_layers import SlimPseudoBlock, SlimPseudoLinear
from ..utils.autocast import naive_amp
from ..utils.compile import compile_model


class LGATrSlimPseudo(nn.Module):
    """L-GATr-slim network with an additional pseudoscalar stream.

    A slimmer L-GATr variant that operates on Lorentz vectors, scalars, and pseudoscalars (no full
    multivector representation). Stacks ``num_blocks`` :class:`SlimPseudoBlock` modules between
    initial and final :class:`SlimPseudoLinear` layers. Usually instantiated indirectly via
    :class:`~lgatr.nets.slim.LGATrSlim` with nonzero pseudoscalar channels.

    Unlike :class:`~lgatr.nets.slim.LGATrSlim`, the hidden layers keep vectors in the
    ``(..., channels, 4)`` layout, because the parity-odd primitives contract over the four-vector
    index.

    Parameters
    ----------
    num_blocks
        Number of Lorentz-transformer blocks.
    in_v_channels
        Number of input vector channels.
    out_v_channels
        Number of output vector channels.
    hidden_v_channels
        Number of hidden vector channels.
    in_s_channels
        Number of input scalar channels.
    out_s_channels
        Number of output scalar channels.
    hidden_s_channels
        Number of hidden scalar channels.
    num_heads
        Number of attention heads.
    in_p_channels
        Number of input pseudoscalar channels.
    out_p_channels
        Number of output pseudoscalar channels.
    hidden_p_channels
        Number of hidden pseudoscalar channels.
    nonlinearity
        Nonlinearity for the MLP layers.
    nonlinearity_v
        Optional override for the vector-path gate nonlinearity in every GLU. ``None`` falls
        back to ``nonlinearity``.
    mlp_ratio
        Expansion ratio for MLP hidden channels.
    attn_ratio
        Expansion ratio for attention hidden channels.
    num_layers_mlp
        Number of layers in each MLP (must be ``>= 2``).
    dropout_prob
        Dropout probability.
    norm_elementwise_affine
        Whether the block :class:`SlimPseudoRMSNorm` instances learn a per-channel gain.
    checkpoint_blocks
        Whether to use gradient checkpointing for the blocks.
    cp_triple_product
        Enable the lower-rank lab-frame triple-product CP-odd primitive in every linear layer (see
        :class:`~lgatr.layers.slim_pseudo_layers.VectorToTripleProduct`). Defaults to ``False``.
    cp_scalar_pseudo_mixing
        Enable scalar-context modulation of the pseudoscalar path in every linear layer. Defaults
        to ``False``.
    naive_amp
        Whether to bypass the fp32 precision islands so the whole forward runs in the surrounding
        autocast dtype (e.g. bf16). When ``False`` (default), under autocast the vector stream and
        metric contractions stay fp32 while the scalar GEMMs run in bf16.
    compile
        Whether to wrap the model with :func:`torch.compile`.
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
        in_v_channels: int,
        out_v_channels: int,
        hidden_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        hidden_s_channels: int,
        num_heads: int,
        in_p_channels: int = 0,
        out_p_channels: int = 0,
        hidden_p_channels: int = 0,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
        checkpoint_blocks: bool = False,
        cp_triple_product: bool = False,
        cp_scalar_pseudo_mixing: bool = False,
        naive_amp: bool = False,
        compile: bool = False,
        compile_kwargs: Mapping | None = None,
        activation_memory_budget: float | None = None,
    ) -> None:
        super().__init__()
        self._in_p_channels = in_p_channels
        self._naive_amp = naive_amp

        self.linear_in = SlimPseudoLinear(
            in_v_channels=in_v_channels,
            in_s_channels=in_s_channels,
            in_p_channels=in_p_channels,
            out_v_channels=hidden_v_channels,
            out_s_channels=hidden_s_channels,
            out_p_channels=hidden_p_channels,
            cp_triple_product=cp_triple_product,
            cp_scalar_pseudo_mixing=cp_scalar_pseudo_mixing,
        )

        self.blocks = nn.ModuleList(
            [
                SlimPseudoBlock(
                    v_channels=hidden_v_channels,
                    s_channels=hidden_s_channels,
                    p_channels=hidden_p_channels,
                    num_heads=num_heads,
                    nonlinearity=nonlinearity,
                    nonlinearity_v=nonlinearity_v,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    num_layers_mlp=num_layers_mlp,
                    dropout_prob=dropout_prob,
                    norm_elementwise_affine=norm_elementwise_affine,
                    cp_triple_product=cp_triple_product,
                    cp_scalar_pseudo_mixing=cp_scalar_pseudo_mixing,
                )
                for _ in range(num_blocks)
            ]
        )

        self.linear_out = SlimPseudoLinear(
            in_v_channels=hidden_v_channels,
            in_s_channels=hidden_s_channels,
            in_p_channels=hidden_p_channels,
            out_v_channels=out_v_channels,
            out_s_channels=out_s_channels,
            out_p_channels=out_p_channels,
            cp_triple_product=cp_triple_product,
            cp_scalar_pseudo_mixing=cp_scalar_pseudo_mixing,
        )
        self._checkpoint_blocks = checkpoint_blocks

        if compile:
            compile_model(
                self,
                compile_kwargs=compile_kwargs,
                activation_memory_budget=activation_memory_budget,
            )

    def forward(
        self,
        vectors: torch.Tensor,
        scalars: torch.Tensor,
        pseudoscalars: torch.Tensor | None = None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., items, in_v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., items, in_s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., items, in_p_channels)``. May be ``None`` only
            when the model expects no input pseudoscalar channels.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, out_v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., items, out_s_channels)``.
        outputs_p
            Pseudoscalar features of shape ``(..., items, out_p_channels)``.
        """
        _require_scalars(scalars=scalars)
        with naive_amp(self._naive_amp):
            return self._forward(vectors, scalars, pseudoscalars, **attn_kwargs)

    def _forward(
        self,
        vectors: torch.Tensor,
        scalars: torch.Tensor,
        pseudoscalars: torch.Tensor | None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pseudoscalars is None:
            assert self._in_p_channels == 0, (
                "Pseudoscalar input cannot be None if the model expects pseudoscalar channels."
            )
            pseudoscalars = scalars.new_zeros(*scalars.shape[:-1], 0)

        h_v, h_s, h_p = self.linear_in(vectors, scalars, pseudoscalars)

        for block in self.blocks:
            if self._checkpoint_blocks:
                h_v, h_s, h_p = checkpoint(block, h_v, h_s, h_p, use_reentrant=False, **attn_kwargs)
            else:
                h_v, h_s, h_p = block(h_v, h_s, h_p, **attn_kwargs)

        outputs_v, outputs_s, outputs_p = self.linear_out(h_v, h_s, h_p)
        return outputs_v, outputs_s, outputs_p
