"""Equivariant transformer for vector, scalar, and pseudoscalar data (layers built on slim_layers).

Select it with ``_target_: lgatr.nets.slim_pseudo_v2.LGATrSlimPseudo``.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers.slim_layers import _require_scalars
from ..layers.slim_pseudo_layers_v2 import (
    SlimPseudoBlock,
    SlimPseudoLinear,
    _freeze_dead_tail,
)
from ..utils.autocast import naive_amp


class LGATrSlimPseudo(nn.Module):
    """L-GATr-slim network with an additional pseudoscalar stream.

    All operations are those of :class:`~lgatr.nets.slim.LGATrSlim`, with the pseudoscalars
    treated like the scalars and gated by absolute values of pseudoscalars. The streams mix
    through one :class:`~lgatr.layers.slim_pseudo_layers.PseudoDeterminant` per block
    (vectors to pseudoscalars through a learned determinant, requires ``hidden_v_channels >= 4``)

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
    in_p_channels
        Number of input pseudoscalar channels.
    out_p_channels
        Number of output pseudoscalar channels.
    hidden_p_channels
        Number of hidden pseudoscalar channels.
    num_heads
        Number of attention heads.
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
    split_norm
        Whether the norms normalize the vector, scalar, and pseudoscalar streams separately
        instead of with one shared factor.
    checkpoint_blocks
        Whether to use gradient checkpointing for the blocks.
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
        in_p_channels: int,
        out_p_channels: int,
        hidden_p_channels: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
        split_norm: bool = True,
        checkpoint_blocks: bool = False,
        naive_amp: bool = False,
    ) -> None:
        super().__init__()

        assert hidden_p_channels > 0, (
            "LGATrSlimPseudo needs hidden pseudoscalar channels, otherwise use LGATrSlim."
        )
        assert (in_s_channels > 0 and hidden_s_channels > 0) or (
            in_s_channels == 0 and hidden_s_channels == 0 and out_s_channels == 0
        ), "Scalars cannot be used without scalar inputs and hidden channels."
        assert hidden_v_channels >= 4 or in_p_channels > 0, (
            "Pseudoscalars need pseudoscalar inputs or at least 4 hidden vector channels."
        )

        self._in_p_channels = in_p_channels > 0
        self._in_s_channels = in_s_channels > 0
        self._naive_amp = naive_amp
        self._out_s_channels = out_s_channels

        self.linear_in = SlimPseudoLinear(
            in_v_channels=in_v_channels,
            in_s_channels=in_s_channels,
            in_p_channels=in_p_channels,
            out_v_channels=hidden_v_channels,
            out_s_channels=hidden_s_channels,
            out_p_channels=hidden_p_channels,
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
                    split_norm=split_norm,
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
        )
        self._checkpoint_blocks = checkpoint_blocks

        if num_blocks:
            _freeze_dead_tail(
                self.blocks[-1].norm2,
                self.blocks[-1].mlp,
                out_v_channels,
                out_s_channels,
                out_p_channels,
            )

    def forward(
        self,
        vectors: torch.Tensor,
        scalars: torch.Tensor | None = None,
        pseudoscalars: torch.Tensor | None = None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., items, in_v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., items, in_s_channels)``. May be ``None`` only when
            the model expects no input scalar channels.
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
        if self._in_s_channels:
            _require_scalars(scalars=scalars)
        if self._in_p_channels:
            _require_scalars(pseudoscalars=pseudoscalars)
        with naive_amp(self._naive_amp):
            return self._forward(vectors, scalars, pseudoscalars, **attn_kwargs)

    def _forward(
        self,
        vectors: torch.Tensor,
        scalars: torch.Tensor | None,
        pseudoscalars: torch.Tensor | None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if scalars is None:
            scalars = vectors.new_zeros(*vectors.shape[:-2], 0)
        if pseudoscalars is None:
            pseudoscalars = vectors.new_zeros(*vectors.shape[:-2], 0)

        # hidden layers keep vectors channel-last (..., 4, channels) as in LGATrSlim
        # and carry scalars and pseudoscalars merged as [s | p]
        h_v, h_sp = self.linear_in(
            vectors.transpose(-2, -1), torch.cat([scalars, pseudoscalars], dim=-1)
        )

        for block in self.blocks:
            if self._checkpoint_blocks:
                h_v, h_sp = checkpoint(block, h_v, h_sp, use_reentrant=False, **attn_kwargs)
            else:
                h_v, h_sp = block(h_v, h_sp, **attn_kwargs)

        outputs_v, outputs_sp = self.linear_out(h_v, h_sp)
        outputs_s = outputs_sp[..., : self._out_s_channels]
        outputs_p = outputs_sp[..., self._out_s_channels :]
        return outputs_v.transpose(-2, -1), outputs_s, outputs_p
