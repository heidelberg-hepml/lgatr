"""Equivariant conditional transformer for vector and scalar data."""

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..utils.autocast import naive_amp
from ..utils.compile import compile_model
from .slim_layers import ConditionalSlimBlock, SlimLinear, _freeze_dead_tail, _require_scalars


class ConditionalLGATrSlim(nn.Module):
    """Conditional L-GATr-slim network.

    Stacks ``num_blocks`` :class:`ConditionalSlimBlock` modules between initial and final
    :class:`SlimLinear` layers.

    Parameters
    ----------
    in_v_channels
        Number of input vector channels.
    v_channels_cond
        Number of conditional vector channels.
    out_v_channels
        Number of output vector channels.
    hidden_v_channels
        Number of hidden vector channels.
    in_s_channels
        Number of input scalar channels.
    s_channels_cond
        Number of conditional scalar channels.
    out_s_channels
        Number of output scalar channels.
    hidden_s_channels
        Number of hidden scalar channels.
    num_blocks
        Number of Lorentz-transformer blocks.
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
        Number of layers in each MLP.
    dropout_prob
        Dropout probability.
    norm_elementwise_affine
        Whether the :class:`SlimRMSNorm` instances learn per-channel gains.
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
        in_v_channels: int,
        v_channels_cond: int,
        out_v_channels: int,
        hidden_v_channels: int,
        in_s_channels: int,
        s_channels_cond: int,
        out_s_channels: int,
        hidden_s_channels: int,
        num_blocks: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
        checkpoint_blocks: bool = False,
        naive_amp: bool = False,
        compile: bool = False,
        compile_kwargs: Mapping | None = None,
        activation_memory_budget: float | None = None,
    ) -> None:
        super().__init__()
        self._naive_amp = naive_amp

        self.linear_in = SlimLinear(
            in_v_channels=in_v_channels,
            in_s_channels=in_s_channels,
            out_v_channels=hidden_v_channels,
            out_s_channels=hidden_s_channels,
        )

        self.blocks = nn.ModuleList(
            [
                ConditionalSlimBlock(
                    v_channels=hidden_v_channels,
                    s_channels=hidden_s_channels,
                    v_channels_cond=v_channels_cond,
                    s_channels_cond=s_channels_cond,
                    num_heads=num_heads,
                    nonlinearity=nonlinearity,
                    nonlinearity_v=nonlinearity_v,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    num_layers_mlp=num_layers_mlp,
                    dropout_prob=dropout_prob,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_blocks)
            ]
        )

        self.linear_out = SlimLinear(
            in_v_channels=hidden_v_channels,
            in_s_channels=hidden_s_channels,
            out_v_channels=out_v_channels,
            out_s_channels=out_s_channels,
        )
        self._checkpoint_blocks = checkpoint_blocks

        # norm3 is the pre-MLP norm; norm2 (cross-attention) stays alive
        if num_blocks:
            _freeze_dead_tail(
                self.blocks[-1].norm3, self.blocks[-1].mlp, out_v_channels, out_s_channels
            )

        if compile:
            compile_model(
                self,
                compile_kwargs=compile_kwargs,
                activation_memory_budget=activation_memory_budget,
            )

    def forward(
        self,
        vectors: torch.Tensor,
        vectors_cond: torch.Tensor,
        scalars: torch.Tensor,
        scalars_cond: torch.Tensor,
        attn_kwargs: dict | None = None,
        crossattn_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., items, in_v_channels, 4)``.
        vectors_cond
            Condition Lorentz vectors of shape ``(..., items_cond, v_channels_cond, 4)``.
        scalars
            Scalar features of shape ``(..., items, in_s_channels)``.
        scalars_cond
            Condition scalar features of shape ``(..., items_cond, s_channels_cond)``.
        attn_kwargs
            Optional keyword arguments forwarded to self-attention.
        crossattn_kwargs
            Optional keyword arguments forwarded to cross-attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, out_v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., items, out_s_channels)``.
        """
        _require_scalars(scalars=scalars, scalars_cond=scalars_cond)
        with naive_amp(self._naive_amp):
            return self._forward(
                vectors,
                vectors_cond,
                scalars,
                scalars_cond,
                attn_kwargs=attn_kwargs,
                crossattn_kwargs=crossattn_kwargs,
            )

    def _forward(
        self,
        vectors: torch.Tensor,
        vectors_cond: torch.Tensor,
        scalars: torch.Tensor,
        scalars_cond: torch.Tensor,
        attn_kwargs: dict | None = None,
        crossattn_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        crossattn_kwargs = crossattn_kwargs if crossattn_kwargs is not None else {}

        # hidden layers keep vectors channel-last (..., 4, channels) so the vector linears run
        # as flat GEMMs; only the public interface uses (..., channels, 4)
        h_v, h_s = self.linear_in(vectors.transpose(-2, -1), scalars)
        vectors_cond = vectors_cond.transpose(-2, -1)

        for block in self.blocks:
            if self._checkpoint_blocks:
                h_v, h_s = checkpoint(
                    block,
                    vectors=h_v,
                    scalars=h_s,
                    vectors_cond=vectors_cond,
                    scalars_cond=scalars_cond,
                    use_reentrant=False,
                    attn_kwargs=attn_kwargs,
                    crossattn_kwargs=crossattn_kwargs,
                )
            else:
                h_v, h_s = block(
                    vectors=h_v,
                    scalars=h_s,
                    vectors_cond=vectors_cond,
                    scalars_cond=scalars_cond,
                    attn_kwargs=attn_kwargs,
                    crossattn_kwargs=crossattn_kwargs,
                )

        outputs_v, outputs_s = self.linear_out(h_v, h_s)
        return outputs_v.transpose(-2, -1), outputs_s
