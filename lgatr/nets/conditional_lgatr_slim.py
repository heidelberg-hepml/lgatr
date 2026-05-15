"""Equivariant conditional transformer for vector and scalar data."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..utils.compile import compile_model
from .lgatr_slim import (
    MLP,
    Dropout,
    Linear,
    RMSNorm,
    SelfAttention,
    _call_attention,
    _post_attention_reshape,
)


class CrossAttention(nn.Module):
    """Cross-attention for Lorentz vectors and scalar features.

    Parameters
    ----------
    q_v_channels
        Number of query vector channels.
    kv_v_channels
        Number of key/value vector channels.
    q_s_channels
        Number of query scalar channels.
    kv_s_channels
        Number of key/value scalar channels.
    num_heads
        Number of attention heads.
    attn_ratio
        Expansion ratio for the attention hidden channels.
    dropout_prob
        Dropout probability.
    """

    def __init__(
        self,
        q_v_channels: int,
        kv_v_channels: int,
        q_s_channels: int,
        kv_s_channels: int,
        num_heads: int,
        attn_ratio: int = 1,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_v_channels = max(attn_ratio * q_v_channels // num_heads, 1)
        self.hidden_s_channels = max(attn_ratio * q_s_channels // num_heads, 4)
        self.num_heads = num_heads

        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)

        self.linear_in_q = Linear(
            in_v_channels=q_v_channels,
            out_v_channels=self.hidden_v_channels * self.num_heads,
            in_s_channels=q_s_channels,
            out_s_channels=self.hidden_s_channels * self.num_heads,
            initialization="small",
        )
        self.linear_in_kv = Linear(
            in_v_channels=kv_v_channels,
            out_v_channels=2 * self.hidden_v_channels * self.num_heads,
            in_s_channels=kv_s_channels,
            out_s_channels=2 * self.hidden_s_channels * self.num_heads,
            initialization="small",
        )
        self.linear_out = Linear(
            in_v_channels=self.hidden_v_channels * self.num_heads,
            out_v_channels=q_v_channels,
            in_s_channels=self.hidden_s_channels * self.num_heads,
            out_s_channels=q_s_channels,
            initialization="small",
        )

        self.norm_q = RMSNorm(
            self.hidden_v_channels,
            self.hidden_s_channels,
            elementwise_affine=norm_elementwise_affine,
        )
        self.norm_kv = RMSNorm(
            self.hidden_v_channels,
            self.hidden_s_channels,
            elementwise_affine=norm_elementwise_affine,
        )
        if dropout_prob is not None:
            self.dropout = Dropout(dropout_prob)
        else:
            self.dropout = None

    def _pre_attention_reshape(
        self,
        q_v: torch.Tensor,
        kv_v: torch.Tensor,
        q_s: torch.Tensor,
        kv_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kv_v = (
            kv_v.unflatten(-2, (2, self.hidden_v_channels, self.num_heads))
            .movedim(-4, 0)
            .movedim(-2, -4)
        )  # (2, *B, H, N, Cv, 4)
        kv_s = (
            kv_s.unflatten(-1, (2, self.hidden_s_channels, self.num_heads))
            .movedim(-3, 0)
            .movedim(-1, -3)
        )  # (2, *B, H, N, Cs)
        q_v = q_v.unflatten(-2, (self.hidden_v_channels, self.num_heads)).movedim(
            -2, -4
        )  # (*B, H, Nc, Cv, 4)
        q_s = q_s.unflatten(-1, (self.hidden_s_channels, self.num_heads)).movedim(
            -1, -3
        )  # (*B, H, Nc, Cs)

        # normalize for stability (important)
        q_v, q_s = self.norm_q(q_v, q_s)
        kv_v, kv_s = self.norm_kv(kv_v, kv_s)

        k_v, v_v = kv_v.unbind(0)
        k_s, v_s = kv_s.unbind(0)

        q_v_mod = q_v * self.metric.to(q_v.dtype)
        q = torch.cat([q_v_mod.flatten(start_dim=-2), q_s], dim=-1)
        k = torch.cat([k_v.flatten(start_dim=-2), k_s], dim=-1)
        v = torch.cat([v_v.flatten(start_dim=-2), v_s], dim=-1)
        return q, k, v

    def forward(
        self,
        vectors_q: torch.Tensor,
        vectors_kv: torch.Tensor,
        scalars_q: torch.Tensor,
        scalars_kv: torch.Tensor,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-attention.

        Parameters
        ----------
        vectors_q
            Query Lorentz vectors of shape ``(..., items_q, q_v_channels, 4)``.
        vectors_kv
            Key/value Lorentz vectors of shape ``(..., items_kv, kv_v_channels, 4)``.
        scalars_q
            Query scalar features of shape ``(..., items_q, q_s_channels)``.
        scalars_kv
            Key/value scalar features of shape ``(..., items_kv, kv_s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items_q, q_v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., items_q, q_s_channels)``.
        """
        q_v, q_s = self.linear_in_q(vectors_q, scalars_q)
        kv_v, kv_s = self.linear_in_kv(vectors_kv, scalars_kv)

        q, k, v = self._pre_attention_reshape(q_v, kv_v, q_s, kv_s)
        out = _call_attention(q, k, v, **attn_kwargs)
        h_v, h_s = _post_attention_reshape(out, self.hidden_v_channels)

        outputs_v, outputs_s = self.linear_out(h_v, h_s)

        if self.dropout is not None:
            outputs_v, outputs_s = self.dropout(outputs_v, outputs_s)
        return outputs_v, outputs_s


class ConditionalLGATrSlimBlock(nn.Module):
    """A single block of the conditional L-GATr-slim network.

    Pre-norm + self-attention + residual, then pre-norm + cross-attention + residual, then
    pre-norm + MLP + residual.

    Parameters
    ----------
    v_channels
        Number of vector channels.
    v_channels_cond
        Number of condition vector channels.
    s_channels
        Number of scalar channels.
    s_channels_cond
        Number of condition scalar channels.
    num_heads
        Number of attention heads.
    nonlinearity
        Nonlinearity for the MLP layers.
    mlp_ratio
        Expansion ratio for MLP hidden channels.
    attn_ratio
        Expansion ratio for attention hidden channels.
    num_layers_mlp
        Number of layers in the MLP.
    dropout_prob
        Dropout probability.
    norm_elementwise_affine
        Whether the :class:`RMSNorm` instances learn per-channel gains.
    """

    def __init__(
        self,
        v_channels: int,
        v_channels_cond: int,
        s_channels: int,
        s_channels_cond: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = False,
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(v_channels, s_channels, elementwise_affine=norm_elementwise_affine)
        self.norm2 = RMSNorm(v_channels, s_channels, elementwise_affine=norm_elementwise_affine)
        self.norm3 = RMSNorm(v_channels, s_channels, elementwise_affine=norm_elementwise_affine)

        self.selfattention = SelfAttention(
            v_channels=v_channels,
            s_channels=s_channels,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
            norm_elementwise_affine=norm_elementwise_affine,
        )
        self.crossattention = CrossAttention(
            q_v_channels=v_channels,
            kv_v_channels=v_channels_cond,
            q_s_channels=s_channels,
            kv_s_channels=s_channels_cond,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
            norm_elementwise_affine=norm_elementwise_affine,
        )

        self.mlp = MLP(
            v_channels=v_channels,
            s_channels=s_channels,
            nonlinearity=nonlinearity,
            mlp_ratio=mlp_ratio,
            num_layers=num_layers_mlp,
            dropout_prob=dropout_prob,
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
            Lorentz vectors of shape ``(..., items, v_channels, 4)``.
        vectors_cond
            Condition Lorentz vectors of shape ``(..., items_cond, v_channels_cond, 4)``.
        scalars
            Scalar features of shape ``(..., items, s_channels)``.
        scalars_cond
            Condition scalar features of shape ``(..., items_cond, s_channels_cond)``.
        attn_kwargs
            Optional keyword arguments forwarded to self-attention.
        crossattn_kwargs
            Optional keyword arguments forwarded to cross-attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., items, s_channels)``.
        """
        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        crossattn_kwargs = crossattn_kwargs if crossattn_kwargs is not None else {}

        # self-attention block
        h_v, h_s = self.norm1(vectors, scalars)
        h_v, h_s = self.selfattention(
            h_v,
            h_s,
            **attn_kwargs,
        )
        outputs_v = vectors + h_v
        outputs_s = scalars + h_s

        # cross-attention block
        h_v, h_s = self.norm2(outputs_v, outputs_s)
        h_v, h_s = self.crossattention(
            h_v,
            vectors_cond,
            h_s,
            scalars_cond,
            **crossattn_kwargs,
        )
        outputs_v = outputs_v + h_v
        outputs_s = outputs_s + h_s

        # MLP block
        h_v, h_s = self.norm3(outputs_v, outputs_s)
        h_v, h_s = self.mlp(h_v, h_s)
        outputs_v = outputs_v + h_v
        outputs_s = outputs_s + h_s

        return outputs_v, outputs_s


class ConditionalLGATrSlim(nn.Module):
    """Conditional L-GATr-slim network.

    Stacks ``num_blocks`` :class:`ConditionalLGATrSlimBlock` modules between initial and final
    :class:`Linear` layers.

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
    mlp_ratio
        Expansion ratio for MLP hidden channels.
    attn_ratio
        Expansion ratio for attention hidden channels.
    num_layers_mlp
        Number of layers in each MLP.
    dropout_prob
        Dropout probability.
    norm_elementwise_affine
        Whether the :class:`RMSNorm` instances learn per-channel gains.
    checkpoint_blocks
        Whether to use gradient checkpointing for the blocks.
    compile
        Whether to wrap the model with :func:`torch.compile`.
    **compile_kwargs
        Forwarded to :func:`lgatr.utils.compile.compile_model` when ``compile=True``;
        see there for the supported keys (``compile_mode``, ``compile_dynamic``,
        ``compile_fullgraph``) and their defaults.
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
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = False,
        checkpoint_blocks: bool = False,
        compile: bool = False,
        **compile_kwargs,
    ) -> None:
        super().__init__()

        self.linear_in = Linear(
            in_v_channels=in_v_channels,
            in_s_channels=in_s_channels,
            out_v_channels=hidden_v_channels,
            out_s_channels=hidden_s_channels,
        )

        self.blocks = nn.ModuleList(
            [
                ConditionalLGATrSlimBlock(
                    v_channels=hidden_v_channels,
                    s_channels=hidden_s_channels,
                    v_channels_cond=v_channels_cond,
                    s_channels_cond=s_channels_cond,
                    num_heads=num_heads,
                    nonlinearity=nonlinearity,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    num_layers_mlp=num_layers_mlp,
                    dropout_prob=dropout_prob,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_blocks)
            ]
        )

        self.linear_out = Linear(
            in_v_channels=hidden_v_channels,
            in_s_channels=hidden_s_channels,
            out_v_channels=out_v_channels,
            out_s_channels=out_s_channels,
        )
        self._checkpoint_blocks = checkpoint_blocks

        if compile:
            compile_model(self, **compile_kwargs)

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
        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        crossattn_kwargs = crossattn_kwargs if crossattn_kwargs is not None else {}

        h_v, h_s = self.linear_in(vectors, scalars)

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
        return outputs_v, outputs_s
