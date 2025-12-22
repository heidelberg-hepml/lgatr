"""Equivariant conditional transformer for vector and scalar data."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .lgatr_slim import (
    MLP,
    Dropout,
    Linear,
    RMSNorm,
    SelfAttention,
    scaled_dot_product_attention_careful,
)


class CrossAttention(nn.Module):
    """Cross-attention module for Lorentz vectors and scalar features."""

    def __init__(
        self,
        q_v_channels: int,
        kv_v_channels: int,
        q_s_channels: int,
        kv_s_channels: int,
        num_heads: int,
        attn_ratio: int = 1,
        dropout_prob: float | None = None,
    ):
        super().__init__()
        self.hidden_v_channels = max(attn_ratio * q_v_channels // num_heads, 1)
        self.hidden_s_channels = max(attn_ratio * q_s_channels // num_heads, 4)
        self.num_heads = num_heads

        metric = torch.tensor([1.0, -1.0, -1.0, -1.0])
        self.register_buffer("metric", metric)

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

        self.norm = RMSNorm()
        if dropout_prob is not None:
            self.dropout = Dropout(dropout_prob)
        else:
            self.dropout = None

    def _pre_reshape(self, q_v, kv_v, q_s, kv_s):
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
        q_v, q_s = self.norm(q_v, q_s)
        kv_v, kv_s = self.norm(kv_v, kv_s)

        k_v, v_v = kv_v.unbind(0)
        k_s, v_s = kv_s.unbind(0)

        q_v_mod = q_v * self.metric
        q = torch.cat([q_v_mod.flatten(start_dim=-2), q_s], dim=-1)
        k = torch.cat([k_v.flatten(start_dim=-2), k_s], dim=-1)
        v = torch.cat([v_v.flatten(start_dim=-2), v_s], dim=-1)
        return q, k, v

    def _post_reshape(self, out):
        h_v = out[..., : self.hidden_v_channels * 4].reshape(
            *out.shape[:-1], self.hidden_v_channels, 4
        )
        h_s = out[..., self.hidden_v_channels * 4 :]

        h_v = h_v.movedim(-3, -4).flatten(-3, -2)
        h_s = h_s.movedim(-2, -3).flatten(-2, -1)
        return h_v, h_s

    def forward(self, vectors, vectors_condition, scalars, scalars_condition, **attn_kwargs):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        **attn_kwargs : dict
            Additional keyword arguments for the attention function.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the normalized vectors and scalars.
        """
        q_v, q_s = self.linear_in_q(vectors, scalars)
        kv_v, kv_s = self.linear_in_kv(vectors_condition, scalars_condition)

        q, k, v = self._pre_reshape(q_v, kv_v, q_s, kv_s)
        out = scaled_dot_product_attention_careful(q, k, v, **attn_kwargs)
        h_v, h_s = self._post_reshape(out)

        out_v, out_s = self.linear_out(h_v, h_s)

        if self.dropout is not None:
            out_v, out_s = self.dropout(out_v, out_s)
        return out_v, out_s


class ConditionalLGATrSlimBlock(nn.Module):
    """A single block of the conditional L-GATr-slim,
    consisting of self-attention, cross-attention and MLP layers, pre-norm and residual connections."""

    def __init__(
        self,
        v_channels: int,
        condition_v_channels: int,
        s_channels: int,
        condition_s_channels: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
    ):
        super().__init__()

        self.norm = RMSNorm()

        self.selfattention = SelfAttention(
            v_channels=v_channels,
            s_channels=s_channels,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
        )
        self.crossattention = CrossAttention(
            q_v_channels=v_channels,
            kv_v_channels=condition_v_channels,
            q_s_channels=s_channels,
            kv_s_channels=condition_s_channels,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
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
        vectors,
        vectors_condition,
        scalars,
        scalars_condition,
        attn_kwargs=None,
        crossattn_kwargs=None,
    ):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        vectors_condition : torch.Tensor
            A tensor of shape (..., condition_v_channels, 4) representing a Lorentz vector condition included in cross-attention.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        scalars_condition : torch.Tensor
            A tensor of shape (..., condition_s_channels) representing a scalar condition included in cross-attention.
        **attn_kwargs : dict
            Additional keyword arguments for the attention function.
        **crossattn_kwargs : dict
            Additional keyword arguments for the cross-attention function.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the normalized vectors and scalars.
        """
        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        crossattn_kwargs = crossattn_kwargs if crossattn_kwargs is not None else {}

        # self-attention block
        h_v, h_s = self.norm(vectors, scalars)
        h_v, h_s = self.selfattention(
            h_v,
            h_s,
            **attn_kwargs,
        )
        outputs_v = vectors + h_v
        outputs_s = scalars + h_s

        # cross-attention block
        h_v, h_s = self.norm(outputs_v, outputs_s)
        h_v, h_s = self.crossattention(
            h_v,
            vectors_condition,
            h_s,
            scalars_condition,
            **crossattn_kwargs,
        )
        outputs_v = outputs_v + h_v
        outputs_s = outputs_s + h_s

        # MLP block
        h_v, h_s = self.norm(outputs_v, outputs_s)
        h_v, h_s = self.mlp(h_v, h_s)
        outputs_v = outputs_v + h_v
        outputs_s = outputs_s + h_s

        return outputs_v, outputs_s


class ConditionalLGATrSlim(nn.Module):
    """Conditional L-GATr-slim network."""

    def __init__(
        self,
        in_v_channels: int,
        condition_v_channels: int,
        out_v_channels: int,
        hidden_v_channels: int,
        in_s_channels: int,
        condition_s_channels: int,
        out_s_channels: int,
        hidden_s_channels: int,
        num_blocks: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        checkpoint_blocks: bool = False,
        compile: bool = False,
    ):
        """
        Parameters
        ----------
        in_v_channels : int
            Number of input vector channels.
        condition_v_channels : int
            Number of conditional vector channels.
        out_v_channels : int
            Number of output vector channels.
        hidden_v_channels : int
            Number of hidden vector channels.
        in_s_channels : int
            Number of input scalar channels.
        condition_s_channels : int
            Number of conditional scalar channels.
        out_s_channels : int
            Number of output scalar channels.
        hidden_s_channels : int
            Number of hidden scalar channels.
        num_blocks : int
            Number of Lorentz Transformer blocks.
        num_heads : int
            Number of attention heads.
        nonlinearity : str, optional
            Nonlinearity type for MLP layers, by default "gelu".
        mlp_ratio : int, optional
            Expansion ratio for MLP hidden layers, by default 2.
        attn_ratio : int, optional
            Expansion ratio for attention hidden layers, by default 1.
        num_layers_mlp : int, optional
            Number of layers in MLP, by default 2.
        dropout_prob : float | None, optional
            Dropout probability, by default None.
        checkpoint_blocks : bool, optional
            Whether to use gradient checkpointing for blocks, by default False.
        compile : bool, optional
            Whether to compile the model with torch.compile, by default False.
        """
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
                    condition_v_channels=condition_v_channels,
                    condition_s_channels=condition_s_channels,
                    num_heads=num_heads,
                    nonlinearity=nonlinearity,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    num_layers_mlp=num_layers_mlp,
                    dropout_prob=dropout_prob,
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

        self.compile = compile
        if compile:
            # ugly hack to make torch.compile convenient for users
            # the clean solution is model = torch.compile(model, **kwargs) outside of the constructor
            # note that we need fullgraph=False because of the torch.compiler.disable for attention
            self.__class__ = torch.compile(self.__class__, dynamic=True, mode="default")

    def forward(
        self,
        vectors,
        vectors_condition,
        scalars,
        scalars_condition,
        attn_kwargs=None,
        crossattn_kwargs=None,
    ):
        """
        Parameters
        ----------
        vectors : torch.Tensor
            A tensor of shape (..., v_channels, 4) representing Lorentz vectors.
        vectors_condition : torch.Tensor
            A tensor of shape (..., v_channels_condition, 4) representing a Lorentz vector condition included in cross-attention.
        scalars : torch.Tensor
            A tensor of shape (..., s_channels) representing scalar features.
        scalars_condition : torch.Tensor
            A tensor of shape (..., s_channels_condition) representing a scalar condition included in cross-attention.
        attn_kwargs : dict
            Additional keyword arguments for the self-attention function.
        crossattn_kwargs : dict
            Additional keyword arguments for the cross-attention function.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Tensors of the same shape as input representing the normalized vectors and scalars.
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
                    vectors_condition=vectors_condition,
                    scalars_condition=scalars_condition,
                    use_reentrant=False,
                    **attn_kwargs,
                )
            else:
                h_v, h_s = block(
                    vectors=h_v,
                    scalars=h_s,
                    vectors_condition=vectors_condition,
                    scalars_condition=scalars_condition,
                    **attn_kwargs,
                )

        outputs_v, outputs_s = self.linear_out(h_v, h_s)
        return outputs_v, outputs_s
