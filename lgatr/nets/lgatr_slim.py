"""Equivariant transformer for vector and scalar data."""

import math

import torch
from torch import nn
from torch.nn.functional import dropout, dropout1d
from torch.utils.checkpoint import checkpoint

from ..primitives.attention import scaled_dot_product_attention
from ..utils.autocast import minimum_autocast_precision
from ..utils.compile import compile_model
from ..utils.misc import get_nonlinearity


def _post_attention_reshape(
    out: torch.Tensor, hidden_v_channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    h_v = out[..., : hidden_v_channels * 4].unflatten(-1, (hidden_v_channels, 4))
    h_s = out[..., hidden_v_channels * 4 :]

    h_v = h_v.movedim(-3, -4).flatten(-3, -2)
    h_s = h_s.movedim(-2, -3).flatten(-2, -1)
    return h_v, h_s


@minimum_autocast_precision(torch.float32)
def _call_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)


class Dropout(nn.Module):
    """Dropout for vector and scalar features.

    For vector features the same dropout mask is applied to all four components of each vector.

    Parameters
    ----------
    dropout_prob
        Dropout probability.
    """

    def __init__(self, dropout_prob: float) -> None:
        super().__init__()
        self._dropout_prob = dropout_prob

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply dropout.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., s_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors with dropout, same shape as ``vectors``.
        outputs_s
            Scalar features with dropout, same shape as ``scalars``.
        """
        if not self.training or self._dropout_prob == 0.0:
            return vectors, scalars

        # have to reshape vectors because dropout1d constrains input shape
        flat_v = vectors.reshape(-1, 4)
        outputs_v = dropout1d(flat_v, p=self._dropout_prob, training=True).reshape(vectors.shape)
        outputs_s = dropout(scalars, p=self._dropout_prob, training=True)
        return outputs_v, outputs_s


class RMSNorm(nn.Module):
    """Joint RMS normalization over vector and scalar features.

    For vectors the absolute value of the squared norm is used; otherwise the squared norm could
    be negative under the Lorentz metric.

    Parameters
    ----------
    epsilon
        Small numerical offset to avoid instabilities.
    """

    def __init__(self, epsilon: float = 0.01) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)

    @minimum_autocast_precision(torch.float32)
    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize jointly.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., s_channels)``.

        Returns
        -------
        outputs_v
            Normalized Lorentz vectors, same shape as ``vectors``.
        outputs_s
            Normalized scalar features, same shape as ``scalars``.
        """
        v_squared_norm = (vectors.square() * self.metric).sum(-1).abs()
        s_squared_norm = scalars.square()
        total_features = v_squared_norm.shape[-1] + s_squared_norm.shape[-1]
        mean_squared_norms = (v_squared_norm.sum(-1) + s_squared_norm.sum(-1)) / total_features
        norm = torch.rsqrt(mean_squared_norms + self.epsilon)

        outputs_v = vectors * norm[..., None, None]
        outputs_s = scalars * norm[..., None]
        return outputs_v, outputs_s


class Linear(nn.Module):
    """Linear layer for vector and scalar features.

    The vector and scalar streams are kept separate; mixing happens elsewhere.

    Parameters
    ----------
    in_v_channels
        Number of input vector channels.
    out_v_channels
        Number of output vector channels.
    in_s_channels
        Number of input scalar channels.
    out_s_channels
        Number of output scalar channels.
    bias
        Whether to include a bias term in the scalar linear layer.
    initialization
        Initialization scheme for the weights. ``"default"`` or ``"small"`` (smaller weights, used
        for attention projections to improve stability).
    """

    def __init__(
        self,
        in_v_channels: int,
        out_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        bias: bool = True,
        initialization: str = "default",
    ) -> None:
        super().__init__()
        self._in_v_channels = in_v_channels
        self._out_v_channels = out_v_channels
        self._in_s_channels = in_s_channels
        self._out_s_channels = out_s_channels
        self._bias = bias

        self.weight_v = nn.Parameter(
            torch.empty(
                (
                    out_v_channels,
                    in_v_channels,
                )
            )
        )
        self.linear_s: nn.Linear | None
        if in_s_channels and out_s_channels:
            self.linear_s = nn.Linear(in_s_channels, out_s_channels, bias=bias)
        else:
            self.linear_s = None

        self.reset_parameters(initialization)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the linear map.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., in_v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., in_s_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., out_v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., out_s_channels)``.
        """
        outputs_v = self.weight_v @ vectors
        if self.linear_s is not None:
            outputs_s = self.linear_s(scalars)
        else:
            outputs_s = scalars.new_zeros(*scalars.shape[:-1], self._out_s_channels)
        return outputs_v, outputs_s

    def reset_parameters(self, initialization: str, additional_factor: float = 1.0) -> None:
        """Re-initialize the weights with the given scheme."""
        if initialization == "default":
            v_factor = additional_factor
            s_factor = additional_factor
        elif initialization == "small":
            v_factor = 0.1 * additional_factor
            s_factor = 0.1 * additional_factor
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        if self.weight_v.numel() > 0:
            fan_in = max(self._in_v_channels, 1)
            bound = v_factor / math.sqrt(fan_in)
            nn.init.uniform_(self.weight_v, a=-bound, b=bound)

        if self.linear_s is not None:
            fan_in = max(self._in_s_channels, 1)
            bound = s_factor / math.sqrt(fan_in)
            nn.init.uniform_(self.linear_s.weight, a=-bound, b=bound)


class GatedLinearUnit(nn.Module):
    """Gated linear unit (GLU) for vector and scalar features.

    Scalar gates are computed from scalar features; vector gates are computed from inner products
    of (transformed) vector features.

    Parameters
    ----------
    in_v_channels
        Number of input vector channels.
    out_v_channels
        Number of output vector channels.
    in_s_channels
        Number of input scalar channels.
    out_s_channels
        Number of output scalar channels.
    nonlinearity
        Nonlinearity for the gate. One of ``"relu"``, ``"sigmoid"``, ``"tanh"``, ``"gelu"``,
        ``"silu"``.
    """

    def __init__(
        self,
        in_v_channels: int,
        out_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        nonlinearity: str = "gelu",
    ) -> None:
        super().__init__()
        self.linear = Linear(
            in_v_channels=in_v_channels,
            out_v_channels=3 * out_v_channels,
            in_s_channels=in_s_channels,
            out_s_channels=2 * out_s_channels,
        )
        self.nonlinearity = get_nonlinearity(nonlinearity)
        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the GLU.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., in_v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., in_s_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., out_v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., out_s_channels)``.
        """
        v_full, s_full = self.linear(vectors, scalars)
        v_pre, v_gates_1, v_gates_2 = v_full.chunk(3, dim=-2)
        s_pre, s_gates = s_full.chunk(2, dim=-1)

        v_gates = self._get_inner_product(v_gates_1, v_gates_2)

        outputs_v = self.nonlinearity(v_gates) * v_pre
        outputs_s = self.nonlinearity(s_gates) * s_pre
        return outputs_v, outputs_s

    @minimum_autocast_precision(torch.float32)
    def _get_inner_product(self, v_gates_1: torch.Tensor, v_gates_2: torch.Tensor) -> torch.Tensor:
        return ((v_gates_1 * v_gates_2) * self.metric).sum(dim=-1, keepdim=True)


class SelfAttention(nn.Module):
    """Self-attention for Lorentz vectors and scalar features.

    Parameters
    ----------
    v_channels
        Number of vector channels.
    s_channels
        Number of scalar channels.
    num_heads
        Number of attention heads.
    attn_ratio
        Expansion ratio for the attention hidden channels.
    dropout_prob
        Dropout probability.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        num_heads: int,
        attn_ratio: int = 1,
        dropout_prob: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_v_channels = max(attn_ratio * v_channels // num_heads, 1)
        self.hidden_s_channels = max(attn_ratio * s_channels // num_heads, 4)
        self.num_heads = num_heads

        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)

        self.linear_in = Linear(
            in_v_channels=v_channels,
            out_v_channels=3 * self.hidden_v_channels * self.num_heads,
            in_s_channels=s_channels,
            out_s_channels=3 * self.hidden_s_channels * self.num_heads,
            initialization="small",
        )
        self.linear_out = Linear(
            in_v_channels=self.hidden_v_channels * self.num_heads,
            out_v_channels=v_channels,
            in_s_channels=self.hidden_s_channels * self.num_heads,
            out_s_channels=s_channels,
            initialization="small",
        )
        self.norm = RMSNorm()
        if dropout_prob is not None:
            self.dropout = Dropout(dropout_prob)
        else:
            self.dropout = None

    def _pre_attention_reshape(
        self, qkv_v: torch.Tensor, qkv_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_v = (
            qkv_v.unflatten(-2, (3, self.hidden_v_channels, self.num_heads))
            .movedim(-4, 0)
            .movedim(-2, -4)
        )
        qkv_s = (
            qkv_s.unflatten(-1, (3, self.hidden_s_channels, self.num_heads))
            .movedim(-3, 0)
            .movedim(-1, -3)
        )

        # normalize for stability (important)
        qkv_v, qkv_s = self.norm(qkv_v, qkv_s)

        q_v, k_v, v_v = qkv_v.unbind(0)
        q_s, k_s, v_s = qkv_s.unbind(0)

        q_v_mod = q_v * self.metric.to(q_v.dtype)
        q = torch.cat([q_v_mod.flatten(start_dim=-2), q_s], dim=-1)
        k = torch.cat([k_v.flatten(start_dim=-2), k_s], dim=-1)
        v = torch.cat([v_v.flatten(start_dim=-2), v_s], dim=-1)
        return q, k, v

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, **attn_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., items, v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., items, s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., items, s_channels)``.
        """
        qkv_v, qkv_s = self.linear_in(vectors, scalars)

        q, k, v = self._pre_attention_reshape(qkv_v, qkv_s)
        out = _call_attention(q, k, v, **attn_kwargs)
        h_v, h_s = _post_attention_reshape(out, self.hidden_v_channels)

        outputs_v, outputs_s = self.linear_out(h_v, h_s)

        if self.dropout is not None:
            outputs_v, outputs_s = self.dropout(outputs_v, outputs_s)
        return outputs_v, outputs_s


class MLP(nn.Module):
    """Multi-layer perceptron for vector and scalar features.

    Parameters
    ----------
    v_channels
        Number of vector channels.
    s_channels
        Number of scalar channels.
    nonlinearity
        Nonlinearity for the GLU layers.
    mlp_ratio
        Expansion ratio for hidden channels.
    num_layers
        Total number of layers (must be ``>= 2``).
    dropout_prob
        Dropout probability.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        nonlinearity: str = "gelu",
        mlp_ratio: int = 2,
        num_layers: int = 2,
        dropout_prob: float | None = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 2
        layers: list[nn.Module] = []

        v_channels_list = [v_channels] + [mlp_ratio * v_channels] * (num_layers - 1) + [v_channels]
        s_channels_list = [s_channels] + [mlp_ratio * s_channels] * (num_layers - 1) + [s_channels]

        for i in range(num_layers - 1):
            layers.append(
                GatedLinearUnit(
                    in_v_channels=v_channels_list[i],
                    out_v_channels=v_channels_list[i + 1],
                    in_s_channels=s_channels_list[i],
                    out_s_channels=s_channels_list[i + 1],
                    nonlinearity=nonlinearity,
                )
            )
            if dropout_prob is not None:
                layers.append(Dropout(dropout_prob))
        layers.append(
            Linear(
                in_v_channels=v_channels_list[-2],
                out_v_channels=v_channels_list[-1],
                in_s_channels=s_channels_list[-2],
                out_s_channels=s_channels_list[-1],
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., s_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., s_channels)``.
        """
        h_v, h_s = vectors, scalars

        for layer in self.layers:
            h_v, h_s = layer(h_v, scalars=h_s)

        return h_v, h_s


class LGATrSlimBlock(nn.Module):
    """A single block of the L-GATr-slim network.

    Pre-norm + self-attention + residual, then pre-norm + MLP + residual.

    Parameters
    ----------
    v_channels
        Number of vector channels.
    s_channels
        Number of scalar channels.
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
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
    ) -> None:
        super().__init__()

        self.norm = RMSNorm()

        self.attention = SelfAttention(
            v_channels=v_channels,
            s_channels=s_channels,
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
        self, vectors: torch.Tensor, scalars: torch.Tensor, **attn_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., items, v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., items, s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., items, s_channels)``.
        """
        h_v, h_s = self.norm(vectors, scalars)

        h_v, h_s = self.attention(
            h_v,
            h_s,
            **attn_kwargs,
        )

        outputs_v = vectors + h_v
        outputs_s = scalars + h_s

        h_v, h_s = self.norm(outputs_v, outputs_s)

        h_v, h_s = self.mlp(h_v, h_s)

        outputs_v = outputs_v + h_v
        outputs_s = outputs_s + h_s

        return outputs_v, outputs_s


class LGATrSlim(nn.Module):
    """L-GATr-slim network.

    A slimmer L-GATr variant that operates on Lorentz vectors and scalars (no full multivector
    representation). Stacks ``num_blocks`` :class:`LGATrSlimBlock` modules between initial and
    final :class:`Linear` layers.

    Parameters
    ----------
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
        out_v_channels: int,
        hidden_v_channels: int,
        in_s_channels: int,
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
                LGATrSlimBlock(
                    v_channels=hidden_v_channels,
                    s_channels=hidden_s_channels,
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

        if compile:
            compile_model(self, **compile_kwargs)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, **attn_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., items, in_v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., items, in_s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, out_v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., items, out_s_channels)``.
        """
        h_v, h_s = self.linear_in(vectors, scalars)

        for block in self.blocks:
            if self._checkpoint_blocks:
                h_v, h_s = checkpoint(block, h_v, h_s, use_reentrant=False, **attn_kwargs)
            else:
                h_v, h_s = block(h_v, h_s, **attn_kwargs)

        outputs_v, outputs_s = self.linear_out(h_v, h_s)
        return outputs_v, outputs_s
