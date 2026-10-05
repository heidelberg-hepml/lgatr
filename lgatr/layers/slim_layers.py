"""Building blocks for the slim (vector + scalar) L-GATr networks."""

import math

import torch
from torch import nn
from torch.nn.functional import dropout, dropout1d

from ..primitives.attention import scaled_dot_product_attention
from ..utils.autocast import minimum_autocast_precision
from ..utils.misc import get_nonlinearity


def _require_scalars(**named: torch.Tensor | None) -> None:
    """Raise if any named scalar tensor is None or has zero channels (slim nets require scalars)."""
    for name, tensor in named.items():
        if tensor is None or tensor.shape[-1] == 0:
            raise ValueError(
                f"{name} must be a non-empty scalar tensor; slim networks require scalars."
            )


def _post_attention_reshape(
    out: torch.Tensor, hidden_v_channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    h_v = out[..., : hidden_v_channels * 4].unflatten(-1, (4, hidden_v_channels))
    h_s = out[..., hidden_v_channels * 4 :]

    h_v = h_v.movedim(-4, -2).flatten(-2, -1)
    h_s = h_s.movedim(-2, -3).flatten(-2, -1)
    return h_v, h_s


@minimum_autocast_precision(torch.float32, output="high")
def _call_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)


def _freeze_dead_tail(
    norm: nn.Module, mlp: nn.Module, out_v_channels: int, out_s_channels: int
) -> None:
    """Freeze last-block params that cannot receive grads when an output stream is empty."""
    if out_v_channels == 0:
        if norm.weight_v is not None:
            norm.weight_v.requires_grad_(False)
        for name, p in mlp.named_parameters():
            if name.endswith("weight_v"):
                p.requires_grad_(False)
    if out_s_channels == 0:
        if norm.weight_s is not None:
            norm.weight_s.requires_grad_(False)
        for name, p in mlp.named_parameters():
            if "linear_s" in name:
                p.requires_grad_(False)


class SlimDropout(nn.Module):
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
            Lorentz vectors of shape ``(..., 4, v_channels)``.
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
        flat_v = vectors.transpose(-1, -2).reshape(-1, 4)
        outputs_v = (
            dropout1d(flat_v, p=self._dropout_prob, training=True)
            .reshape(*vectors.shape[:-2], vectors.shape[-1], 4)
            .transpose(-1, -2)
        )
        outputs_s = dropout(scalars, p=self._dropout_prob, training=True)
        return outputs_v, outputs_s


class SlimRMSNorm(nn.Module):
    """Joint RMS normalization over vector and scalar features.

    For vectors the absolute value of the squared norm is used; otherwise the squared norm could
    be negative under the Lorentz metric.

    Parameters
    ----------
    v_channels
        Number of vector channels.
    s_channels
        Number of scalar channels.
    epsilon
        Small numerical offset to avoid instabilities.
    elementwise_affine
        Whether to learn a per-channel gain for the vector and scalar streams.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        epsilon: float = 0.01,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.elementwise_affine = elementwise_affine
        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)
        if elementwise_affine:
            self.weight_v = nn.Parameter(torch.ones(v_channels))
            self.weight_s = nn.Parameter(torch.ones(s_channels))
            # zero-size params get grads only sometimes under compile, breaking DDP
            for weight in (self.weight_v, self.weight_s):
                if weight.numel() == 0:
                    weight.requires_grad_(False)
        else:
            self.register_parameter("weight_v", None)
            self.register_parameter("weight_s", None)

    @minimum_autocast_precision(torch.float32, output="high")
    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize jointly.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., 4, v_channels)``.
        scalars
            Scalar features of shape ``(..., s_channels)``.

        Returns
        -------
        outputs_v
            Normalized Lorentz vectors, same shape as ``vectors``.
        outputs_s
            Normalized scalar features, same shape as ``scalars``.
        """
        v_squared_norm = (vectors.square() * self.metric[..., None]).sum(-2).abs()
        s_squared_norm = scalars.square()
        total_features = v_squared_norm.shape[-1] + s_squared_norm.shape[-1]
        mean_squared_norms = (v_squared_norm.sum(-1) + s_squared_norm.sum(-1)) / total_features
        norm = torch.rsqrt(mean_squared_norms + self.epsilon)

        outputs_v = vectors * norm[..., None, None]
        outputs_s = scalars * norm[..., None]
        if self.elementwise_affine:
            outputs_v = outputs_v * self.weight_v
            outputs_s = outputs_s * self.weight_s
        return outputs_v, outputs_s


class SlimLinear(nn.Module):
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

        # zero-size params get grads only sometimes under compile, breaking DDP
        if self.weight_v.numel() == 0:
            self.weight_v.requires_grad_(False)

    @minimum_autocast_precision(torch.float32, output="high")
    def _linear_v(self, vectors: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(vectors, self.weight_v)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the linear map.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., 4, in_v_channels)``.
        scalars
            Scalar features of shape ``(..., in_s_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., 4, out_v_channels)``.
        outputs_s
            Scalar features of shape ``(..., out_s_channels)``.
        """
        outputs_v = self._linear_v(vectors)
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
            if self.linear_s.bias is not None:
                nn.init.zeros_(self.linear_s.bias)


class SlimGLU(nn.Module):
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
        Nonlinearity for the scalar gate (and for the vector gate when ``nonlinearity_v`` is
        ``None``). One of ``"relu"``, ``"sigmoid"``, ``"tanh"``, ``"gelu"``, ``"silu"``.
    nonlinearity_v
        Optional override for the vector-path gate nonlinearity. ``None`` falls back to
        ``nonlinearity``.
    """

    def __init__(
        self,
        in_v_channels: int,
        out_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
    ) -> None:
        super().__init__()
        self.linear = SlimLinear(
            in_v_channels=in_v_channels,
            out_v_channels=3 * out_v_channels,
            in_s_channels=in_s_channels,
            out_s_channels=2 * out_s_channels,
        )
        self.nonlinearity = get_nonlinearity(nonlinearity)
        self.nonlinearity_v = (
            get_nonlinearity(nonlinearity_v) if nonlinearity_v is not None else self.nonlinearity
        )
        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the GLU.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., 4, in_v_channels)``.
        scalars
            Scalar features of shape ``(..., in_s_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., 4, out_v_channels)``.
        outputs_s
            Scalar features of shape ``(..., out_s_channels)``.
        """
        v_full, s_full = self.linear(vectors, scalars)
        v_pre, v_gates_1, v_gates_2 = v_full.chunk(3, dim=-1)
        s_pre, s_gates = s_full.chunk(2, dim=-1)

        v_gates = self._get_inner_product(v_gates_1, v_gates_2)

        outputs_v = self.nonlinearity_v(v_gates) * v_pre
        outputs_s = self.nonlinearity(s_gates) * s_pre
        return outputs_v, outputs_s

    @minimum_autocast_precision(torch.float32)
    def _get_inner_product(self, v_gates_1: torch.Tensor, v_gates_2: torch.Tensor) -> torch.Tensor:
        # 0.5 = 1/sqrt(4) controls the scale, like 1/sqrt(d_k) in attention
        return 0.5 * ((v_gates_1 * v_gates_2) * self.metric[..., None]).sum(dim=-2, keepdim=True)


class SlimSelfAttention(nn.Module):
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

        self.linear_in = SlimLinear(
            in_v_channels=v_channels,
            out_v_channels=3 * self.hidden_v_channels * self.num_heads,
            in_s_channels=s_channels,
            out_s_channels=3 * self.hidden_s_channels * self.num_heads,
            bias=False,
            initialization="small",
        )
        self.linear_out = SlimLinear(
            in_v_channels=self.hidden_v_channels * self.num_heads,
            out_v_channels=v_channels,
            in_s_channels=self.hidden_s_channels * self.num_heads,
            out_s_channels=s_channels,
            initialization="small",
        )
        self.norm = SlimRMSNorm(
            self.hidden_v_channels,
            self.hidden_s_channels,
            elementwise_affine=False,
        )
        if dropout_prob is not None:
            self.dropout = SlimDropout(dropout_prob)
        else:
            self.dropout = None

    def _pre_attention_reshape(
        self, qkv_v: torch.Tensor, qkv_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_v = (
            qkv_v.unflatten(-1, (3, self.hidden_v_channels, self.num_heads))
            .movedim(-3, 0)
            .movedim(-1, -4)
        )
        qkv_s = (
            qkv_s.unflatten(-1, (3, self.hidden_s_channels, self.num_heads))
            .movedim(-3, 0)
            .movedim(-1, -3)
        )

        # norm QK to avoid attention logit blowup (standard in LLMs)
        # we find that normalizing V as well helps with stability+performance
        qkv_v, qkv_s = self.norm(qkv_v, qkv_s)
        q_v, k_v, v_v = qkv_v.unbind(0)
        q_s, k_s, v_s = qkv_s.unbind(0)

        q_v = q_v * self.metric.to(q_v.dtype)[..., None]

        q = torch.cat([q_v.flatten(start_dim=-2), q_s], dim=-1)
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
            Lorentz vectors of shape ``(..., items, 4, v_channels)``.
        scalars
            Scalar features of shape ``(..., items, s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, 4, v_channels)``.
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


class SlimMLP(nn.Module):
    """Multi-layer perceptron for vector and scalar features.

    Parameters
    ----------
    v_channels
        Number of vector channels.
    s_channels
        Number of scalar channels.
    nonlinearity
        Nonlinearity for the GLU layers (scalar gate, and vector gate when
        ``nonlinearity_v`` is ``None``).
    nonlinearity_v
        Optional override for the vector-path gate nonlinearity in each GLU.
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
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        num_layers: int = 2,
        dropout_prob: float | None = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, f"SlimMLP needs num_layers >= 2, got {num_layers}"
        layers: list[nn.Module] = []

        v_channels_list = [v_channels] + [mlp_ratio * v_channels] * (num_layers - 1) + [v_channels]
        s_channels_list = [s_channels] + [mlp_ratio * s_channels] * (num_layers - 1) + [s_channels]

        for i in range(num_layers - 1):
            layers.append(
                SlimGLU(
                    in_v_channels=v_channels_list[i],
                    out_v_channels=v_channels_list[i + 1],
                    in_s_channels=s_channels_list[i],
                    out_s_channels=s_channels_list[i + 1],
                    nonlinearity=nonlinearity,
                    nonlinearity_v=nonlinearity_v,
                )
            )
            if dropout_prob is not None:
                layers.append(SlimDropout(dropout_prob))
        layers.append(
            SlimLinear(
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
            Lorentz vectors of shape ``(..., 4, v_channels)``.
        scalars
            Scalar features of shape ``(..., s_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., 4, v_channels)``.
        outputs_s
            Scalar features of shape ``(..., s_channels)``.
        """
        h_v, h_s = vectors, scalars

        for layer in self.layers:
            h_v, h_s = layer(h_v, scalars=h_s)

        return h_v, h_s


class SlimBlock(nn.Module):
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
    nonlinearity_v
        Optional override for the vector-path gate nonlinearity in the MLP's GLUs.
    mlp_ratio
        Expansion ratio for MLP hidden channels.
    attn_ratio
        Expansion ratio for attention hidden channels.
    num_layers_mlp
        Number of layers in the MLP (must be ``>= 2``).
    dropout_prob
        Dropout probability.
    norm_elementwise_affine
        Whether the RMS norms learn a per-channel gain.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        self.norm1 = SlimRMSNorm(v_channels, s_channels, elementwise_affine=norm_elementwise_affine)
        self.norm2 = SlimRMSNorm(v_channels, s_channels, elementwise_affine=norm_elementwise_affine)

        self.attention = SlimSelfAttention(
            v_channels=v_channels,
            s_channels=s_channels,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
        )

        self.mlp = SlimMLP(
            v_channels=v_channels,
            s_channels=s_channels,
            nonlinearity=nonlinearity,
            nonlinearity_v=nonlinearity_v,
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
            Lorentz vectors of shape ``(..., items, 4, v_channels)``.
        scalars
            Scalar features of shape ``(..., items, s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, 4, v_channels)``.
        outputs_s
            Scalar features of shape ``(..., items, s_channels)``.
        """
        h_v, h_s = self.norm1(vectors, scalars)

        h_v, h_s = self.attention(
            h_v,
            h_s,
            **attn_kwargs,
        )

        outputs_v = vectors + h_v
        outputs_s = scalars + h_s

        h_v, h_s = self.norm2(outputs_v, outputs_s)

        h_v, h_s = self.mlp(h_v, h_s)

        outputs_v = outputs_v + h_v
        outputs_s = outputs_s + h_s

        return outputs_v, outputs_s


class SlimCrossAttention(nn.Module):
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
    ) -> None:
        super().__init__()
        self.hidden_v_channels = max(attn_ratio * q_v_channels // num_heads, 1)
        self.hidden_s_channels = max(attn_ratio * q_s_channels // num_heads, 4)
        self.num_heads = num_heads

        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)

        self.linear_in_q = SlimLinear(
            in_v_channels=q_v_channels,
            out_v_channels=self.hidden_v_channels * self.num_heads,
            in_s_channels=q_s_channels,
            out_s_channels=self.hidden_s_channels * self.num_heads,
            bias=False,
            initialization="small",
        )
        self.linear_in_kv = SlimLinear(
            in_v_channels=kv_v_channels,
            out_v_channels=2 * self.hidden_v_channels * self.num_heads,
            in_s_channels=kv_s_channels,
            out_s_channels=2 * self.hidden_s_channels * self.num_heads,
            bias=False,
            initialization="small",
        )
        self.linear_out = SlimLinear(
            in_v_channels=self.hidden_v_channels * self.num_heads,
            out_v_channels=q_v_channels,
            in_s_channels=self.hidden_s_channels * self.num_heads,
            out_s_channels=q_s_channels,
            initialization="small",
        )

        self.norm = SlimRMSNorm(
            self.hidden_v_channels,
            self.hidden_s_channels,
            elementwise_affine=False,
        )
        if dropout_prob is not None:
            self.dropout = SlimDropout(dropout_prob)
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
            kv_v.unflatten(-1, (2, self.hidden_v_channels, self.num_heads))
            .movedim(-3, 0)
            .movedim(-1, -4)
        )  # (2, *B, H, N, 4, Cv)
        kv_s = (
            kv_s.unflatten(-1, (2, self.hidden_s_channels, self.num_heads))
            .movedim(-3, 0)
            .movedim(-1, -3)
        )  # (2, *B, H, N, Cs)
        q_v = q_v.unflatten(-1, (self.hidden_v_channels, self.num_heads)).movedim(
            -1, -4
        )  # (*B, H, Nc, 4, Cv)
        q_s = q_s.unflatten(-1, (self.hidden_s_channels, self.num_heads)).movedim(
            -1, -3
        )  # (*B, H, Nc, Cs)

        q_v, q_s = self.norm(q_v, q_s)
        kv_v, kv_s = self.norm(kv_v, kv_s)
        k_v, v_v = kv_v.unbind(0)
        k_s, v_s = kv_s.unbind(0)

        q_v = q_v * self.metric.to(q_v.dtype)[..., None]

        q = torch.cat([q_v.flatten(start_dim=-2), q_s], dim=-1)
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
            Query Lorentz vectors of shape ``(..., items_q, 4, q_v_channels)``.
        vectors_kv
            Key/value Lorentz vectors of shape ``(..., items_kv, 4, kv_v_channels)``.
        scalars_q
            Query scalar features of shape ``(..., items_q, q_s_channels)``.
        scalars_kv
            Key/value scalar features of shape ``(..., items_kv, kv_s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items_q, 4, q_v_channels)``.
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


class ConditionalSlimBlock(nn.Module):
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
    nonlinearity_v
        Optional override for the vector-path gate nonlinearity in every GLU. ``None`` falls
        back to ``nonlinearity``.
    mlp_ratio
        Expansion ratio for MLP hidden channels.
    attn_ratio
        Expansion ratio for attention hidden channels.
    num_layers_mlp
        Number of layers in the MLP (must be ``>= 2``).
    dropout_prob
        Dropout probability.
    norm_elementwise_affine
        Whether the :class:`SlimRMSNorm` instances learn per-channel gains.
    """

    def __init__(
        self,
        v_channels: int,
        v_channels_cond: int,
        s_channels: int,
        s_channels_cond: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        self.norm1 = SlimRMSNorm(v_channels, s_channels, elementwise_affine=norm_elementwise_affine)
        self.norm2 = SlimRMSNorm(v_channels, s_channels, elementwise_affine=norm_elementwise_affine)
        self.norm3 = SlimRMSNorm(v_channels, s_channels, elementwise_affine=norm_elementwise_affine)

        self.selfattention = SlimSelfAttention(
            v_channels=v_channels,
            s_channels=s_channels,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
        )
        self.crossattention = SlimCrossAttention(
            q_v_channels=v_channels,
            kv_v_channels=v_channels_cond,
            q_s_channels=s_channels,
            kv_s_channels=s_channels_cond,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
        )

        self.mlp = SlimMLP(
            v_channels=v_channels,
            s_channels=s_channels,
            nonlinearity=nonlinearity,
            nonlinearity_v=nonlinearity_v,
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
            Lorentz vectors of shape ``(..., items, 4, v_channels)``.
        vectors_cond
            Condition Lorentz vectors of shape ``(..., items_cond, 4, v_channels_cond)``.
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
            Lorentz vectors of shape ``(..., items, 4, v_channels)``.
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
