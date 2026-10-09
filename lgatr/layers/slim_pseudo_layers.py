"""Building blocks for the slim-pseudo (vector + scalar + pseudoscalar) L-GATr network.

Every layer is its :mod:`~lgatr.layers.slim_layers` counterpart plus a pseudoscalar stream that
is treated like the scalars. The streams only couple through attention and through
:class:`PseudoDeterminant`.
"""

import math

import torch
from torch import nn
from torch.nn.functional import dropout

from ..utils.autocast import minimum_autocast_precision
from ..utils.misc import get_nonlinearity
from .slim_layers import (
    SlimDropout,
    SlimGLU,
    SlimLinear,
    SlimRMSNorm,
    _call_attention,
)


def _require_tensor(**named: torch.Tensor | None) -> None:
    """Raise ValueError if any named tensor is None or has zero channels."""
    for name, tensor in named.items():
        if tensor is None or tensor.shape[-1] == 0:
            raise ValueError(f"{name} must be a non-empty tensor.")

def _post_attention_reshape(
    out: torch.Tensor, hidden_v_channels: int, hidden_s_channels: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_v = out[..., : hidden_v_channels * 4].unflatten(-1, (4, hidden_v_channels))
    h_s = out[..., hidden_v_channels * 4 : hidden_v_channels * 4 + hidden_s_channels]
    h_p = out[..., hidden_v_channels * 4 + hidden_s_channels :]

    h_v = h_v.movedim(-4, -2).flatten(-2, -1)
    h_s = h_s.movedim(-2, -3).flatten(-2, -1)
    h_p = h_p.movedim(-2, -3).flatten(-2, -1)

    return h_v, h_s, h_p


def _freeze_dead_tail(
    norm: nn.Module, mlp: nn.Module, out_v_channels: int, out_s_channels: int, out_p_channels: int
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
    if out_p_channels == 0:
        if norm.weight_p is not None:
            norm.weight_p.requires_grad_(False)
        for name, p in mlp.named_parameters():
            if "linear_p" in name:
                p.requires_grad_(False)


def _det3x3(m: torch.Tensor) -> torch.Tensor:
    """Determinant of a batch of 3x3 matrices ``(..., 3, 3)``."""
    return (
        m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
        - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
        + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0])
    )


def det4x4(m: torch.Tensor) -> torch.Tensor:
    """Determinant of a batch of 4x4 matrices ``(..., 4, 4)`` via cofactor expansion.

    The expansion is a polynomial, so forward and backward stay finite for any finite input.
    """
    det = m.new_zeros(m.shape[:-2])
    for j in range(4):
        minor = torch.cat([m[..., 1:, :j], m[..., 1:, j + 1 :]], dim=-1)
        det = det + ((-1.0) ** j) * m[..., 0, j] * _det3x3(minor)
    return det


class PseudoDeterminant(nn.Module):
    """Map vectors to pseudoscalars through a learned oriented 4-volume.

    The module first projects the input channels to four learned Lorentz vectors for each output
    pseudoscalar channel, then takes the determinant of the resulting 4x4 matrix. The determinant
    of four four-vectors is a parity-odd Lorentz scalar (an oriented 4-volume), so the output flips
    sign under a CP transformation.
    This module requires at least four input vector channels, otherwise the projected vector
    channels are colinear and the determinant is zero.

    Parameters
    ----------
    in_v_channels
        Number of input vector channels.
    out_p_channels
        Number of output pseudoscalar channels.
    """

    def __init__(self, in_v_channels: int, out_p_channels: int) -> None:
        super().__init__()
        assert in_v_channels >= 4, "PseudoDeterminant needs at least 4 input vector channels."
        self._in_v_channels = in_v_channels
        self.weight = nn.Parameter(torch.empty(out_p_channels, 4, in_v_channels))
        self.reset_parameters()

        # zero-size params get grads only sometimes under compile, breaking DDP
        if self.weight.numel() == 0:
            self.weight.requires_grad_(False)

    def reset_parameters(self, factor: float = 1.0) -> None:
        """Re-initialize the projection weights."""
        fan_in = max(self._in_v_channels, 1)
        bound = factor / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, a=-bound, b=bound)

    @minimum_autocast_precision(torch.float32)
    def forward(self, vectors: torch.Tensor, scalars: torch.Tensor, pseudoscalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pseudoscalars from vector inputs.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., 4, in_v_channels)``.

        Returns
        -------
        pseudoscalars
            Pseudoscalar features of shape ``(..., out_p_channels)``.
        """
        projected = torch.einsum("...Mc,pac->...paM", vectors, self.weight)
        pseudoscalars = pseudoscalars + det4x4(projected)
        return vectors, scalars, pseudoscalars


class SlimPseudoDropout(SlimDropout):
    """:class:`~lgatr.layers.slim_layers.SlimDropout` with an additional pseudoscalar stream.

    Parameters
    ----------
    dropout_prob
        Dropout probability.
    """

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, pseudoscalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply dropout.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., 4, v_channels)``.
        scalars
            Scalar features of shape ``(..., s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., p_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors with dropout, same shape as ``vectors``.
        outputs_s
            Scalar features with dropout, same shape as ``scalars``.
        outputs_p
            Pseudoscalar features with dropout, same shape as ``pseudoscalars``.
        """
        if not self.training or self._dropout_prob == 0.0:
            return vectors, scalars, pseudoscalars
        outputs_v, outputs_s = super().forward(vectors, scalars)
        outputs_p = dropout(pseudoscalars, p=self._dropout_prob, training=True)
        return outputs_v, outputs_s, outputs_p


class SlimPseudoRMSNorm(SlimRMSNorm):
    """Joint RMS normalization over vector, scalar, and pseudoscalar features.

    Parameters
    ----------
    v_channels
        Number of vector channels.
    s_channels
        Number of scalar channels.
    p_channels
        Number of pseudoscalar channels.
    epsilon
        Small numerical offset to avoid instabilities.
    elementwise_affine
        Whether to learn a per-channel gain for each stream.
    split_norm
        Whether to normalize the vector, scalar, and pseudoscalar streams with three separate
        factors instead of one shared factor. Empty streams are passed through unchanged.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        p_channels: int,
        epsilon: float = 0.01,
        elementwise_affine: bool = True,
        split_norm: bool = False,
    ) -> None:
        super().__init__(v_channels, s_channels, epsilon, elementwise_affine)
        self.split_norm = split_norm
        if elementwise_affine:
            self.weight_p = nn.Parameter(torch.ones(p_channels))
            # zero-size params get grads only sometimes under compile, breaking DDP
            if self.weight_p.numel() == 0:
                self.weight_p.requires_grad_(False)
        else:
            self.register_parameter("weight_p", None)

    @minimum_autocast_precision(torch.float32, output="high")
    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, pseudoscalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize jointly.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., 4, v_channels)``.
        scalars
            Scalar features of shape ``(..., s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., p_channels)``.

        Returns
        -------
        outputs_v
            Normalized Lorentz vectors, same shape as ``vectors``.
        outputs_s
            Normalized scalar features, same shape as ``scalars``.
        outputs_p
            Normalized pseudoscalar features, same shape as ``pseudoscalars``.
        """
        v_squared_norm = (vectors.square() * self.metric[..., None]).sum(-2).abs()
        s_squared_norm = scalars.square()
        p_squared_norm = pseudoscalars.square()

        if self.split_norm:
            v_norm, s_norm, p_norm = (
                torch.rsqrt(x.sum(-1) / max(x.shape[-1], 1) + self.epsilon)
                for x in (v_squared_norm, s_squared_norm, p_squared_norm)
            )
        else:
            total_features = (
                v_squared_norm.shape[-1] + s_squared_norm.shape[-1] + p_squared_norm.shape[-1]
            )
            sum_squared_norms = (
                v_squared_norm.sum(-1) + s_squared_norm.sum(-1) + p_squared_norm.sum(-1)
            )
            v_norm = s_norm = p_norm = torch.rsqrt(
                sum_squared_norms / total_features + self.epsilon
            )

        outputs_v = vectors * v_norm[..., None, None]
        outputs_s = scalars * s_norm[..., None]
        outputs_p = pseudoscalars * p_norm[..., None]
        if self.elementwise_affine:
            outputs_v = outputs_v * self.weight_v
            outputs_s = outputs_s * self.weight_s
            outputs_p = outputs_p * self.weight_p
        return outputs_v, outputs_s, outputs_p


class SlimPseudoLinear(SlimLinear):
    """:class:`~lgatr.layers.slim_layers.SlimLinear` plus a separate pseudoscalar linear map.

    The three streams are kept separate; mixing happens elsewhere.

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
    in_p_channels
        Number of input pseudoscalar channels.
    out_p_channels
        Number of output pseudoscalar channels.
    bias
        Whether to include a bias term in the scalar linear layer. The pseudoscalar map never has
        one, since a bias would break parity.
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
        in_p_channels: int,
        out_p_channels: int,
        bias: bool = True,
        initialization: str = "default",
    ) -> None:
        super().__init__(
            in_v_channels=in_v_channels,
            out_v_channels=out_v_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            bias=bias,
            initialization=initialization,
        )
        self._in_p_channels = in_p_channels
        self.linear_p = nn.Linear(in_p_channels, out_p_channels, bias=False)
        self._reset_p(initialization)

        # zero-size params get grads only sometimes under compile, breaking DDP
        if self.linear_p.weight.numel() == 0:
            self.linear_p.weight.requires_grad_(False)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, pseudoscalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the linear map.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., 4, in_v_channels)``.
        scalars
            Scalar features of shape ``(..., in_s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., in_p_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., 4, out_v_channels)``.
        outputs_s
            Scalar features of shape ``(..., out_s_channels)``.
        outputs_p
            Pseudoscalar features of shape ``(..., out_p_channels)``.
        """
        outputs_v, outputs_s = super().forward(vectors, scalars)
        outputs_p = self.linear_p(pseudoscalars)
        return outputs_v, outputs_s, outputs_p

    def reset_parameters(self, initialization: str, additional_factor: float = 1.0) -> None:
        """Re-initialize the weights with the given scheme."""
        super().reset_parameters(initialization, additional_factor)
        self._reset_p(initialization, additional_factor)

    def _reset_p(self, initialization: str, additional_factor: float = 1.0) -> None:
        factor = 0.1 * additional_factor if initialization == "small" else additional_factor
        bound = factor / math.sqrt(max(self._in_p_channels, 1))
        nn.init.uniform_(self.linear_p.weight, a=-bound, b=bound)


class SlimPseudoGLU(SlimGLU):
    """:class:`~lgatr.layers.slim_layers.SlimGLU` with an additional pseudoscalar stream.

    The pseudoscalar gates are computed from absolute values of pseudoscalar features.

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
    in_p_channels
        Number of input pseudoscalar channels.
    out_p_channels
        Number of output pseudoscalar channels.
    nonlinearity
        Nonlinearity for the scalar and pseudoscalar gates (and for the vector gate when
        ``nonlinearity_v`` is ``None``). One of ``"relu"``, ``"sigmoid"``, ``"tanh"``, ``"gelu"``,
        ``"silu"``.
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
        in_p_channels: int,
        out_p_channels: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
    ) -> None:
        super().__init__(
            in_v_channels=0, # self.linear replaced by SlimPseudoLinear
            out_v_channels=0,
            in_s_channels=0,
            out_s_channels=0,
            nonlinearity=nonlinearity,
            nonlinearity_v=nonlinearity_v
        )
        self.linear = SlimPseudoLinear(
            in_v_channels=in_v_channels,
            out_v_channels=3 * out_v_channels,
            in_s_channels=in_s_channels,
            out_s_channels=2 * out_s_channels,
            in_p_channels=in_p_channels,
            out_p_channels=2 * out_p_channels,
        )
        # Add bias to |p|, otherwise it is always non-negative and the gate is approx. linear
        self.bias_p = nn.Parameter(torch.zeros(out_p_channels))
        if self.bias_p.numel() == 0:
            self.bias_p.requires_grad_(False)

    def reset_parameters(self, initialization: str) -> None:
        """Re-initialize the weights with the given scheme."""
        self.linear.reset_parameters(initialization)
        nn.init.zeros_(self.bias_p)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, pseudoscalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the GLU.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., 4, in_v_channels)``.
        scalars
            Scalar features of shape ``(..., in_s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., in_p_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., 4, out_v_channels)``.
        outputs_s
            Scalar features of shape ``(..., out_s_channels)``.
        outputs_p
            Pseudoscalar features of shape ``(..., out_p_channels)``.
        """
        v_full, s_full, p_full = self.linear(vectors, scalars, pseudoscalars)
        v_pre, v_gates_1, v_gates_2 = v_full.chunk(3, dim=-1)
        s_pre, s_gates = s_full.chunk(2, dim=-1)
        p_pre, p_gates = p_full.chunk(2, dim=-1)

        v_gates = self._get_inner_product(v_gates_1, v_gates_2)

        outputs_v = self.nonlinearity_v(v_gates) * v_pre
        outputs_s = self.nonlinearity(s_gates) * s_pre
        outputs_p = self.nonlinearity(p_gates.abs() + self.bias_p) * p_pre
        return outputs_v, outputs_s, outputs_p


class SlimPseudoSelfAttention(nn.Module):
    """Self-attention for Lorentz vectors, scalars, and pseudoscalars.

    Pseudoscalars enter queries, keys, and values like scalars: their contribution to the
    attention logits is a product of two pseudoscalars and therefore CP-even.

    Parameters
    ----------
    v_channels
        Number of vector channels.
    s_channels
        Number of scalar channels.
    p_channels
        Number of pseudoscalar channels.
    num_heads
        Number of attention heads.
    attn_ratio
        Expansion ratio for the attention hidden channels.
    dropout_prob
        Dropout probability.
    split_norm
        Whether the QKV-norm normalizes the three streams separately.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        p_channels: int,
        num_heads: int,
        attn_ratio: int = 1,
        dropout_prob: float | None = None,
        split_norm: bool = False,
    ) -> None:
        super().__init__()
        # split min channels to avoid dead streams when one of the input streams is empty
        # set min to 4 or 2+2 like LGATrSlim
        if s_channels == 0:
            min_s_channels = 0
            min_p_channels = 4
        elif p_channels == 0:
            min_s_channels = 4
            min_p_channels = 0
        else:
            min_s_channels = 2
            min_p_channels = 2
        self.hidden_v_channels = max(attn_ratio * v_channels // num_heads, 1)
        self.hidden_s_channels = max(attn_ratio * s_channels // num_heads, min_s_channels)
        self.hidden_p_channels = max(attn_ratio * p_channels // num_heads, min_p_channels)
        self.num_heads = num_heads

        self.metric: torch.Tensor
        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)

        self.linear_in = SlimPseudoLinear(
            in_v_channels=v_channels,
            out_v_channels=3 * self.hidden_v_channels * self.num_heads,
            in_s_channels=s_channels,
            out_s_channels=3 * self.hidden_s_channels * self.num_heads,
            in_p_channels=p_channels,
            out_p_channels=3 * self.hidden_p_channels * self.num_heads,
            bias=False,
            initialization="small",
        )
        self.linear_out = SlimPseudoLinear(
            in_v_channels=self.hidden_v_channels * self.num_heads,
            out_v_channels=v_channels,
            in_s_channels=self.hidden_s_channels * self.num_heads,
            out_s_channels=s_channels,
            in_p_channels=self.hidden_p_channels * self.num_heads,
            out_p_channels=p_channels,
            initialization="small",
        )
        self.norm = SlimPseudoRMSNorm(
            self.hidden_v_channels,
            self.hidden_s_channels,
            self.hidden_p_channels,
            elementwise_affine=False,
            split_norm=split_norm,
        )
        if dropout_prob is not None:
            self.dropout = SlimPseudoDropout(dropout_prob)
        else:
            self.dropout = None

    def _pre_attention_reshape(
        self, qkv_v: torch.Tensor, qkv_s: torch.Tensor, qkv_p: torch.Tensor
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
        qkv_p = (
            qkv_p.unflatten(-1, (3, self.hidden_p_channels, self.num_heads))
            .movedim(-3, 0)
            .movedim(-1, -3)
        )

        # norm QK to avoid attention logit blowup (standard in LLMs)
        # we find that normalizing V as well helps with stability+performance
        qkv_v, qkv_s, qkv_p = self.norm(qkv_v, qkv_s, qkv_p)
        q_v, k_v, v_v = qkv_v.unbind(0)
        q_s, k_s, v_s = qkv_s.unbind(0)
        q_p, k_p, v_p = qkv_p.unbind(0)

        q_v = q_v * self.metric.to(q_v.dtype)[..., None]

        q = torch.cat([q_v.flatten(start_dim=-2), q_s, q_p], dim=-1)
        k = torch.cat([k_v.flatten(start_dim=-2), k_s, k_p], dim=-1)
        v = torch.cat([v_v.flatten(start_dim=-2), v_s, v_p], dim=-1)
        return q, k, v

    def forward(
        self,
        vectors: torch.Tensor,
        scalars: torch.Tensor,
        pseudoscalars: torch.Tensor,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply self-attention.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., items, 4, v_channels)``.
        scalars
            Scalar features of shape ``(..., items, s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., items, p_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, 4, v_channels)``.
        outputs_s
            Scalar features of shape ``(..., items, s_channels)``.
        outputs_p
            Pseudoscalar features of shape ``(..., items, p_channels)``.
        """
        qkv_v, qkv_s, qkv_p = self.linear_in(vectors, scalars, pseudoscalars)

        q, k, v = self._pre_attention_reshape(qkv_v, qkv_s, qkv_p)
        out = _call_attention(q, k, v, **attn_kwargs)
        h_v, h_s, h_p = _post_attention_reshape(out, self.hidden_v_channels, self.hidden_s_channels)

        outputs_v, outputs_s, outputs_p = self.linear_out(h_v, h_s, h_p)

        if self.dropout is not None:
            outputs_v, outputs_s, outputs_p = self.dropout(outputs_v, outputs_s, outputs_p)
        return outputs_v, outputs_s, outputs_p


class SlimPseudoMLP(nn.Module):
    """Multi-layer perceptron for vector, scalar, and pseudoscalar features.

    Parameters
    ----------
    v_channels
        Number of vector channels.
    s_channels
        Number of scalar channels.
    p_channels
        Number of pseudoscalar channels.
    nonlinearity
        Nonlinearity for the GLU layers (scalar/pseudoscalar gates, and vector gate when
        ``nonlinearity_v`` is ``None``).
    nonlinearity_v
        Optional override for the vector-path gate nonlinearity in each GLU.
    mlp_ratio
        Expansion ratio for hidden vector and scalar channels.
    num_layers
        Total number of layers (must be ``>= 2``).
    dropout_prob
        Dropout probability.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        p_channels: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        num_layers: int = 2,
        dropout_prob: float | None = None,
        vectors_det: bool = True,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, f"SlimPseudoMLP needs num_layers >= 2, got {num_layers}"
        layers: list[nn.Module] = []

        v_channels_list = [v_channels] + [mlp_ratio * v_channels] * (num_layers - 1) + [v_channels]
        s_channels_list = [s_channels] + [mlp_ratio * s_channels] * (num_layers - 1) + [s_channels]
        p_channels_list = [p_channels] + [mlp_ratio * p_channels] * (num_layers - 1) + [p_channels]

        if v_channels >= 4 and vectors_det:
            layers.append(PseudoDeterminant(v_channels, p_channels))

        for i in range(num_layers - 1):
            layers.append(
                SlimPseudoGLU(
                    in_v_channels=v_channels_list[i],
                    out_v_channels=v_channels_list[i + 1],
                    in_s_channels=s_channels_list[i],
                    out_s_channels=s_channels_list[i + 1],
                    in_p_channels=p_channels_list[i],
                    out_p_channels=p_channels_list[i + 1],
                    nonlinearity=nonlinearity,
                    nonlinearity_v=nonlinearity_v,
                )
            )
            if dropout_prob is not None:
                layers.append(SlimPseudoDropout(dropout_prob))
        layers.append(
            SlimPseudoLinear(
                in_v_channels=v_channels_list[-2],
                out_v_channels=v_channels_list[-1],
                in_s_channels=s_channels_list[-2],
                out_s_channels=s_channels_list[-1],
                in_p_channels=p_channels_list[-2],
                out_p_channels=p_channels_list[-1],
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, pseudoscalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., 4, v_channels)``.
        scalars
            Scalar features of shape ``(..., s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., p_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., 4, v_channels)``.
        outputs_s
            Scalar features of shape ``(..., s_channels)``.
        outputs_p
            Pseudoscalar features of shape ``(..., p_channels)``.
        """
        h_v, h_s, h_p = vectors, scalars, pseudoscalars

        for layer in self.layers:
            h_v, h_s, h_p = layer(h_v, scalars=h_s, pseudoscalars=h_p)

        return h_v, h_s, h_p


class SlimPseudoBlock(nn.Module):
    """A single block of the pseudoscalar-extended L-GATr-slim network.

    Pre-norm + self-attention + residual, then pre-norm + MLP + residual.

    Parameters
    ----------
    v_channels
        Number of vector channels.
    s_channels
        Number of scalar channels.
    p_channels
        Number of pseudoscalar channels.
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
    split_norm
        Whether the norms normalize the three streams separately.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        p_channels: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
        split_norm: bool = True,
    ) -> None:
        super().__init__()

        assert p_channels > 0, "SlimPseudoBlock requires p_channels > 0, otherwise use SlimBlock."

        norm_kwargs = dict(elementwise_affine=norm_elementwise_affine, split_norm=split_norm)
        self.norm1 = SlimPseudoRMSNorm(v_channels, s_channels, p_channels, **norm_kwargs)
        self.norm2 = SlimPseudoRMSNorm(v_channels, s_channels, p_channels, **norm_kwargs)

        self.attention = SlimPseudoSelfAttention(
            v_channels=v_channels,
            s_channels=s_channels,
            p_channels=p_channels,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
            split_norm=split_norm,
        )

        self.mlp = SlimPseudoMLP(
            v_channels=v_channels,
            s_channels=s_channels,
            p_channels=p_channels,
            nonlinearity=nonlinearity,
            nonlinearity_v=nonlinearity_v,
            mlp_ratio=mlp_ratio,
            num_layers=num_layers_mlp,
            dropout_prob=dropout_prob,
        )

    def forward(
        self,
        vectors: torch.Tensor,
        scalars: torch.Tensor,
        pseudoscalars: torch.Tensor,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., items, 4, v_channels)``.
        scalars
            Scalar features of shape ``(..., items, s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., items, p_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, 4, v_channels)``.
        outputs_s
            Scalar features of shape ``(..., items, s_channels)``.
        outputs_p
            Pseudoscalar features of shape ``(..., items, p_channels)``.
        """
        h_v, h_s, h_p = self.norm1(vectors, scalars, pseudoscalars)

        h_v, h_s, h_p = self.attention(h_v, h_s, h_p, **attn_kwargs)

        outputs_v = vectors + h_v
        outputs_s = scalars + h_s
        outputs_p = pseudoscalars + h_p

        h_v, h_s, h_p = self.norm2(outputs_v, outputs_s, outputs_p)

        h_v, h_s, h_p = self.mlp(h_v, h_s, h_p)

        outputs_v = outputs_v + h_v
        outputs_s = outputs_s + h_s
        outputs_p = outputs_p + h_p

        return outputs_v, outputs_s, outputs_p
