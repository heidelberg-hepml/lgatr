"""Building blocks for the slim-pseudo L-GATr network, built on :mod:`~lgatr.layers.slim_layers`.

Inside the network the scalars and pseudoscalars travel as one tensor
``torch.cat([scalars, pseudoscalars], dim=-1)`` with the split point fixed at construction, so
the layers keep the ``(vectors, scalars)`` interface of the slim layers. The streams are only split
where parity requires it: in the linear maps (no scalar-pseudoscalar mixing, no pseudoscalar bias),
in the GLU gates, in the split norm, and in :class:`PseudoDeterminant`. Dropout, the attention
core, the MLP forward and the block forward are the slim ones.
"""

import math

import torch
from torch import nn
from torch.nn.functional import pad

from ..utils.autocast import minimum_autocast_precision
from . import slim_layers
from .slim_layers import (
    SlimBlock,
    SlimDropout,
    SlimGLU,
    SlimLinear,
    SlimMLP,
    SlimRMSNorm,
    SlimSelfAttention,
    _call_attention,
    _post_attention_reshape,
)


def _freeze_dead_tail(
    norm: nn.Module, mlp: nn.Module, out_v_channels: int, out_s_channels: int, out_p_channels: int
) -> None:
    """Freeze last-block params that cannot receive grads when an output stream is empty."""
    # the norm's weight_s covers scalars and pseudoscalars
    slim_layers._freeze_dead_tail(norm, mlp, out_v_channels, out_s_channels + out_p_channels)
    for name, p in mlp.named_parameters():
        if (out_s_channels == 0 and "linear_s" in name) or (
            out_p_channels == 0 and "linear_p" in name
        ):
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
        assert in_v_channels >= 4 and out_p_channels >= 1, (
            "PseudoDeterminant needs at least 4 vector inputs and 1 pseudoscalar output."
        )
        self._in_v_channels = in_v_channels
        self._out_p_channels = out_p_channels
        self.weight = nn.Parameter(torch.empty(out_p_channels, 4, in_v_channels))
        self.reset_parameters()

    def reset_parameters(self, factor: float = 1.0) -> None:
        """Re-initialize the projection weights."""
        fan_in = max(self._in_v_channels, 1)
        bound = factor / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, a=-bound, b=bound)

    @minimum_autocast_precision(torch.float32)
    def forward(self, vectors: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        scalars[..., -self._out_p_channels :] += det4x4(projected)
        return vectors, scalars


class SlimPseudoRMSNorm(SlimRMSNorm):
    """:class:`~lgatr.layers.slim_layers.SlimRMSNorm` on merged scalars and pseudoscalars.

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
        factors instead of one shared factor.
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
        super().__init__(v_channels, s_channels + p_channels, epsilon, elementwise_affine)
        self._s_channels = s_channels
        self.split_norm = split_norm
        if split_norm and elementwise_affine:
            self.weight_p = nn.Parameter(torch.ones(p_channels))
            if self.weight_p.numel() == 0:
                self.weight_p.requires_grad_(False)

    @minimum_autocast_precision(torch.float32, output="high")
    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize ``vectors`` ``(..., 4, v)`` and merged ``scalars`` ``(..., s + p)``."""
        if not self.split_norm:
            return super().forward(vectors, scalars)

        v_squared_norm = (vectors.square() * self.metric[..., None]).sum(-2).abs()
        s_squared_norm = scalars[..., : self._s_channels].square()
        p_squared_norm = scalars[..., self._s_channels :].square()
        v_norm, s_norm, p_norm = (
            torch.rsqrt(x.sum(-1) / max(x.shape[-1], 1) + self.epsilon)
            for x in (v_squared_norm, s_squared_norm, p_squared_norm)
        )
        if self.elementwise_affine:
            v_norm = v_norm * self.weight_v
            s_norm = s_norm * self.weight_s
            p_norm = p_norm * self.weight_p
        outputs_v = vectors * v_norm[..., None, None]
        outputs_s = torch.cat(
            [
                scalars[..., : self._s_channels] * s_norm[..., None],
                scalars[..., self._s_channels :] * p_norm[..., None],
            ],
            dim=-1,
        )
        return outputs_v, outputs_s


class SlimPseudoLinear(SlimLinear):
    """:class:`~lgatr.layers.slim_layers.SlimLinear` plus a separate pseudoscalar linear map.

    Parameters
    ----------
    in_v_channels, out_v_channels, in_s_channels, out_s_channels, in_p_channels, out_p_channels
        Number of input / output channels per stream.
    bias
        Whether to include a bias term in the scalar linear layer. The pseudoscalar map never has
        one, since a bias would break parity.
    initialization
        ``"default"`` or ``"small"`` (used for attention projections).
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
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``vectors`` ``(..., 4, in_v)`` and merged ``scalars`` ``(..., in_s + in_p)``."""
        outputs_v, outputs_s = super().forward(vectors, scalars[..., : - self._in_p_channels])
        outputs_p = self.linear_p(scalars[..., - self._in_p_channels :])
        return outputs_v, torch.cat([outputs_s, outputs_p], dim=-1)

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
    in_v_channels, out_v_channels, in_s_channels, out_s_channels, in_p_channels, out_p_channels
        Number of input / output channels per stream.
    nonlinearity
        Nonlinearity for the scalar and pseudoscalar gates.
    nonlinearity_v
        Optional override for the vector-path gate nonlinearity.
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
            in_v_channels, out_v_channels, in_s_channels, out_s_channels, nonlinearity=nonlinearity, nonlinearity_v=nonlinearity_v)
        self._in_p_channels = in_p_channels
        self.linear_p = SlimPseudoLinear(
            0, 0, 0, 0,
            in_p_channels=in_p_channels,
            out_p_channels=2 * out_p_channels,
        )
        # Add bias to |p|, otherwise it is always non-negative and the gate is approx. linear
        self.bias_p = nn.Parameter(torch.zeros(out_p_channels))
        if self.bias_p.numel() == 0:
            self.bias_p.requires_grad_(False)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the GLU to ``vectors`` ``(..., 4, in_v)`` and merged ``scalars``."""

        outputs_v, outputs_s = super().forward(vectors, scalars[..., : -self._in_p_channels])
        p_pre, p_gates = self.linear_p(vectors[..., :0], scalars[..., -self._in_p_channels :]).chunk(2, dim=-1)

        outputs_p = self.nonlinearity(p_gates.abs() + self.bias_p) * p_pre
        return outputs_v, torch.cat([outputs_s, outputs_p], dim=-1)


class SlimPseudoSelfAttention(SlimSelfAttention):
    """:class:`~lgatr.layers.slim_layers.SlimSelfAttention` with an additional pseudoscalar stream.

    Pseudoscalars enter queries, keys, and values like scalars: their contribution to the
    attention logits is a product of two pseudoscalars and therefore CP-even.

    Parameters
    ----------
    v_channels, s_channels, p_channels
        Number of channels per stream.
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
        # zero scalars: linear_in, linear_out and norm are replaced below
        super().__init__(v_channels, 0, num_heads, attn_ratio, dropout_prob)
        # split min channels to avoid dead streams when one of the input streams is empty
        # set min to 4 or 2+2 like LGATrSlim
        if s_channels == 0:
            min_s_channels, min_p_channels = 0, 4
        elif p_channels == 0:
            min_s_channels, min_p_channels = 4, 0
        else:
            min_s_channels, min_p_channels = 2, 2
        self._hidden_s = max(attn_ratio * s_channels // num_heads, min_s_channels)
        self._hidden_p = max(attn_ratio * p_channels // num_heads, min_p_channels)
        # the inherited QKV reshape treats the merged scalars as one stream
        self.hidden_s_channels = self._hidden_s + self._hidden_p

        self.linear_in = SlimPseudoLinear(
            in_v_channels=v_channels,
            out_v_channels=3 * self.hidden_v_channels * num_heads,
            in_s_channels=s_channels,
            out_s_channels=3 * self._hidden_s * num_heads,
            in_p_channels=p_channels,
            out_p_channels=3 * self._hidden_p * num_heads,
            bias=False,
            initialization="small",
        )
        self.linear_out = SlimPseudoLinear(
            in_v_channels=self.hidden_v_channels * num_heads,
            out_v_channels=v_channels,
            in_s_channels=self._hidden_s * num_heads,
            out_s_channels=s_channels,
            in_p_channels=self._hidden_p * num_heads,
            out_p_channels=p_channels,
            initialization="small",
        )
        self.norm = SlimPseudoRMSNorm(
            self.hidden_v_channels,
            self._hidden_s,
            self._hidden_p,
            elementwise_affine=False,
            split_norm=split_norm,
        )
        self._s_channels = s_channels

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, **attn_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention to ``vectors`` ``(..., items, 4, v)`` and merged ``scalars``."""
        hs, hp, heads = self._hidden_s, self._hidden_p, self.num_heads
        qkv_v, qkv_sp = self.linear_in(vectors, scalars)

        # [s (3, hs, heads) | p (3, hp, heads)] -> (3, hs + hp, heads) as the slim reshape expects
        split = 3 * hs * heads
        qkv_sp = torch.cat(
            [
                qkv_sp[..., :split].unflatten(-1, (3, hs, heads)),
                qkv_sp[..., split:].unflatten(-1, (3, hp, heads)),
            ],
            dim=-2,
        ).flatten(-3)

        q, k, v = self._pre_attention_reshape(qkv_v, qkv_sp)
        out = _call_attention(q, k, v, **attn_kwargs)
        h_v, h_sp = _post_attention_reshape(out, self.hidden_v_channels)

        # (heads, hs + hp) -> [s (heads, hs) | p (heads, hp)] as linear_out expects
        h_sp = h_sp.unflatten(-1, (heads, hs + hp))
        h_sp = torch.cat([h_sp[..., :hs].flatten(-2), h_sp[..., hs:].flatten(-2)], dim=-1)

        outputs_v, outputs_sp = self.linear_out(h_v, h_sp)

        if self.dropout is not None:
            outputs_v, outputs_sp = self.dropout(outputs_v, outputs_sp)

        return outputs_v, outputs_sp


class SlimPseudoMLP(SlimMLP):
    """:class:`~lgatr.layers.slim_layers.SlimMLP` with an additional pseudoscalar stream and a
    :class:`~lgatr.layers.slim_layers.PseudoDeterminant` at the start.

    Parameters
    ----------
    v_channels, s_channels, p_channels
        Number of channels per stream.
    nonlinearity
        Nonlinearity for the GLU layers.
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
        p_channels: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        num_layers: int = 2,
        dropout_prob: float | None = None,
    ) -> None:
        # zero channels: self.layers is replaced below
        super().__init__(0, 0, num_layers=num_layers)
        layers: list[nn.Module] = []

        def channels(c: int) -> list[int]:
            return [c] + [mlp_ratio * c] * (num_layers - 1) + [c]

        v_list, s_list, p_list = channels(v_channels), channels(s_channels), channels(p_channels)

        if v_channels >= 4:
            layers.append(PseudoDeterminant(v_channels, p_channels))

        for i in range(num_layers - 1):
            layers.append(
                SlimPseudoGLU(
                    v_list[i],
                    v_list[i + 1],
                    s_list[i],
                    s_list[i + 1],
                    p_list[i],
                    p_list[i + 1],
                    nonlinearity=nonlinearity,
                    nonlinearity_v=nonlinearity_v,
                )
            )
            if dropout_prob is not None:
                layers.append(SlimDropout(dropout_prob))
        layers.append(
            SlimPseudoLinear(v_list[-2], v_list[-1], s_list[-2], s_list[-1], p_list[-2], p_list[-1])
        )
        self.layers = nn.ModuleList(layers)


class SlimPseudoBlock(SlimBlock):
    """:class:`~lgatr.layers.slim_layers.SlimBlock` with an additional pseudoscalar stream.

    The forward pass is the slim one; the vector-to-pseudoscalar determinant lives in
    :class:`SlimPseudoSelfAttention`.

    Parameters
    ----------
    v_channels, s_channels, p_channels
        Number of channels per stream.
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
        assert p_channels > 0, "SlimPseudoBlock requires p_channels > 0, otherwise use SlimBlock."
        # zero channels: all submodules are replaced below
        super().__init__(0, 0, num_heads, num_layers_mlp=num_layers_mlp)

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
