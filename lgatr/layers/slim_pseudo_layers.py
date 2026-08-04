"""Building blocks for the slim-pseudo (vector + scalar + pseudoscalar) L-GATr network."""

import math

import torch
from torch import nn
from torch.nn.functional import dropout, dropout1d

from ..utils.autocast import minimum_autocast_precision
from ..utils.misc import get_nonlinearity
from .slim_layers import _call_attention


def inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lorentz inner product over the last (four-vector) dimension, signature ``(+, -, -, -)``."""
    time = x[..., 0] * y[..., 0]
    space = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    return time - space


def squared_norm(x: torch.Tensor) -> torch.Tensor:
    """Lorentz squared norm over the last (four-vector) dimension."""
    return inner_product(x, x)


def _det3x3(m: torch.Tensor) -> torch.Tensor:
    """Determinant of a batch of 3x3 matrices ``(..., 3, 3)`` via the rule of Sarrus."""
    return (
        m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
        - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
        + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0])
    )


def det4x4(m: torch.Tensor) -> torch.Tensor:
    """Determinant of a batch of 4x4 matrices ``(..., 4, 4)`` via cofactor expansion.

    Unlike :func:`torch.linalg.det`, whose backward is ``det(A) * inv(A).mT`` and therefore
    diverges to ``inf``/``nan`` when ``A`` is singular (e.g. linearly dependent four-vectors), this
    explicit polynomial expansion has a smooth, bounded gradient everywhere -- including at exactly
    singular configurations -- which is essential for training stability.
    """
    det = m.new_zeros(m.shape[:-2])
    for j in range(4):
        minor = torch.cat([m[..., 1:, :j], m[..., 1:, j + 1 :]], dim=-1)
        det = det + ((-1.0) ** j) * m[..., 0, j] * _det3x3(minor)
    return det


def _post_attention_reshape(
    out: torch.Tensor, hidden_v_channels: int, hidden_s_channels: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the concatenated attention output into vector, scalar, and pseudoscalar streams."""
    v_end = hidden_v_channels * 4
    s_end = v_end + hidden_s_channels
    h_v = out[..., :v_end].unflatten(-1, (hidden_v_channels, 4))
    h_s = out[..., v_end:s_end]
    h_p = out[..., s_end:]

    h_v = h_v.movedim(-3, -4).flatten(-3, -2)
    h_s = h_s.movedim(-2, -3).flatten(-2, -1)
    h_p = h_p.movedim(-2, -3).flatten(-2, -1)
    return h_v, h_s, h_p


class VectorToPseudoscalar(nn.Module):
    """Map vectors to pseudoscalars through a learned oriented 4-volume.

    The module first projects the input channels to four learned Lorentz vectors for each output
    pseudoscalar channel, then takes the determinant of the resulting 4x4 matrix. The determinant
    of four four-vectors is a parity-odd Lorentz scalar (an oriented 4-volume), so the output flips
    sign under spatial inversion.

    Parameters
    ----------
    in_v_channels
        Number of input vector channels.
    out_p_channels
        Number of output pseudoscalar channels.
    """

    def __init__(self, in_v_channels: int, out_p_channels: int) -> None:
        super().__init__()
        self._in_v_channels = in_v_channels
        self._out_p_channels = out_p_channels
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
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute pseudoscalars from vector inputs.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., in_v_channels, 4)``.

        Returns
        -------
        pseudoscalars
            Pseudoscalar features of shape ``(..., out_p_channels)``.
        """
        projected = torch.einsum("...cM,pac->...paM", vectors, self.weight)
        # det4x4 (explicit cofactor expansion) instead of torch.linalg.det: the latter's backward
        # is det(A) * inv(A).mT, which blows up to nan when the four projected vectors are linearly
        # dependent (a common occurrence that silently poisons the weights, since AMP is off).
        return det4x4(projected)


class VectorToTripleProduct(nn.Module):
    """Map vectors to pseudoscalars through a triple product against a fixed reference.

    Projects the input channels to *three* learned Lorentz vectors per output channel and contracts
    them with a learnable reference four-vector through the 4D Levi-Civita symbol -- equivalently the
    determinant of ``[ref, a, b, c]``. Like :class:`VectorToPseudoscalar` the result is parity-odd,
    but it is only *rank-3* in the data (one determinant row is the fixed reference), so it is far
    less dominated by the product-of-magnitudes tail that makes the full four-vector determinant a
    high-variance / low-SNR observable. It is the four-volume analogue of a spatial triple product
    ``n_a . (n_b x n_c)`` -- the CP-odd observable that actually carries the ttH(->gamma gamma)
    signal.

    The reference is a learnable *parameter* four-vector rather than a data vector: this deliberately
    singles out a preferred frame (initialized to the time/lab direction), exactly as the beam/time
    spurions injected in the data embedding already do. With a covariant (data-derived) reference the
    contraction would collapse back to a plain four-vector determinant and add nothing over
    :class:`VectorToPseudoscalar`; breaking the reference covariance is what makes this a genuinely
    new, lower-rank CP-odd primitive.

    Parameters
    ----------
    in_v_channels
        Number of input vector channels.
    out_p_channels
        Number of output pseudoscalar channels.
    """

    def __init__(self, in_v_channels: int, out_p_channels: int) -> None:
        super().__init__()
        self._in_v_channels = in_v_channels
        self._out_p_channels = out_p_channels
        self.weight = nn.Parameter(torch.empty(out_p_channels, 3, in_v_channels))
        self.reference = nn.Parameter(torch.empty(out_p_channels, 4))
        self.reset_parameters()

        # zero-size params get grads only sometimes under compile, breaking DDP
        if self.weight.numel() == 0:
            self.weight.requires_grad_(False)
            self.reference.requires_grad_(False)

    def reset_parameters(self, factor: float = 1.0) -> None:
        """Re-initialize the projection weights and the reference vector."""
        fan_in = max(self._in_v_channels, 1)
        bound = factor / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, a=-bound, b=bound)
        # reference initialized near the time direction (the natural lab-frame reference), with a
        # small random tilt so the out_p channels are not degenerate.
        with torch.no_grad():
            self.reference.zero_()
            if self.reference.numel() > 0:
                self.reference[:, 0] = 1.0
                self.reference.add_(torch.randn_like(self.reference) * 0.1)

    @minimum_autocast_precision(torch.float32)
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute pseudoscalars from vector inputs.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., in_v_channels, 4)``.

        Returns
        -------
        pseudoscalars
            Pseudoscalar features of shape ``(..., out_p_channels)``.
        """
        projected = torch.einsum("...cM,pac->...paM", vectors, self.weight)  # (..., p, 3, 4)
        ref = self.reference[..., None, :].expand(
            *projected.shape[:-3], -1, -1, -1
        )  # (..., p, 1, 4)
        stacked = torch.cat([ref, projected], dim=-2)  # (..., p, 4, 4)
        return det4x4(stacked)


class SlimPseudoDropout(nn.Module):
    """SlimPseudoDropout for vector, scalar, and pseudoscalar features.

    For vector features the same dropout mask is applied to all four components of each vector.

    Parameters
    ----------
    dropout_prob
        SlimPseudoDropout probability.
    """

    def __init__(self, dropout_prob: float) -> None:
        super().__init__()
        self._dropout_prob = dropout_prob

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, pseudoscalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply dropout.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., v_channels, 4)``.
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

        # have to reshape vectors because dropout1d constrains input shape
        flat_v = vectors.reshape(-1, 4)
        outputs_v = dropout1d(flat_v, p=self._dropout_prob, training=True).reshape(vectors.shape)
        outputs_s = dropout(scalars, p=self._dropout_prob, training=True)
        outputs_p = dropout(pseudoscalars, p=self._dropout_prob, training=True)
        return outputs_v, outputs_s, outputs_p


class SlimPseudoRMSNorm(nn.Module):
    """Joint RMS normalization over vector, scalar, and pseudoscalar features.

    For vectors the absolute value of the squared norm is used; otherwise the squared norm could
    be negative under the Lorentz metric. With ``elementwise_affine`` a learnable per-channel gain
    is applied after normalization, which requires the channel counts to be known at construction.

    Parameters
    ----------
    v_channels
        Number of vector channels. Required for the learnable gain; ``None`` disables affine.
    s_channels
        Number of scalar channels. Required for the learnable gain; ``None`` disables affine.
    p_channels
        Number of pseudoscalar channels. Required for the learnable gain; ``None`` disables affine.
    epsilon
        Small numerical offset to avoid instabilities.
    elementwise_affine
        Whether to apply a learnable per-channel gain. Silently disabled when the channel counts
        are not provided (e.g. ``SlimPseudoRMSNorm()``).
    """

    def __init__(
        self,
        v_channels: int | None = None,
        s_channels: int | None = None,
        p_channels: int | None = None,
        epsilon: float = 0.01,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.elementwise_affine = elementwise_affine and None not in (
            v_channels,
            s_channels,
            p_channels,
        )
        if self.elementwise_affine:
            self.weight_v = nn.Parameter(torch.ones(v_channels))
            self.weight_s = nn.Parameter(torch.ones(s_channels))
            self.weight_p = nn.Parameter(torch.ones(p_channels))
            # zero-size params get grads only sometimes under compile, breaking DDP
            for weight in (self.weight_v, self.weight_s, self.weight_p):
                if weight.numel() == 0:
                    weight.requires_grad_(False)
        else:
            self.register_parameter("weight_v", None)
            self.register_parameter("weight_s", None)
            self.register_parameter("weight_p", None)

    @minimum_autocast_precision(torch.float32)
    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, pseudoscalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize jointly.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., v_channels, 4)``.
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
        v_squared_norm = squared_norm(vectors).abs()
        s_squared_norm = scalars.square()
        p_squared_norm = pseudoscalars.square()
        total_features = vectors.shape[-2] + scalars.shape[-1] + pseudoscalars.shape[-1]
        sum_squared_norms = v_squared_norm.sum(-1) + s_squared_norm.sum(-1) + p_squared_norm.sum(-1)
        norm = torch.rsqrt(sum_squared_norms / total_features + self.epsilon)

        outputs_v = vectors * norm[..., None, None]
        outputs_s = scalars * norm[..., None]
        outputs_p = pseudoscalars * norm[..., None]
        if self.elementwise_affine:
            outputs_v = outputs_v * self.weight_v[..., None]
            outputs_s = outputs_s * self.weight_s
            outputs_p = outputs_p * self.weight_p
        return outputs_v, outputs_s, outputs_p


class SlimPseudoLinear(nn.Module):
    """Linear layer for vector, scalar, and pseudoscalar features.

    The vector and scalar streams are kept separate; pseudoscalars couple back into the other
    streams in the only parity-consistent ways: a parity-odd vector-to-pseudoscalar contribution
    feeds the pseudoscalar output (via :class:`VectorToPseudoscalar`), and a parity-even
    contribution from squared pseudoscalars feeds the scalar output.

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
        Whether to include a bias term in the scalar linear layer.
    initialization
        Initialization scheme for the weights. ``"default"`` or ``"small"`` (smaller weights, used
        for attention projections to improve stability).
    cp_triple_product
        If ``True``, add a :class:`VectorToTripleProduct` contribution to the pseudoscalar output --
        a lower-rank, lower-variance CP-odd primitive (lab-frame triple product) alongside the full
        four-vector determinant of :class:`VectorToPseudoscalar`. Defaults to ``False`` (no extra
        parameters), so the default model is byte-for-byte unchanged.
    cp_scalar_pseudo_mixing
        If ``True``, modulate the pseudoscalar linear path by a parity-even gate derived from the
        scalar features (``even x odd = odd``), letting event context shape the CP-odd observable.
        The gate is zero-initialized so it starts as the identity; defaults to ``False``.
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
        cp_triple_product: bool = False,
        cp_scalar_pseudo_mixing: bool = False,
    ) -> None:
        super().__init__()
        self._in_v_channels = in_v_channels
        self._out_v_channels = out_v_channels
        self._in_s_channels = in_s_channels
        self._out_s_channels = out_s_channels
        self._in_p_channels = in_p_channels
        self._out_p_channels = out_p_channels
        self._bias = bias
        self._cp_triple_product = cp_triple_product
        self._cp_scalar_pseudo_mixing = cp_scalar_pseudo_mixing

        self.weight_v = nn.Parameter(torch.empty((out_v_channels, in_v_channels)))
        self.linear_s = nn.Linear(in_s_channels, out_s_channels, bias=bias)
        self.p_to_s = nn.Linear(in_p_channels, out_s_channels, bias=False)
        self.linear_p = nn.Linear(in_p_channels, out_p_channels, bias=False)
        self.vector_to_p = VectorToPseudoscalar(in_v_channels, out_p_channels)
        # (1) lower-rank CP-odd primitive: spurion/lab-referenced triple product
        self.vector_to_p_triple = (
            VectorToTripleProduct(in_v_channels, out_p_channels) if cp_triple_product else None
        )
        # (2) scalar-context modulation of the CP-odd (pseudoscalar) path
        self.s_to_p_gate = (
            nn.Linear(in_s_channels, out_p_channels, bias=True) if cp_scalar_pseudo_mixing else None
        )

        self.reset_parameters(initialization)

        # zero-size params get grads only sometimes under compile, breaking DDP
        if self.weight_v.numel() == 0:
            self.weight_v.requires_grad_(False)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, pseudoscalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the linear map.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., in_v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., in_s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., in_p_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., out_v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., out_s_channels)``.
        outputs_p
            Pseudoscalar features of shape ``(..., out_p_channels)``.
        """
        outputs_v = nn.functional.linear(vectors.mT, self.weight_v).mT
        outputs_s = self.linear_s(scalars) + self.p_to_s(pseudoscalars.square())

        p_lin = self.linear_p(pseudoscalars)
        if self.s_to_p_gate is not None:
            # even x odd = odd; gate zero-initialized so this starts as the identity
            p_lin = p_lin * (1.0 + self.s_to_p_gate(scalars))
        outputs_p = p_lin + self.vector_to_p(vectors)
        if self.vector_to_p_triple is not None:
            outputs_p = outputs_p + self.vector_to_p_triple(vectors)
        return outputs_v, outputs_s, outputs_p

    def reset_parameters(self, initialization: str, additional_factor: float = 1.0) -> None:
        """Re-initialize the weights with the given scheme."""
        if initialization == "default":
            v_factor = additional_factor
            s_factor = additional_factor
            p_factor = additional_factor
        elif initialization == "small":
            v_factor = 0.1 * additional_factor
            s_factor = 0.1 * additional_factor
            p_factor = 0.1 * additional_factor
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        if self.weight_v.numel() > 0:
            fan_in = max(self._in_v_channels, 1)
            bound = v_factor / math.sqrt(fan_in)
            nn.init.uniform_(self.weight_v, a=-bound, b=bound)

        fan_in = max(self._in_s_channels, 1)
        bound = s_factor / math.sqrt(fan_in)
        nn.init.uniform_(self.linear_s.weight, a=-bound, b=bound)
        if self.linear_s.bias is not None:
            nn.init.zeros_(self.linear_s.bias)

        fan_in = max(self._in_p_channels, 1)
        bound = p_factor / math.sqrt(fan_in)
        nn.init.uniform_(self.linear_p.weight, a=-bound, b=bound)
        nn.init.uniform_(self.p_to_s.weight, a=-bound, b=bound)
        self.vector_to_p.reset_parameters(p_factor)
        if self.vector_to_p_triple is not None:
            self.vector_to_p_triple.reset_parameters(p_factor)
        if self.s_to_p_gate is not None:
            # start as the identity modulation: gate(s) = 0 -> factor (1 + 0) = 1
            nn.init.zeros_(self.s_to_p_gate.weight)
            nn.init.zeros_(self.s_to_p_gate.bias)


class SlimPseudoGLU(nn.Module):
    """Gated linear unit (GLU) for vector, scalar, and pseudoscalar features.

    Scalar and pseudoscalar gates are computed from scalar features (parity-even quantities);
    vector gates are computed from Lorentz inner products of (transformed) vector features. Gating
    a parity-odd pseudoscalar pre-activation with a parity-even gate keeps the output parity-odd.

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
        cp_triple_product: bool = False,
        cp_scalar_pseudo_mixing: bool = False,
    ) -> None:
        super().__init__()
        self._out_s_channels = out_s_channels
        self._out_p_channels = out_p_channels
        self.linear = SlimPseudoLinear(
            in_v_channels=in_v_channels,
            out_v_channels=3 * out_v_channels,
            in_s_channels=in_s_channels,
            out_s_channels=2 * out_s_channels + out_p_channels,
            in_p_channels=in_p_channels,
            out_p_channels=out_p_channels,
            cp_triple_product=cp_triple_product,
            cp_scalar_pseudo_mixing=cp_scalar_pseudo_mixing,
        )
        self.nonlinearity = get_nonlinearity(nonlinearity)
        self.nonlinearity_v = (
            get_nonlinearity(nonlinearity_v) if nonlinearity_v is not None else self.nonlinearity
        )

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, pseudoscalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the GLU.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., in_v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., in_s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., in_p_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., out_v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., out_s_channels)``.
        outputs_p
            Pseudoscalar features of shape ``(..., out_p_channels)``.
        """
        v_full, s_full, p_pre = self.linear(vectors, scalars, pseudoscalars)
        v_pre, v_gates_1, v_gates_2 = v_full.chunk(3, dim=-2)
        s_pre = s_full[..., : self._out_s_channels]
        s_gates = s_full[..., self._out_s_channels : 2 * self._out_s_channels]
        p_gates = s_full[..., 2 * self._out_s_channels :]

        v_gates = self._get_inner_product(v_gates_1, v_gates_2)

        outputs_v = self.nonlinearity_v(v_gates) * v_pre
        outputs_s = self.nonlinearity(s_gates) * s_pre
        outputs_p = self.nonlinearity(p_gates) * p_pre
        return outputs_v, outputs_s, outputs_p

    @minimum_autocast_precision(torch.float32)
    def _get_inner_product(self, v_gates_1: torch.Tensor, v_gates_2: torch.Tensor) -> torch.Tensor:
        # 0.5 = 1/sqrt(4) controls the scale, like 1/sqrt(d_k) in attention
        return 0.5 * inner_product(v_gates_1, v_gates_2).unsqueeze(-1)


class SlimPseudoSelfAttention(nn.Module):
    """Self-attention for Lorentz vectors, scalars, and pseudoscalars.

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
        SlimPseudoDropout probability.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        p_channels: int,
        num_heads: int,
        attn_ratio: int = 1,
        dropout_prob: float | None = None,
        cp_triple_product: bool = False,
        cp_scalar_pseudo_mixing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_v_channels = max(attn_ratio * v_channels // num_heads, 1)
        self.hidden_s_channels = max(attn_ratio * s_channels // num_heads, 4)
        self.hidden_p_channels = max(attn_ratio * p_channels // num_heads, 1)
        self.num_heads = num_heads

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
            cp_triple_product=cp_triple_product,
            cp_scalar_pseudo_mixing=cp_scalar_pseudo_mixing,
        )
        self.linear_out = SlimPseudoLinear(
            in_v_channels=self.hidden_v_channels * self.num_heads,
            out_v_channels=v_channels,
            in_s_channels=self.hidden_s_channels * self.num_heads,
            out_s_channels=s_channels,
            in_p_channels=self.hidden_p_channels * self.num_heads,
            out_p_channels=p_channels,
            initialization="small",
            cp_triple_product=cp_triple_product,
            cp_scalar_pseudo_mixing=cp_scalar_pseudo_mixing,
        )
        self.norm = SlimPseudoRMSNorm(
            self.hidden_v_channels,
            self.hidden_s_channels,
            self.hidden_p_channels,
            elementwise_affine=False,
        )
        if dropout_prob is not None:
            self.dropout = SlimPseudoDropout(dropout_prob)
        else:
            self.dropout = None

    def _pre_attention_reshape(
        self, qkv_v: torch.Tensor, qkv_s: torch.Tensor, qkv_p: torch.Tensor
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

        q_v = q_v * self.metric.to(q_v.dtype)

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
            Lorentz vectors of shape ``(..., items, v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., items, s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., items, p_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, v_channels, 4)``.
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
        Expansion ratio for hidden channels.
    num_layers
        Total number of layers (must be ``>= 2``).
    dropout_prob
        SlimPseudoDropout probability.
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
        cp_triple_product: bool = False,
        cp_scalar_pseudo_mixing: bool = False,
    ) -> None:
        super().__init__()
        assert num_layers >= 2
        layers: list[nn.Module] = []

        v_channels_list = [v_channels] + [mlp_ratio * v_channels] * (num_layers - 1) + [v_channels]
        s_channels_list = [s_channels] + [mlp_ratio * s_channels] * (num_layers - 1) + [s_channels]
        p_channels_list = [p_channels] * (num_layers + 1)

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
                    cp_triple_product=cp_triple_product,
                    cp_scalar_pseudo_mixing=cp_scalar_pseudo_mixing,
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
                cp_triple_product=cp_triple_product,
                cp_scalar_pseudo_mixing=cp_scalar_pseudo_mixing,
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
            Lorentz vectors of shape ``(..., v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., p_channels)``.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., v_channels, 4)``.
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
        Number of layers in the MLP.
    dropout_prob
        SlimPseudoDropout probability.
    norm_elementwise_affine
        Whether the pre-norms use a learnable per-channel gain.
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
        cp_triple_product: bool = False,
        cp_scalar_pseudo_mixing: bool = False,
    ) -> None:
        super().__init__()

        self.norm1 = SlimPseudoRMSNorm(
            v_channels, s_channels, p_channels, elementwise_affine=norm_elementwise_affine
        )
        self.norm2 = SlimPseudoRMSNorm(
            v_channels, s_channels, p_channels, elementwise_affine=norm_elementwise_affine
        )

        self.attention = SlimPseudoSelfAttention(
            v_channels=v_channels,
            s_channels=s_channels,
            p_channels=p_channels,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
            cp_triple_product=cp_triple_product,
            cp_scalar_pseudo_mixing=cp_scalar_pseudo_mixing,
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
            cp_triple_product=cp_triple_product,
            cp_scalar_pseudo_mixing=cp_scalar_pseudo_mixing,
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
            Lorentz vectors of shape ``(..., items, v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., items, s_channels)``.
        pseudoscalars
            Pseudoscalar features of shape ``(..., items, p_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, v_channels, 4)``.
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
