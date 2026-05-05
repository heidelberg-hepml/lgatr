"""Query/key/value projections for self-attention (multi-head and multi-query variants)."""

import torch
from torch import nn

from ..layer_norm import EquiLayerNorm
from ..linear import EquiLinear
from .config import SelfAttentionConfig


class QKVModule(nn.Module):
    """Compute multivector and scalar queries, keys, and values for multi-head self-attention.

    Used by self-attention only; cross-attention does the equivalent computation manually.

    Parameters
    ----------
    config
        Attention configuration.
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()
        self.in_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=3 * config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=3 * config.hidden_s_channels * config.num_heads,
        )
        self.norm_qkv = EquiLayerNorm()
        self.config = config

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor | None = None,
        additional_qk_features_mv: torch.Tensor | None = None,
        additional_qk_features_s: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute head-wise queries, keys, and values.

        The heads have size ``head_mv_channels = mv_channels * increase_hidden_channels // num_heads``
        and ``head_s_channels = s_channels * increase_hidden_channels // num_heads``.

        Parameters
        ----------
        multivectors
            Multivector inputs of shape ``(..., items, mv_channels, 16)``.
        scalars
            Optional scalar inputs of shape ``(..., items, s_channels)``. If None, the scalar
            Q/K/V outputs are also None.
        additional_qk_features_mv
            Additional multivector features for the Q/K computation.
        additional_qk_features_s
            Additional scalar features for the Q/K computation.

        Returns
        -------
        q_mv
            Multivector queries of shape ``(..., heads, items, head_mv_channels, 16)``.
        k_mv
            Multivector keys of shape ``(..., heads, items, head_mv_channels, 16)``.
        v_mv
            Multivector values of shape ``(..., heads, items, head_mv_channels, 16)``.
        q_s
            Scalar queries of shape ``(..., heads, items, head_s_channels)``, or None.
        k_s
            Scalar keys of shape ``(..., heads, items, head_s_channels)``, or None.
        v_s
            Scalar values of shape ``(..., heads, items, head_s_channels)``, or None.
        """

        # Additional inputs
        if additional_qk_features_mv is not None:
            multivectors = torch.cat((multivectors, additional_qk_features_mv), dim=-2)
        if scalars is not None and additional_qk_features_s is not None:
            scalars = torch.cat((scalars, additional_qk_features_s), dim=-1)

        qkv_mv, qkv_s = self.in_linear(
            multivectors, scalars
        )  # (..., num_items, 3 * hidden_channels * num_heads, 16)
        # "... items (qkv hidden num_heads) x -> qkv ... num_heads items hidden x"
        qkv_mv = qkv_mv.unflatten(
            -2, (3, self.config.hidden_mv_channels, self.config.num_heads)
        ).movedim((-4, -2), (0, -4))
        q_mv, k_mv, v_mv = qkv_mv.unbind(0)  # each: (..., num_heads, num_items, num_channels, 16)

        # Same, for optional scalar components
        if qkv_s is not None:
            # "... items (qkv hidden num_heads) -> qkv ... num_heads items hidden"
            qkv_s = qkv_s.unflatten(
                -1, (3, self.config.hidden_s_channels, self.config.num_heads)
            ).movedim((-3, -1), (0, -3))
            q_s, k_s, v_s = qkv_s.unbind(0)  # each: (..., num_heads, num_items, num_channels)
        else:
            q_s, k_s, v_s = None, None, None

        q_mv, q_s = self.norm_qkv(q_mv, scalars=q_s)
        k_mv, k_s = self.norm_qkv(k_mv, scalars=k_s)
        v_mv, v_s = self.norm_qkv(v_mv, scalars=v_s)

        return q_mv, k_mv, v_mv, q_s, k_s, v_s


class MultiQueryQKVModule(nn.Module):
    """Compute Q/K/V for multi-query self-attention (keys and values shared across heads).

    Used by self-attention only; cross-attention does the equivalent computation manually.

    Parameters
    ----------
    config
        Attention configuration.
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()

        # Q projection
        self.q_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=config.hidden_s_channels * config.num_heads,
        )

        # Key and value projections (shared between heads)
        self.k_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=config.hidden_mv_channels,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=config.hidden_s_channels,
        )
        self.v_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels,
            out_mv_channels=config.hidden_mv_channels,
            in_s_channels=config.in_s_channels,
            out_s_channels=config.hidden_s_channels,
        )
        self.norm_qkv = EquiLayerNorm()
        self.config = config

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor | None = None,
        additional_qk_features_mv: torch.Tensor | None = None,
        additional_qk_features_s: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute head-wise queries, keys, and values (multi-query).

        Keys and values are shared across heads, i.e. the head dimension is set to 1 and
        broadcast later.

        Parameters
        ----------
        multivectors
            Multivector inputs of shape ``(..., items, mv_channels, 16)``.
        scalars
            Optional scalar inputs of shape ``(..., items, s_channels)``. If None, the scalar
            Q/K/V outputs are also None.
        additional_qk_features_mv
            Additional multivector features for the Q/K computation (e.g. positions of objects).
        additional_qk_features_s
            Additional scalar features for the Q/K computation (e.g. object types).

        Returns
        -------
        q_mv
            Multivector queries of shape ``(..., heads, items, head_mv_channels, 16)``.
        k_mv
            Multivector keys of shape ``(..., 1, items, head_mv_channels, 16)``.
        v_mv
            Multivector values of shape ``(..., 1, items, head_mv_channels, 16)``.
        q_s
            Scalar queries of shape ``(..., heads, items, head_s_channels)``, or None.
        k_s
            Scalar keys of shape ``(..., 1, items, head_s_channels)``, or None.
        v_s
            Scalar values of shape ``(..., 1, items, head_s_channels)``, or None.
        """

        # Additional inputs
        if additional_qk_features_mv is not None:
            qk_multivectors = torch.cat((multivectors, additional_qk_features_mv), dim=-2)
        else:
            qk_multivectors = multivectors
        if scalars is not None and additional_qk_features_s is not None:
            qk_scalars = torch.cat((scalars, additional_qk_features_s), dim=-1)
        else:
            qk_scalars = scalars

        # Project to queries, keys, and values (multivector reps)
        q_mv, q_s = self.q_linear(
            qk_multivectors, qk_scalars
        )  # (..., num_items, hidden_channels * num_heads, 16)
        k_mv, k_s = self.k_linear(
            qk_multivectors, qk_scalars
        )  # (..., num_items, hidden_channels, 16)
        v_mv, v_s = self.v_linear(multivectors, scalars)  # (..., num_items, hidden_channels, 16)

        # Rearrange to (..., heads, items, channels, 16) shape
        q_mv = q_mv.unflatten(-2, (self.config.hidden_mv_channels, self.config.num_heads)).movedim(
            -2, -4
        )
        k_mv = k_mv.unsqueeze(-4)
        v_mv = v_mv.unsqueeze(-4)

        # Same for scalars
        if q_s is not None:
            q_s = q_s.unflatten(-1, (self.config.hidden_s_channels, self.config.num_heads)).movedim(
                -1, -3
            )
            k_s = k_s.unsqueeze(-3)
            v_s = v_s.unsqueeze(-3)
        else:
            q_s, k_s, v_s = None, None, None

        q_mv, q_s = self.norm_qkv(q_mv, scalars=q_s)
        k_mv, k_s = self.norm_qkv(k_mv, scalars=k_s)
        v_mv, v_s = self.norm_qkv(v_mv, scalars=v_s)

        return q_mv, k_mv, v_mv, q_s, k_s, v_s
