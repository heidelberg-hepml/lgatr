"""Geometric attention module."""

import torch
from torch import nn

from ...primitives.attention import sdp_attention
from .config import SelfAttentionConfig


class GeometricAttention(nn.Module):
    """Geometric attention layer.

    Given multivector and scalar queries, keys, and values, computes:

    .. code-block::

        attn_weights[..., i, j] = softmax_j[
            ga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
            + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
        ]
        outputs_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
        outputs_s[..., i, c]     = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm

    Parameters
    ----------
    config
        Attention configuration.
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()

    def forward(
        self,
        q_mv: torch.Tensor,
        k_mv: torch.Tensor,
        v_mv: torch.Tensor,
        q_s: torch.Tensor | None,
        k_s: torch.Tensor | None,
        v_s: torch.Tensor | None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        q_mv
            Multivector queries of shape ``(..., items_out, mv_channels, 16)``.
        k_mv
            Multivector keys of shape ``(..., items_in, mv_channels, 16)``.
        v_mv
            Multivector values of shape ``(..., items_in, mv_channels, 16)``.
        q_s
            Scalar queries of shape ``(..., items_out, s_channels)``.
        k_s
            Scalar keys of shape ``(..., items_in, s_channels)``.
        v_s
            Scalar values of shape ``(..., items_in, s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        h_mv
            Multivector result of shape ``(..., items_out, mv_channels, 16)``.
        h_s
            Scalar result of shape ``(..., items_out, s_channels)``, or None if ``q_s`` is None.
        """

        h_mv, h_s = sdp_attention(
            q_mv,
            k_mv,
            v_mv,
            q_s,
            k_s,
            v_s,
            **attn_kwargs,
        )

        return h_mv, h_s
