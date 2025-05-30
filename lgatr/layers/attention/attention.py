"""Self-attention layers."""

from torch import nn

from ...primitives.attention import sdp_attention
from .config import SelfAttentionConfig


class GeometricAttention(nn.Module):
    """Geometric attention layer.

    This is the main attention mechanism used in L-GATr.

    Given multivector and scalar queries, keys, and values, this layer computes:

    .. code-block::

        attn_weights[..., i, j] = softmax_j[
            ga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
            + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
        ]
        out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
        out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm

    Parameters
    ----------
    config : SelfAttentionConfig
        Attention configuration.
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()

    def forward(self, q_mv, k_mv, v_mv, q_s, k_s, v_s, **attn_kwargs):
        """Forward pass through geometric attention.

        Given multivector and scalar queries, keys, and values, this forward pass computes:

        .. code-block::

            attn_weights[..., i, j] = softmax_j[
                ga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
                + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
            ]
            out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
            out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm

        Parameters
        ----------
        q_mv : torch.Tensor
            Multivector queries with shape (..., items_out, mv_channels, 16).
        k_mv : torch.Tensor
            Multivector keys with shape (..., items_in, mv_channels, 16).
        v_mv : torch.Tensor
            Multivector values with shape (..., items_in, mv_channels, 16).
        q_s : torch.Tensor
            Scalar queries with shape (..., items_out, s_channels).
        k_s : torch.Tensor
            Scalar keys with shape (..., items_in, s_channels).
        v_s : torch.Tensor
            Scalar values with shape (..., items_in, s_channels).
        **attn_kwargs
            Optional keyword arguments passed to attention.
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
