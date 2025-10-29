"""Equivariant attention."""

import torch
from einops import rearrange
from torch import Tensor

from .attention_backends import get_attention_backend
from .invariants import _load_inner_product_factors


def sdp_attention(
    q_mv: Tensor,
    k_mv: Tensor,
    v_mv: Tensor,
    q_s: Tensor,
    k_s: Tensor,
    v_s: Tensor,
    **attn_kwargs,
) -> tuple[Tensor, Tensor]:
    """Equivariant geometric attention based on scaled dot products.

    Expects both multivector and scalar queries, keys, and values as inputs.
    Then this function computes multivector and scalar outputs in the following way:

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
        Multivector queries with shape (..., items_out, mv_channels, 16)
    k_mv : torch.Tensor
        Multivector keys with shape (..., items_out, mv_channels, 16)
    v_mv : torch.Tensor
        Multivector values with shape (..., items_out, mv_channels, 16)
    q_s : torch.Tensor
        Scalar queries with shape (..., items_out, s_channels)
    k_s : torch.Tensor
        Scalar keys with shape (..., items_out, s_channels)
    v_s : torch.Tensor
        Scalar values with shape (..., items_out, s_channels)
    **attn_kwargs
        Optional keyword arguments passed to attention.

    Returns
    -------
    outputs_mv : torch.Tensor
        Multivector result with shape (..., items_out, mv_channels, 16)
    outputs_s : torch.Tensor
        Scalar result with shape (..., items_out, s_channels)
    """

    # Construct queries and keys by concatenating relevant MV components and aux scalars
    q = torch.cat(
        [
            rearrange(
                q_mv * _load_inner_product_factors(device=q_mv.device, dtype=q_mv.dtype),
                "... c x -> ... (c x)",
            ),
            q_s,
        ],
        -1,
    )
    k = torch.cat([rearrange(k_mv, "... c x -> ... (c x)"), k_s], -1)

    num_channels_out = v_mv.shape[-2]
    v = torch.cat([rearrange(v_mv, "... c x -> ... (c x)"), v_s], -1)

    v_out = scaled_dot_product_attention(q, k, v, **attn_kwargs)

    v_out_mv = rearrange(v_out[..., : num_channels_out * 16], "... (c x) -> ...  c x", x=16)
    v_out_s = v_out[..., num_channels_out * 16 :]

    return v_out_mv, v_out_s


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    **attn_kwargs,
) -> Tensor:
    """Execute scaled dot-product attention.
    The attention backend is determined dynamically
    based on the ``attn_kwargs`` provided.

    Parameters
    ----------
    query : torch.Tensor
        Tensor of shape (..., items_out, channels)
    key : torch.Tensor
        Tensor of shape (..., items_in, channels)
    value : torch.Tensor
        Tensor of shape (..., items_in, channels)
    **attn_kwargs
        Optional keyword arguments passed to attention.

    Returns
    -------
    torch.Tensor
        Tensor of shape (..., head, item_out, channels)
    """
    attention_backend = get_attention_backend(**attn_kwargs)
    return attention_backend(query, key, value, **attn_kwargs)
