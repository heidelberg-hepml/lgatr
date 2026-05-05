"""Equivariant attention."""

import torch

from .attention_backends import get_attention_backend
from .invariants import _load_inner_product_factors


def sdp_attention(
    q_mv: torch.Tensor,
    k_mv: torch.Tensor,
    v_mv: torch.Tensor,
    q_s: torch.Tensor | None = None,
    k_s: torch.Tensor | None = None,
    v_s: torch.Tensor | None = None,
    **attn_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Equivariant geometric attention based on scaled dot products.

    Expects multivector and (optionally) scalar queries, keys, and values, and computes:

    .. code-block::

        attn_weights[..., i, j] = softmax_j[
            ga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
            + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
        ]
        outputs_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
        outputs_s[..., i, c]     = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm

    Parameters
    ----------
    q_mv
        Multivector queries of shape ``(..., items_out, mv_channels, 16)``.
    k_mv
        Multivector keys of shape ``(..., items_in, mv_channels, 16)``.
    v_mv
        Multivector values of shape ``(..., items_in, mv_channels, 16)``.
    q_s
        Optional scalar queries of shape ``(..., items_out, s_channels)``. If None, the scalar
        inner product is omitted and ``outputs_s`` is None.
    k_s
        Optional scalar keys of shape ``(..., items_in, s_channels)``. Must be None iff ``q_s`` is None.
    v_s
        Optional scalar values of shape ``(..., items_in, s_channels)``. Must be None iff ``q_s`` is None.
    **attn_kwargs
        Optional keyword arguments forwarded to the attention backend.

    Returns
    -------
    outputs_mv
        Multivector result of shape ``(..., items_out, mv_channels, 16)``.
    outputs_s
        Scalar result of shape ``(..., items_out, s_channels)``, or None if ``q_s`` is None.
    """

    # Construct queries and keys by concatenating relevant MV components and aux scalars
    q = (q_mv * _load_inner_product_factors(device=q_mv.device, dtype=q_mv.dtype)).flatten(-2, -1)
    k = k_mv.flatten(-2, -1)
    num_channels_out = v_mv.shape[-2]
    v = v_mv.flatten(-2, -1)

    if q_s is not None:
        q = torch.cat([q, q_s], -1)
        k = torch.cat([k, k_s], -1)
        v = torch.cat([v, v_s], -1)

    outputs = scaled_dot_product_attention(q, k, v, **attn_kwargs)

    outputs_mv = outputs[..., : num_channels_out * 16].unflatten(-1, (-1, 16))
    outputs_s = None if q_s is None else outputs[..., num_channels_out * 16 :]

    return outputs_mv, outputs_s


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **attn_kwargs,
) -> torch.Tensor:
    """Execute scaled dot-product attention.

    The attention backend is determined dynamically based on the ``attn_kwargs`` provided
    (see :func:`lgatr.primitives.attention_backends.get_attention_backend`).

    Parameters
    ----------
    query
        Tensor of shape ``(..., items_out, channels)``.
    key
        Tensor of shape ``(..., items_in, channels)``.
    value
        Tensor of shape ``(..., items_in, channels)``.
    **attn_kwargs
        Optional keyword arguments forwarded to the attention backend.

    Returns
    -------
    outputs
        Tensor of shape ``(..., items_out, channels)``.
    """
    attention_backend = get_attention_backend(**attn_kwargs)
    return attention_backend(query, key, value, **attn_kwargs)
