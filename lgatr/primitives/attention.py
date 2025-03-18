from typing import Callable, Tuple

import torch
from einops import rearrange
from torch import Tensor

# flex_attention should be torch.compile'd for performance
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

from .invariants import _load_inner_product_factors


def sdp_attention(
    q_mv: Tensor,
    k_mv: Tensor,
    v_mv: Tensor,
    q_s: Tensor,
    k_s: Tensor,
    v_s: Tensor,
    **attn_kwargs,
) -> Tuple[Tensor, Tensor]:
    """Equivariant geometric attention based on scaled dot products.

    Expects both multivector and scalar queries, keys, and values as inputs.
    Then this function computes multivector and scalar outputs in the following way:

    ```
    attn_weights[..., i, j] = softmax_j[
        ga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Parameters
    ----------
    q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
        Queries, multivector part.
    k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
        Keys, multivector part.
    v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
        Values, multivector part.
    q_s : Tensor with shape (..., num_items_out, num_s_channels_in)
        Queries, scalar part.
    k_s : Tensor with shape (..., num_items_in, num_s_channels_in)
        Keys, scalar part.
    v_s : Tensor with shape (..., num_items_in, num_s_channels_out)
        Values, scalar part.
    **attn_kwargs
        Optional keyword arguments passed to attention.

    Returns
    -------
    outputs_mv : Tensor with shape (..., num_items_out, num_mv_channels_out, 16)
        Result, multivector part
    outputs_s : Tensor with shape (..., num_items_out, num_s_channels_out)
        Result, scalar part
    """

    # Construct queries and keys by concatenating relevant MV components and aux scalars
    q = torch.cat(
        [
            rearrange(
                q_mv
                * _load_inner_product_factors(device=q_mv.device, dtype=q_mv.dtype),
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

    v_out_mv = rearrange(
        v_out[..., : num_channels_out * 16], "... (c x) -> ...  c x", x=16
    )
    v_out_s = v_out[..., num_channels_out * 16 :]

    return v_out_mv, v_out_s


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: torch.Tensor = None,
    is_causal: bool = False,
    score_mod: Callable = None,
    block_mask: BlockMask = None,
) -> Tensor:
    """Execute scaled dot-product attention.
    The code dynamically selects the backend based on the arguments.
    Currently, torch SDPA and flex_attention backends are implemented.

    Parameters
    ----------
    query : Tensor
        of shape [batch, head, item, d]
    key : Tensor
        of shape [batch, head, item, d]
    value : Tensor
        of shape [batch, head, item, d]
    attn_mask: torch.Tensor
        Bias tensor, used in torch SDPA.
    is_causal: bool
        Make attention mask causal, used in torch SDPA.
    score_mod: Callable
        Function to modify attention scores.
        By default no score_mod is applied.
        Used in flex-attention.
    block_mask: BlockMask
        BlockMask object that controls the blocksparsity pattern of the attention.
        Used in flex-attention.

    Returns
    -------
    Tensor
        of shape [batch, head, item, d]
    """
    if attn_mask is not None or is_causal:
        return torch_sdpa(query, key, value, attn_mask=attn_mask, is_causal=is_causal)
    elif score_mod is not None or block_mask is not None:
        return flex_attention_flexembedding(
            query, key, value, score_mod=score_mod, block_mask=block_mask
        )
    else:
        # default: torch SDPA
        return torch_sdpa(query, key, value, attn_mask=attn_mask, is_causal=is_causal)


def flex_attention_flexembedding(query, key, value, **kwargs):
    # hotfix in case d is not a power of 2 (not supported yet in flex_attention)
    # proper solution in https://github.com/pytorch/pytorch/pull/133495/files
    # TODO: remove this once torch 2.7 is released (at 23.04.2025)

    # zero-pad the input to the next power of 2
    if not _is_power_of_two(query.shape[-1]):
        n = _next_power_of_two(query.shape[-1])
        query = torch.nn.functional.pad(query, (0, n - query.shape[-1]))
        key = torch.nn.functional.pad(key, (0, n - key.shape[-1]))
    if not _is_power_of_two(value.shape[-1]):
        n = _next_power_of_two(value.shape[-1])
        value_mod = torch.nn.functional.pad(value, (0, n - value.shape[-1]))
    else:
        value_mod = value

    out = flex_attention(query, key, value_mod, **kwargs)
    out = out[..., : value.shape[-1]]
    return out


def _is_power_of_two(n):
    return (n & (n - 1)) == 0


def _next_power_of_two(n):
    return 2 ** (n - 1).bit_length()
