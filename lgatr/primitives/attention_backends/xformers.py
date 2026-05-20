"""xformers memory-efficient attention backend."""

import torch

try:
    from xformers.ops import memory_efficient_attention
except ModuleNotFoundError as err:
    raise ImportError(
        "xformers is not installed. Run 'pip install lgatr[xformers-attention]'."
    ) from err


@torch.compiler.disable()
def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype | None = None,
    **kwargs,
) -> torch.Tensor:
    """Forward to xformers' ``memory_efficient_attention``.

    xformers expects shape ``(batch, item, head, channel)`` internally; this wrapper transposes
    between the L-GATr layout and that.

    Parameters
    ----------
    query
        Queries of shape ``(batch, head, items_out, channel)``.
    key
        Keys of shape ``(batch, head, items_in, channel)``.
    value
        Values of shape ``(batch, head, items_in, channel)``.
    dtype
        If specified, cast input tensors to this dtype before passing to attention. Useful to
        trigger flash-attention.
    **kwargs
        Additional keyword arguments forwarded to ``memory_efficient_attention``.

    Returns
    -------
    out
        Result of shape ``(batch, head, items_out, channel)``.
    """
    assert len(query.shape) == 4, (
        "xformers constrains attention input shape to (batch, head, items, channel)."
    )
    if key.shape[1] != query.shape[1]:
        # manual broadcasting for key and value; required for multi-query attention
        key = key.expand(key.shape[0], query.shape[1], *key.shape[2:])
        value = value.expand(value.shape[0], query.shape[1], *value.shape[2:])

    if dtype is not None:
        in_dtype = query.dtype
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    else:
        in_dtype = None

    # xformers expects input shape (batch, item, head, channel)
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    out = memory_efficient_attention(query, key, value, **kwargs)
    out = out.transpose(1, 2).contiguous()

    if in_dtype is not None:
        out = out.to(in_dtype)
    return out
