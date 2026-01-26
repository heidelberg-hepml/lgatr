"""xformers memory-efficient attention backend."""

import torch

try:
    from xformers.ops import memory_efficient_attention
except ModuleNotFoundError as err:
    raise ImportError(
        "xformers is not installed. Run 'pip install lgatr[xformers-attention]'."
    ) from err


@torch.compiler.disable()
def attention(query, key, value, dtype=None, **kwargs):
    """Pass to xformers memory-efficient attention.
    Note that this xformers expects the shape (batch, head, items, channel).

    Parameters
    ----------
    query : torch.Tensor
        Queries with shape (batch, head, items_out, channel)
    key : torch.Tensor
        Keys with shape (batch, head, items_in, channel)
    value : torch.Tensor
        Values with shape (batch, head, items_in, channel)
    dtype : torch.dtype, optional
        If specified, cast input tensors to this dtype before passing to attention.
        This can be useful to trigger flash-attention.
    **kwargs
        Additional keyword arguments passed to memory_efficient_attention.

    Returns
    -------
    out : torch.Tensor
        Result with shape (batch, head, items_out, channel)
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
