"""xformers memory-efficient attention backend."""
try:
    from xformers.ops import memory_efficient_attention
except ModuleNotFoundError:
    raise ImportError(
        "xformers is not installed. Run 'pip install lgatr[xformers_attention]'."
    )


def attention(query, key, value, **kwargs):
    """Pass to xformers memory-efficient attention.
    Note that this xformers expects the shape (batch, head, items_out, channel).

    Parameters
    ----------
    query : torch.Tensor
        Queries with shape (batch, head, items_out, channel)
    key : torch.Tensor
        Keys with shape (batch, head, items_in, channel)
    value : torch.Tensor
        Values with shape (batch, head, items_in, channel)
    **kwargs
        Additional keyword arguments passed to memory_efficient_attention.

    Returns
    -------
    out : torch.Tensor
        Result with shape (batch, head, items_out, channel)
    """
    assert (
        len(query.shape) == 4
    ), "xformers constrains attention input shape to (batch, head, items, channel)."
    if key.shape[1] != query.shape[1]:
        # manual broadcasting for key and value; required for multi-query attention
        key = key.expand(key.shape[0], query.shape[1], *key.shape[2:])
        value = value.expand(value.shape[0], query.shape[1], *value.shape[2:])

    # xformers expects input shape (batch, item, head, channel)
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    out = memory_efficient_attention(query, key, value, **kwargs)
    out = out.transpose(1, 2).contiguous()
    return out
