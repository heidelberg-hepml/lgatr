"""Native PyTorch varlen scaled-dot-product attention implementation."""

import torch

try:
    from torch.nn.attention.varlen import varlen_attn
except ModuleNotFoundError as err:
    raise ImportError(
        "torch>=2.10 is not installed. Run 'pip install lgatr[varlen-attention]'."
    ) from err


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype | None = None,
    **kwargs,
) -> torch.Tensor:
    """Forward to PyTorch's native ``varlen_attn``.

    PyTorch's ``varlen_attn`` closely follows flash-attention (see ``flash.py``) and expects shape
    ``(batch=1, items, head, channel)`` internally; this wrapper transposes between the L-GATr
    layout and that.

    Parameters
    ----------
    query
        Queries of shape ``(batch, head, items_out, channel)``.
    key
        Keys of shape ``(batch, head, items_in, channel)``.
    value
        Values of shape ``(batch, head, items_in, channel)``.
    dtype
        If specified, cast input tensors to this dtype before passing to ``varlen_attn``. If None,
        use ``torch.get_autocast_gpu_dtype()``.
    **kwargs
        Additional keyword arguments forwarded to ``varlen_attn``.

    Returns
    -------
    out
        Result of shape ``(batch, head, items_out, channel)``.
    """
    assert len(query.shape) == 4, (
        "varlen_attn constrains attention input shape to (batch, head, items, channel)."
    )

    if query.dtype not in [torch.float16, torch.bfloat16]:
        # flash-attention only supports fp16 and bf16
        if dtype is None:
            dtype = torch.get_autocast_gpu_dtype()
        in_dtype = query.dtype
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    else:
        in_dtype = None

    def reshape(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == 1
        return x.squeeze(0).transpose(0, 1).contiguous()

    query, key, value = reshape(query), reshape(key), reshape(value)
    out = varlen_attn(query, key, value, **kwargs)
    out = out.transpose(0, 1).unsqueeze(0).contiguous()

    if in_dtype is not None:
        out = out.to(in_dtype)
    return out
