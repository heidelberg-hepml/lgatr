"""Original flash-attention backend."""

import torch

try:
    from flash_attn import flash_attn_varlen_func
except ModuleNotFoundError as err:
    raise ImportError(
        "flash-attn is not installed. Run 'pip install lgatr[flash-attention]'."
    ) from err


@torch.compiler.disable()
def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype | None = None,
    **kwargs,
) -> torch.Tensor:
    """Forward to flash-attention's ``flash_attn_varlen_func``.

    flash-attention expects shape ``(batch=1, items, head, channel)`` internally; this wrapper
    transposes between the L-GATr layout and that.
    See https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py.

    Parameters
    ----------
    query
        Queries of shape ``(batch, head, items_out, channel)``.
    key
        Keys of shape ``(batch, head, items_in, channel)``.
    value
        Values of shape ``(batch, head, items_in, channel)``.
    dtype
        If specified, cast input tensors to this dtype before passing to flash-attention. If None,
        use ``torch.get_autocast_gpu_dtype()``.
    **kwargs
        Additional keyword arguments forwarded to ``flash_attn_varlen_func``.

    Returns
    -------
    out
        Result of shape ``(batch, head, items_out, channel)``.
    """
    assert len(query.shape) == 4, (
        "flash-attn constrains attention input shape to (batch, head, items, channel)."
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
    out = flash_attn_varlen_func(query, key, value, **kwargs)
    out = out.transpose(0, 1).unsqueeze(0).contiguous()

    if in_dtype is not None:
        out = out.to(in_dtype)
    return out
