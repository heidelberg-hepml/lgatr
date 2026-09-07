"""Original flash-attention backend."""

import torch

try:
    from flash_attn import flash_attn_varlen_func
except ModuleNotFoundError as err:
    raise ImportError(
        "flash-attn is not installed. Run 'pip install lgatr[flash-attention]'."
    ) from err

if not hasattr(torch, "compiler"):  # torch.compiler was added in torch 2.1
    raise ImportError("torch>=2.1 is not installed. Run 'pip install lgatr[flash-attention]'.")


@torch.compiler.disable()
def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
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
        raise ValueError(
            f"query.dtype={query.dtype}, but flash attention only supports float16, bfloat16"
        )

    def reshape(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == 1
        return x.squeeze(0).transpose(0, 1).contiguous()

    query, key, value = reshape(query), reshape(key), reshape(value)
    # no head_dim padding here (unlike varlen.py): flash-attn pads head_dim to a multiple of 8
    # internally, computing the scale from the original dim and slicing the output back.
    out = flash_attn_varlen_func(query, key, value, **kwargs)
    out = out.transpose(0, 1).unsqueeze(0).contiguous()
    return out
