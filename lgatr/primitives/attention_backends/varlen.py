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
    **kwargs
        Additional keyword arguments forwarded to ``varlen_attn``. ``scale`` is applied by this
        wrapper instead of being forwarded, as ``varlen_attn`` does not accept it.

    Returns
    -------
    out
        Result of shape ``(batch, head, items_out, channel)``.
    """
    assert len(query.shape) == 4, (
        "varlen_attn constrains attention input shape to (batch, head, items, channel)."
    )

    if query.dtype not in [torch.float16, torch.bfloat16]:
        raise ValueError(
            f"query.dtype={query.dtype}, but varlen attention only supports float16, bfloat16"
        )

    def reshape(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == 1
        return x.squeeze(0).transpose(0, 1).contiguous()

    query, key, value = reshape(query), reshape(key), reshape(value)

    head_dim = query.shape[-1]
    pad = -head_dim % 8
    scale = kwargs.pop("scale", None)
    if pad:
        # varlen_attn requires head_dim to be a multiple of 8; the zero-padding leaves the
        # query-key products untouched.
        query = torch.nn.functional.pad(query, (0, pad))
        key = torch.nn.functional.pad(key, (0, pad))
        value = torch.nn.functional.pad(value, (0, pad))
    if pad or scale is not None:
        # varlen_attn takes no ``scale`` argument and always normalizes by the head_dim it sees,
        # i.e. the padded one. Fold the intended scale into the queries to correct for that.
        scale = head_dim**-0.5 if scale is None else scale
        query = query * (scale * query.shape[-1] ** 0.5)
    out = varlen_attn(query, key, value, **kwargs)
    if pad:
        out = out[..., :head_dim]
    out = out.transpose(0, 1).unsqueeze(0).contiguous()
    return out
