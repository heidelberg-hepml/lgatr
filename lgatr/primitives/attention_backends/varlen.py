"""Native PyTorch varlen scaled-dot-product attention implementation."""

import torch

try:
    from torch.nn.attention.varlen import varlen_attn
except ModuleNotFoundError as err:
    raise ImportError(
        "torch>=2.10 is not installed. Run 'pip install lgatr[varlen-attention]'."
    ) from err


@torch.compiler.disable(reason="Currently breaks torch.compile")
def attention(query, key, value, dtype=None, **kwargs):
    """Pass to pytorchs native varlen_attn.
    Note that pytorchs native varlen_attn closely follows flash-attn, see flash.py.
    Note that flash-attention expects the shape (batch=1, items, head, channel).

    Parameters
    ----------
    query : torch.Tensor
        Queries with shape (batch, head, items_out, channel)
    key : torch.Tensor
        Keys with shape (batch, head, items_in, channel)
    value : torch.Tensor
        Values with shape (batch, head, items_in, channel)
    dtype : torch.dtype, optional
        If specified, cast input tensors to this dtype before passing to flash-attention.
        If None, use torch.get_autocast_gpu_dtype().
    **kwargs
        Additional keyword arguments passed to varlen_attn.

    Returns
    -------
    out : torch.Tensor
        Result with shape (batch, head, items_out, channel)
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

    def reshape(x):
        assert x.shape[0] == 1
        return x.squeeze(0).transpose(0, 1).contiguous()

    query, key, value = reshape(query), reshape(key), reshape(value)
    out = varlen_attn(query, key, value, **kwargs)
    out = out.transpose(0, 1).unsqueeze(0).contiguous()

    if in_dtype is not None:
        out = out.to(in_dtype)
    return out
