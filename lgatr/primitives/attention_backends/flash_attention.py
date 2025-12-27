"""original flash-attention backend."""

import torch

try:
    from flash_attn import flash_attn_varlen_func
except ModuleNotFoundError as err:
    raise ImportError(
        "flash-attn is not installed. Run 'pip install lgatr[flash-attention]'."
    ) from err


# There is no fancy docs website, so one has to check the source code for the interface:
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py
def attention(q, k, v, dtype=torch.bfloat16, **kwargs):
    in_dtype = q.dtype

    def prepare(x):
        assert x.shape[0] == 1
        return x.to(dtype).squeeze(0).transpose(0, 1).contiguous()

    q, k, v = prepare(q), prepare(k), prepare(v)
    out = flash_attn_varlen_func(q, k, v, **kwargs)
    out = out.transpose(0, 1).unsqueeze(0).contiguous()
    return out.to(in_dtype)
