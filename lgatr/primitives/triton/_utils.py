"""Shared helpers for the Triton kernels."""

import torch

TRITON_DTYPES = frozenset({torch.float32, torch.float16, torch.bfloat16})

DTYPE_TAGS: dict[torch.dtype, int] = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def can_dispatch(*tensors: torch.Tensor) -> bool:
    """True iff every tensor is on CUDA and has a Triton-supported dtype."""
    return all(t.is_cuda and t.dtype in TRITON_DTYPES for t in tensors)
