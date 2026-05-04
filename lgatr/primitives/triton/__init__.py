"""Optional Triton kernels for :func:`equi_linear` and :func:`geometric_product`."""

import torch

_HAVE_TRITON: bool = False
try:
    if torch.cuda.is_available():
        import triton  # noqa: F401

        _HAVE_TRITON = True
except (ImportError, RuntimeError, OSError):
    pass

if _HAVE_TRITON:
    from .bilinear import geometric_product_triton
    from .linear import equi_linear_triton
