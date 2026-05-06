"""Dynamic attention backend selection."""

from collections.abc import Callable
from functools import lru_cache
from importlib import metadata

import torch

# Backends keyed by their distinguishing kwargs; iteration order is the dispatch priority.
BACKEND_KWARGS: dict[str, list[str]] = {
    "varlen": ["cu_seq_q", "cu_seq_k", "max_q", "max_k"],
    "xformers": ["attn_bias", "op"],
    "flex": ["score_mod", "block_mask"],
    "flash": ["cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"],
}


@lru_cache
def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device


_REGISTRY = {}
for ep in metadata.entry_points(group="lgatr.primitives.attention_backends"):
    try:
        # check if entry point code be loaded without ImportError
        module = ep.load()
    except ImportError:
        continue

    if ep.name in ["xformers", "flash"] and get_device() == torch.device("cpu"):
        # xformers and flash-attn are not available on CPU
        continue
    _REGISTRY[ep.name] = module


def get_attention_backend(**kwargs) -> Callable:
    """Resolve the attention backend based on the extra keyword arguments.

    Implemented backends:

    - PyTorch native attention: ``torch.nn.functional.scaled_dot_product_attention``
    - PyTorch varlen attention: ``torch.nn.attention.varlen.varlen_attn``
    - xformers attention: ``xformers.ops.memory_efficient_attention``
    - PyTorch flex_attention: ``torch.nn.attention.flex_attention.flex_attention``
    - Flash attention (variable sequence length): ``flash_attn.flash_attn_varlen_func``

    The backend is selected explicitly via ``backend=...`` if provided, otherwise inferred from
    backend-specific kwargs (e.g. ``cu_seqlens_*`` triggers flash). Falls back to the native
    backend.
    """
    # check if backend is explicitly specified
    backend = kwargs.get("backend", None)
    if backend in _REGISTRY:
        return _REGISTRY[backend].attention

    # automatic fall-back based on other **kwargs
    for backend_name, backend_kwargs in BACKEND_KWARGS.items():
        if any(kwargs.get(k) is not None for k in backend_kwargs):
            return _REGISTRY[backend_name].attention

    # fall-back to native torch attention
    try:
        return _REGISTRY["native"].attention
    except KeyError as err:
        raise RuntimeError(
            f"No attention backend could be resolved. Available backends: {list(_REGISTRY)}"
        ) from err
