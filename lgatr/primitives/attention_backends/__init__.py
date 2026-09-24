"""Dynamic attention backend selection."""

from collections.abc import Callable
from importlib import metadata

import torch

from . import native

# Backends keyed by their distinguishing kwargs; iteration order is the dispatch priority.
BACKEND_KWARGS: dict[str, list[str]] = {
    "varlen": ["cu_seq_q", "cu_seq_k", "max_q", "max_k"],
    "xformers": ["attn_bias", "op"],
    "flex": ["score_mod", "block_mask"],
    "flash": ["cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"],
}


# Every installed backend is imported here, at import time
_REGISTRY: dict[str, object] = {"native": native}  # name -> backend module
_UNAVAILABLE: dict[str, str] = {}  # name -> reason it could not be loaded
for ep in metadata.entry_points(group="lgatr.primitives.attention_backends"):
    if ep.name in ("xformers", "flash") and not torch.cuda.is_available():
        _UNAVAILABLE[ep.name] = "xformers and flash-attn are not available on CPU"
        continue
    try:
        _REGISTRY[ep.name] = ep.load()
    except ImportError as err:
        _UNAVAILABLE[ep.name] = str(err)


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
    if backend is not None:
        if backend not in _REGISTRY:
            raise ValueError(_backend_unavailable_message(backend))
        return _REGISTRY[backend].attention

    # automatic fall-back based on other **kwargs
    for backend_name, backend_kwargs in BACKEND_KWARGS.items():
        if any(kwargs.get(k) is not None for k in backend_kwargs):
            if backend_name not in _REGISTRY:
                raise ValueError(_backend_unavailable_message(backend_name))
            return _REGISTRY[backend_name].attention

    # fall-back to native torch attention (always available)
    return native.attention


def _backend_unavailable_message(backend: str) -> str:
    """Build a dispatch error naming the missing backend and why it could not be loaded."""
    known = ", ".join(sorted(_REGISTRY.keys() | _UNAVAILABLE.keys()))
    if backend not in _REGISTRY and backend not in _UNAVAILABLE:
        return f"Unknown attention backend {backend!r}. Known backends: {known}."
    reason = _UNAVAILABLE.get(backend)
    detail = f" ({reason})" if reason else ""
    return f"Attention backend {backend!r} is not available.{detail} Known backends: {known}."
