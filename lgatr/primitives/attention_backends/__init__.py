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

# Pip extra that provides each optional backend (used in dispatch error messages).
BACKEND_EXTRAS: dict[str, str] = {
    "varlen": "lgatr[varlen-attention]",
    "xformers": "lgatr[xformers-attention]",
    "flex": "lgatr[flex-attention]",
    "flash": "lgatr[flash-attention]",
}


@lru_cache
def _cuda_available() -> bool:
    """Whether a CUDA device is available (cached; gates CPU-only backends)."""
    return torch.cuda.is_available()


# Entry points are discovered at import (cheap: no backend module is imported yet). Each backend
# is loaded lazily on first use so that importing lgatr does not pull in xformers / flash-attn /
# etc. for users who never touch those backends.
_ENTRY_POINTS = {
    ep.name: ep for ep in metadata.entry_points(group="lgatr.primitives.attention_backends")
}
_RESOLVED: dict[str, object] = {}  # name -> loaded backend module
_UNAVAILABLE: dict[str, str] = {}  # name -> reason it could not be loaded


def _resolve_backend(name: str) -> object | None:
    """Load and cache a backend module by name, or return None if it is unavailable.

    The result (module or unavailability reason) is cached, so each backend is imported at most
    once. ``ImportError`` (missing optional dependency) and CPU-unsupported backends are treated
    as unavailable rather than raising.
    """
    if name in _RESOLVED:
        return _RESOLVED[name]
    if name in _UNAVAILABLE:
        return None
    ep = _ENTRY_POINTS.get(name)
    if ep is None:
        _UNAVAILABLE[name] = "unknown backend"
        return None
    if name in ("xformers", "flash") and not _cuda_available():
        _UNAVAILABLE[name] = "xformers and flash-attn are not available on CPU"
        return None
    try:
        module = ep.load()
    except ImportError as err:
        _UNAVAILABLE[name] = str(err)
        return None
    _RESOLVED[name] = module
    return module


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
    backend. Backends are imported lazily on first use.
    """
    # check if backend is explicitly specified
    backend = kwargs.get("backend", None)
    if backend is not None:
        module = _resolve_backend(backend)
        if module is None:
            raise ValueError(_backend_unavailable_message(backend))
        return module.attention

    # automatic fall-back based on other **kwargs
    for backend_name, backend_kwargs in BACKEND_KWARGS.items():
        if any(kwargs.get(k) is not None for k in backend_kwargs):
            module = _resolve_backend(backend_name)
            if module is None:
                raise ValueError(_backend_unavailable_message(backend_name))
            return module.attention

    # fall-back to native torch attention (always available)
    native = _resolve_backend("native")
    if native is None:
        raise RuntimeError("The native attention backend failed to load.")
    return native.attention


def _backend_unavailable_message(backend: str) -> str:
    """Build a dispatch error naming the missing backend and how to enable it."""
    known = ", ".join(sorted(_ENTRY_POINTS))
    if backend not in _ENTRY_POINTS:
        return f"Unknown attention backend {backend!r}. Known backends: {known}."
    reason = _UNAVAILABLE.get(backend)
    if reason and reason != "unknown backend":
        detail = f" ({reason})"
    else:
        extra = BACKEND_EXTRAS.get(backend)
        detail = f" (install with 'pip install {extra}')" if extra else ""
    return f"Attention backend {backend!r} is not available.{detail} Known backends: {known}."
