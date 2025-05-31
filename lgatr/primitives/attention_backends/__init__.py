"""Dynamic attention backend selection."""
from importlib import metadata

_REGISTRY = {}
for ep in metadata.entry_points(group="lgatr.primitives.attention_backends"):
    try:
        module = ep.load()
    except ImportError:
        continue
    _REGISTRY[ep.name] = module


def get_attention_backend(**kwargs):
    """
    Dynamically determine the attention backend based on the extra keyword arguments.

    Implemented backends:
    - Default torch attention: torch.nn.functional.scaled_dot_product_attention
    - Xformers attention: xformers.ops.memory_efficient_attention
    """
    # check if backend is explicitly specified
    backend = kwargs.get("backend", None)
    if backend in _REGISTRY:
        return _REGISTRY[backend].attention

    # automatic fall-back based on other **kwargs
    if kwargs.get("attn_bias", None) is not None:
        return _REGISTRY["xformers_attention"].attention

    try:
        return _REGISTRY["default_attention"].attention
    except KeyError:
        raise RuntimeError(
            f"No attention backend could be resolved. Available backends: {list(_REGISTRY)}"
        )
