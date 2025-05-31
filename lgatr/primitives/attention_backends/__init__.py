"""Dynamic attention backend selection."""
from importlib import metadata

_REGISTRY = {}
for ep in metadata.entry_points(group="lgatr.primitives.attention_backends"):
    try:
        module = ep.load()
    except ImportError:
        continue
    _REGISTRY[ep.name] = module

# common kwargs used in custom attention backends
XFORMERS_KWARGS = ["attn_bias", "op"]
FLEX_KWARGS = ["score_mod", "block_mask"]


def get_attention_backend(**kwargs):
    """
    Dynamically determine the attention backend based on the extra keyword arguments.

    Implemented backends:
    - PyTorch's default attention: torch.nn.functional.scaled_dot_product_attention
    - Xformers attention: xformers.ops.memory_efficient_attention
    - PyTorch's flex_attention: torch.nn.attention.flex_attention.flex_attention
    """
    # check if backend is explicitly specified
    backend = kwargs.get("backend", None)
    if backend in _REGISTRY:
        return _REGISTRY[backend].attention

    # automatic fall-back based on other **kwargs
    if any(kwargs.get(kwarg, None) is not None for kwarg in XFORMERS_KWARGS):
        return _REGISTRY["xformers_attention"].attention
    if any(kwargs.get(kwarg, None) is not None for kwarg in FLEX_KWARGS):
        return _REGISTRY["flex_attention"].attention

    # fall-back to default torch attention
    try:
        return _REGISTRY["default_attention"].attention
    except KeyError:
        raise RuntimeError(
            f"No attention backend could be resolved. Available backends: {list(_REGISTRY)}"
        )
