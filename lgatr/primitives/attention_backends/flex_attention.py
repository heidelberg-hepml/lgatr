"""PyTorch's modern and flexible flex_attention backend."""

try:
    from torch.nn.attention.flex_attention import flex_attention
except ModuleNotFoundError as err:
    raise ImportError(
        "torch>=2.5 is not installed. Run 'pip install lgatr[flex_attention]'."
    ) from err

attention = flex_attention
