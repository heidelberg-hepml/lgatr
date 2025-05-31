"""Default PyTorch scaled-dot-product attention implementation."""
from torch.nn.functional import scaled_dot_product_attention


attention = scaled_dot_product_attention
