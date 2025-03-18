from .interface.scalar import embed_scalar, extract_scalar
from .interface.spurions import get_num_spurions, get_spurions
from .interface.vector import embed_vector, extract_vector
from .layers.attention.config import SelfAttentionConfig
from .layers.mlp.config import MLPConfig
from .nets.lgatr import LGATr
from .primitives.config import gatr_config

__all__ = [
    "LGATr",
    "SelfAttentionConfig",
    "MLPConfig",
    "embed_scalar",
    "extract_scalar",
    "embed_vector",
    "extract_vector",
    "get_num_spurions",
    "get_spurions",
    "gatr_config",
]
__version__ = "1.0.0"
