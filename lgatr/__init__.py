from lgatr.interface.scalar import embed_scalar, extract_scalar
from lgatr.interface.spurions import get_num_spurions, get_spurions
from lgatr.interface.vector import embed_vector, extract_vector
from lgatr.layers.attention.config import SelfAttentionConfig
from lgatr.layers.mlp.config import MLPConfig
from lgatr.nets.lgatr import LGATr
from lgatr.primitives.config import gatr_config

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
