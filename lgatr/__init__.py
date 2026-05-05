"""L-GATr: a Lorentz-equivariant geometric-algebra transformer for high-energy physics."""

from importlib.metadata import version as _pkg_version

from .interface.axialvector import embed_axialvector, extract_axialvector
from .interface.pseudoscalar import embed_pseudoscalar, extract_pseudoscalar
from .interface.scalar import embed_scalar, extract_scalar
from .interface.spurions import get_num_spurions, get_spurions
from .interface.vector import embed_vector, extract_vector
from .layers.attention.config import CrossAttentionConfig, SelfAttentionConfig
from .layers.mlp.config import MLPConfig
from .nets.conditional_lgatr import ConditionalLGATr
from .nets.conditional_lgatr_slim import ConditionalLGATrSlim
from .nets.lgatr import LGATr
from .nets.lgatr_slim import LGATrSlim
from .primitives.compile import warmup_caches
from .primitives.config import PrimitivesConfig

__version__ = _pkg_version("lgatr")
