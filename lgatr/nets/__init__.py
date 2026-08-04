"""Top-level L-GATr networks: full multivector variants and slim (vector + scalar) variants."""

from .conditional_lgatr import ConditionalLGATr
from .conditional_slim import ConditionalLGATrSlim
from .lgatr import LGATr
from .slim import LGATrSlim
from .slim_pseudo import LGATrSlimPseudo
