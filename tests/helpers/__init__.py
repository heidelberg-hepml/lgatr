from .constants import (
    BATCH_DIMS,
    COMPILE_SUPPORTED,
    MILD_TOLERANCES,
    STRICT_TOLERANCES,
    TOLERANCES,
    TORCH_VERSION,
)
from .equivariance import check_equivariance, check_pin_equivariance, check_pin_invariance
from .geometric_algebra import check_against_clifford
