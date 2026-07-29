"""Equivariant primitives (linear, bilinear, attention, invariants, normalization, dropout)."""

from .attention import sdp_attention
from .bilinear import geometric_product
from .config import PrimitivesConfig
from .dropout import grade_dropout
from .invariants import abs_squared_norm, inner_product
from .linear import equi_linear, grade_involute, grade_project, reverse
from .normalization import equi_layer_norm
