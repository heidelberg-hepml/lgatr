"""Helpers for using L-GATr primitives with :func:`torch.compile`."""

from __future__ import annotations

import torch

from .bilinear import _compute_sparse_gp_indices, _load_geometric_product_tensor
from .invariants import _load_inner_product_factors, _load_metric_grades
from .linear import (
    _compute_dual_sign,
    _compute_grade_involution,
    _compute_grade_projection_mask,
    _compute_pin_equi_linear_basis,
    _compute_reversal,
)


def warmup_caches(device: torch.device | str, dtype: torch.dtype = torch.float32) -> None:
    """Pre-populate L-GATr's primitive caches for a given device and dtype.

    Without warming, the first call for a new ``(device, dtype)`` pair triggers host-to-device
    copies that partition the captured graph under :func:`torch.compile` with
    ``mode="reduce-overhead"``. Calling this helper once per ``(device, dtype)`` before
    compiling avoids the partition.

    Parameters
    ----------
    device
        Target device, either a :class:`torch.device` or a string like ``"cuda"`` or ``"cpu"``.
    dtype
        Floating-point dtype matching what the model will run in.
    """
    device = torch.device(device)
    for use_subgroup in (True, False):
        _compute_pin_equi_linear_basis(use_subgroup, device=device, dtype=dtype)
    _compute_grade_projection_mask(device=device, dtype=dtype)
    _compute_reversal(device=device, dtype=dtype)
    _compute_grade_involution(device=device, dtype=dtype)
    _compute_dual_sign(device=device, dtype=dtype)
    _load_geometric_product_tensor(device=device, dtype=dtype)
    _compute_sparse_gp_indices(device=device, dtype=dtype)
    _load_inner_product_factors(device=device, dtype=dtype)
    _load_metric_grades(device=device, dtype=dtype)
