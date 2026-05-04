"""Compile-time index tuples for the Triton kernels."""

import torch

from ..bilinear import _GP_INDICES, _GP_SIGNS
from ..linear import _BASIS_FULL, _BASIS_SUBGROUP, _DUAL_PERM, _DUAL_SIGN


def _to_int_tuple(t: torch.Tensor) -> tuple[int, ...]:
    """Flatten a tensor of small ints to a Python tuple."""
    return tuple(int(v) for v in t.flatten().tolist())


def _extract_grade_and_dual_basis_idx() -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-position grade label and dual basis-element index.

    The Python sparse path in :mod:`..linear` doesn't need these; they live here only because
    the Triton kernels consume them as ``tl.constexpr`` index tuples.
    """
    grade_of = torch.zeros(16, dtype=torch.int64)
    for a in range(5):
        idx = torch.diag(_BASIS_FULL[a]).nonzero(as_tuple=True)[0]
        grade_of[idx] = a

    dual_basis_idx = torch.zeros(16, dtype=torch.int64)
    for i in range(16):
        for a in range(5, 10):
            row = _BASIS_SUBGROUP[a, i]
            nz = row.nonzero(as_tuple=True)[0]
            if nz.numel() == 1:
                dual_basis_idx[i] = a
                break
    return grade_of, dual_basis_idx


_GRADE_OF, _DUAL_BASIS_IDX = _extract_grade_and_dual_basis_idx()
_DUAL_SIGN_BWD = _DUAL_SIGN[_DUAL_PERM]

# Per-grade tuples for the grade-grouped equi_linear kernel. d_g = (1, 4, 6, 4, 1).
GRADE_COMPS: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(i) for i in range(16) if int(_GRADE_OF[i]) == g) for g in range(5)
)
GRADE_DUAL_PERM: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(_DUAL_PERM[i]) for i in GRADE_COMPS[g]) for g in range(5)
)
GRADE_DUAL_SIGN_FWD: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(_DUAL_SIGN[i]) for i in GRADE_COMPS[g]) for g in range(5)
)
GRADE_DUAL_SIGN_BWD: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(_DUAL_SIGN_BWD[i]) for i in GRADE_COMPS[g]) for g in range(5)
)
# DUAL_BASIS_IDX is constant within each grade — pick the first component's value.
GRADE_DG_FWD: tuple[int, ...] = tuple(int(_DUAL_BASIS_IDX[GRADE_COMPS[g][0]]) for g in range(5))
GRADE_DG_BWD: tuple[int, ...] = tuple(
    int(_DUAL_BASIS_IDX[_DUAL_PERM[GRADE_COMPS[g][0]]]) for g in range(5)
)


def _build_gp_inverse() -> tuple[
    tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]
]:
    """Build the inverse-index map for grad_y, grouped by output ``k``.

    For each output ``k``, list the 16 ``(i, j)`` pairs and signs that satisfy
    ``GP_INDICES[i, j] == k``.
    """
    inv_i: list[list[int]] = [[] for _ in range(16)]
    inv_j: list[list[int]] = [[] for _ in range(16)]
    inv_s: list[list[int]] = [[] for _ in range(16)]
    for i in range(16):
        for j in range(16):
            k = int(_GP_INDICES[i, j])
            inv_i[k].append(i)
            inv_j[k].append(j)
            inv_s[k].append(int(_GP_SIGNS[i, j]))
    return (
        tuple(tuple(row) for row in inv_i),
        tuple(tuple(row) for row in inv_j),
        tuple(tuple(row) for row in inv_s),
    )


# 2-D ``(16, 16)`` GP tables for the half-unrolled kernel.
# Forward: ``out[k] = sum_n SI[k][n] * a[AI[k][n]] * b[BI[k][n]]`` with a=x, b=y.
GP_AI_FWD: tuple[tuple[int, ...], ...] = tuple(tuple(n for n in range(16)) for _ in range(16))
GP_BI_FWD: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(_GP_INDICES[k, n]) for n in range(16)) for k in range(16)
)
GP_SI_FWD: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(_GP_SIGNS[k, n]) for n in range(16)) for k in range(16)
)
# bwd_x: ``grad_x[j] = sum_n signs[n,j] * grad_out[n] * y[indices[n,j]]``.
GP_AI_BWD_X: tuple[tuple[int, ...], ...] = tuple(tuple(n for n in range(16)) for _ in range(16))
GP_BI_BWD_X: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(_GP_INDICES[n, j]) for n in range(16)) for j in range(16)
)
GP_SI_BWD_X: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(_GP_SIGNS[n, j]) for n in range(16)) for j in range(16)
)
# bwd_y: each output k has 16 contributions ``(i_n, j_n, s_n)`` from the inverse map.
GP_AI_BWD_Y, GP_BI_BWD_Y, GP_SI_BWD_Y = _build_gp_inverse()
