"""Linear operations on multivectors, in particular linear basis maps."""

import math
from functools import lru_cache
from pathlib import Path

import torch

from .config import PrimitivesConfig

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


# Module-level constants loaded once at import. lru_cache helpers below only do `.to(...)`
# casts on these, keeping them traceable under torch.compile(fullgraph=True).
def _load_basis(name: str) -> torch.Tensor:
    return torch.load(Path(__file__).parent.resolve() / name).to(DEFAULT_DTYPE).to_dense()


_BASIS_SUBGROUP = _load_basis("linear_basis_subgroup.pt")
_BASIS_FULL = _load_basis("linear_basis_full.pt")

# Subgroup dual basis: summing the 5 dual rows yields a (16, 16) sign-permutation
# (one nonzero per row). Sparse path replaces the 16-wide gather x[..., dual_perm]
# with per-grade slice + flip(-1); this assert pins the basis layout it relies on.
_DUAL = _BASIS_SUBGROUP[5:10].sum(dim=0)
_DUAL_PERM = _DUAL.abs().argmax(dim=-1)
_DUAL_SIGN = torch.gather(_DUAL, -1, _DUAL_PERM.unsqueeze(-1)).squeeze(-1)
assert torch.equal(_DUAL_PERM, torch.arange(16).flip(0)), "sparse linear: bad basis layout"


@lru_cache
def _compute_pin_equi_linear_basis(
    subgroup: bool = True,
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Lorentz-equivariant basis of shape (10, 16, 16) for the proper orthochronous subgroup,
    # or (5, 16, 16) for the full Pin group.
    src = _BASIS_SUBGROUP if subgroup else _BASIS_FULL
    return src.to(device=device, dtype=dtype)


@lru_cache
def _compute_grade_projection_mask(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Mask of shape (5, 16) selecting positions belonging to grade g in row g.
    mask = torch.zeros(5, 16, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    offset = 0
    for k in range(5):
        d = math.comb(4, k)
        mask[k, offset : offset + d] = 1
        offset += d
    return mask.to(device=device, dtype=dtype)


@lru_cache
def _compute_reversal(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Diagonal of shape (16,) for the multivector reversal (+/-1 entries).
    reversal_flat = torch.ones(16, device=device, dtype=dtype)
    reversal_flat[5:15] = -1
    return reversal_flat


@lru_cache
def _compute_grade_involution(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Diagonal of shape (16,) for the multivector grade involution (+/-1 entries).
    involution_flat = torch.ones(16, device=device, dtype=dtype)
    involution_flat[1:5] = -1
    involution_flat[11:15] = -1
    return involution_flat


@lru_cache
def _compute_dual_sign(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Per-position dual-basis signs of shape (16,), cast to (device, dtype).
    return _DUAL_SIGN.to(device=device, dtype=dtype)


def _equi_linear_dense(
    x: torch.Tensor, coeffs: torch.Tensor, *, config: PrimitivesConfig
) -> torch.Tensor:
    # Equation: out[..., y, i] = sum_{x, a, j} coeffs[y, x, a] * basis[a, i, j] * x[..., x, j]
    # basis is ~1% nonzero, so this path spends most of its FLOPs on zero entries; the sparse
    # path skips those at the cost of less optimized kernels (no fused BLAS GEMM).
    basis = _compute_pin_equi_linear_basis(config.subgroup, device=x.device, dtype=x.dtype)
    # Fold (coeffs, basis) into an effective (out_c, in_c, 16, 16) weight via one GEMM,
    # then contract that weight with x.
    weight = (coeffs @ basis.flatten(-2)).unflatten(-1, (16, 16))
    return torch.einsum("y x i j, ... x j -> ... y i", weight, x)


def _grade_flat(t: torch.Tensor, channels: int) -> torch.Tensor:
    # (..., channels, dim_g) -> (batch*dim_g, channels) so each grade contraction is a single
    # 2-D GEMM; the natural per-batch BMM is several times slower for small channel counts.
    return t.transpose(-1, -2).reshape(-1, channels)


def _grade_unflat(t: torch.Tensor, batch_shape: torch.Size, dim_g: int) -> torch.Tensor:
    # (batch*dim_g, channels) -> (..., channels, dim_g), inverse of _grade_flat.
    return t.unflatten(0, (*batch_shape, dim_g)).transpose(-1, -2)


def _paired_coeffs(coeffs: torch.Tensor, g: int) -> torch.Tensor:
    # Stack the grade-preserving weight of grade g and the Hodge-dual weight consuming grade g
    # into one (2*out_c, in_c) GEMM operand.
    return torch.cat((coeffs[..., g], coeffs[..., 5 + g]), dim=0)


class _EquiLinearSparse(torch.autograd.Function):
    # Per-grade GEMMs on contiguous slices of x: avoids materializing the (10, 16, 16) basis
    # and the multi-axis einsum. Saving only (x, coeffs) and recomputing the flattened grade
    # slices in backward holds 1x of x; autograd through the per-grade GEMMs holds 3x.
    #
    # Subgroup mode: basis indices 0..4 are grade-preserving, 5..9 are the Hodge dual mapping
    # grade g -> grade 4-g via a sign + position reversal. Sign and reversal act on the position
    # dim, which the GEMM treats as batch, so they commute with the GEMM and are applied to its
    # output. That lets the dual weight ride along with the grade-preserving one in a single
    # paired GEMM per grade: z_g[:, :out_c] is the direct term of output grade g, z_g[:, out_c:]
    # is the dual term feeding output grade 4-g.
    #
    # The setup_context style plus generate_vmap_rule keeps torch.func transforms (vmap, grad,
    # jacrev) working; forward-mode AD (jacfwd/jvp) would additionally need a jvp rule.

    generate_vmap_rule = True

    @staticmethod
    def forward(x, coeffs, subgroup):
        batch_shape = x.shape[:-2]
        in_c = x.shape[-2]
        out_c = coeffs.shape[0]

        # Contiguous grade slices of widths 1, 4, 6, 4, 1, flattened for 2-D GEMMs.
        x0 = _grade_flat(x[..., 0:1], in_c)
        x1 = _grade_flat(x[..., 1:5], in_c)
        x2 = _grade_flat(x[..., 5:11], in_c)
        x3 = _grade_flat(x[..., 11:15], in_c)
        x4 = _grade_flat(x[..., 15:16], in_c)

        if subgroup:
            z0 = torch.nn.functional.linear(x0, _paired_coeffs(coeffs, 0))
            z1 = torch.nn.functional.linear(x1, _paired_coeffs(coeffs, 1))
            z2 = torch.nn.functional.linear(x2, _paired_coeffs(coeffs, 2))
            z3 = torch.nn.functional.linear(x3, _paired_coeffs(coeffs, 3))
            z4 = torch.nn.functional.linear(x4, _paired_coeffs(coeffs, 4))

            def direct(z, dim_g):
                return _grade_unflat(z[:, :out_c], batch_shape, dim_g)

            def dual(z, dim_g):
                return _grade_unflat(z[:, out_c:], batch_shape, dim_g).flip(-1)

            s = _compute_dual_sign(device=x.device, dtype=z0.dtype)
            y0 = direct(z0, 1) + s[0:1] * dual(z4, 1)
            y1 = direct(z1, 4) + s[1:5] * dual(z3, 4)
            y2 = direct(z2, 6) + s[5:11] * dual(z2, 6)
            y3 = direct(z3, 4) + s[11:15] * dual(z1, 4)
            y4 = direct(z4, 1) + s[15:16] * dual(z0, 1)
        else:
            # Full Pin group: only the 5 grade-preserving basis elements.
            y0 = _grade_unflat(torch.nn.functional.linear(x0, coeffs[..., 0]), batch_shape, 1)
            y1 = _grade_unflat(torch.nn.functional.linear(x1, coeffs[..., 1]), batch_shape, 4)
            y2 = _grade_unflat(torch.nn.functional.linear(x2, coeffs[..., 2]), batch_shape, 6)
            y3 = _grade_unflat(torch.nn.functional.linear(x3, coeffs[..., 3]), batch_shape, 4)
            y4 = _grade_unflat(torch.nn.functional.linear(x4, coeffs[..., 4]), batch_shape, 1)

        return torch.cat((y0, y1, y2, y3, y4), dim=-1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, coeffs, subgroup = inputs
        ctx.save_for_backward(x, coeffs)
        ctx.subgroup = subgroup

    @staticmethod
    def backward(ctx, grad_out):
        x, coeffs = ctx.saved_tensors
        batch_shape = x.shape[:-2]
        in_c = x.shape[-2]
        out_c = coeffs.shape[0]
        # Under autocast the forward GEMMs emit low precision while (x, coeffs) were saved as-is.
        grad_out = grad_out.to(x.dtype)
        grad_x = grad_coeffs = None

        go0 = grad_out[..., 0:1]
        go1 = grad_out[..., 1:5]
        go2 = grad_out[..., 5:11]
        go3 = grad_out[..., 11:15]
        go4 = grad_out[..., 15:16]

        if ctx.subgroup:
            # Per grade g, pair the direct gradient with the dual gradient routed back from
            # output grade 4-g (the sign + reversal are their own transpose), matching the
            # paired forward weights.
            s = _compute_dual_sign(device=x.device, dtype=grad_out.dtype)
            g0 = torch.cat(
                (_grade_flat(go0, out_c), _grade_flat((s[15:16] * go4).flip(-1), out_c)), dim=-1
            )
            g1 = torch.cat(
                (_grade_flat(go1, out_c), _grade_flat((s[11:15] * go3).flip(-1), out_c)), dim=-1
            )
            g2 = torch.cat(
                (_grade_flat(go2, out_c), _grade_flat((s[5:11] * go2).flip(-1), out_c)), dim=-1
            )
            g3 = torch.cat(
                (_grade_flat(go3, out_c), _grade_flat((s[1:5] * go1).flip(-1), out_c)), dim=-1
            )
            g4 = torch.cat(
                (_grade_flat(go4, out_c), _grade_flat((s[0:1] * go0).flip(-1), out_c)), dim=-1
            )
            weights = (_paired_coeffs(coeffs, g) for g in range(5))
        else:
            g0 = _grade_flat(go0, out_c)
            g1 = _grade_flat(go1, out_c)
            g2 = _grade_flat(go2, out_c)
            g3 = _grade_flat(go3, out_c)
            g4 = _grade_flat(go4, out_c)
            weights = (coeffs[..., g] for g in range(5))

        if ctx.needs_input_grad[0]:
            W0, W1, W2, W3, W4 = (W.to(grad_out.dtype) for W in weights)
            grad_x = torch.cat(
                (
                    _grade_unflat(torch.matmul(g0, W0), batch_shape, 1),
                    _grade_unflat(torch.matmul(g1, W1), batch_shape, 4),
                    _grade_unflat(torch.matmul(g2, W2), batch_shape, 6),
                    _grade_unflat(torch.matmul(g3, W3), batch_shape, 4),
                    _grade_unflat(torch.matmul(g4, W4), batch_shape, 1),
                ),
                dim=-1,
            )

        if ctx.needs_input_grad[1]:
            gW0 = torch.matmul(g0.transpose(0, 1), _grade_flat(x[..., 0:1], in_c))
            gW1 = torch.matmul(g1.transpose(0, 1), _grade_flat(x[..., 1:5], in_c))
            gW2 = torch.matmul(g2.transpose(0, 1), _grade_flat(x[..., 5:11], in_c))
            gW3 = torch.matmul(g3.transpose(0, 1), _grade_flat(x[..., 11:15], in_c))
            gW4 = torch.matmul(g4.transpose(0, 1), _grade_flat(x[..., 15:16], in_c))
            if ctx.subgroup:
                parts = (
                    gW0[:out_c],
                    gW1[:out_c],
                    gW2[:out_c],
                    gW3[:out_c],
                    gW4[:out_c],
                    gW0[out_c:],
                    gW1[out_c:],
                    gW2[out_c:],
                    gW3[out_c:],
                    gW4[out_c:],
                )
            else:
                parts = (gW0, gW1, gW2, gW3, gW4)
            grad_coeffs = torch.stack(parts, dim=-1).to(coeffs.dtype)

        return grad_x, grad_coeffs, None


def _equi_linear_sparse(
    x: torch.Tensor, coeffs: torch.Tensor, *, config: PrimitivesConfig
) -> torch.Tensor:
    return _EquiLinearSparse.apply(x, coeffs, config.subgroup)


def equi_linear(x: torch.Tensor, coeffs: torch.Tensor, *, config: PrimitivesConfig) -> torch.Tensor:
    """Pin-equivariant linear map ``f(x) = sum_{a,j} coeffs_a W^a_ij x_j``.

    The :math:`W^a` are 5 or 10 pre-defined basis elements (see :func:`_compute_pin_equi_linear_basis`).

    Parameters
    ----------
    x
        Input multivector of shape ``(..., in_channels, 16)``.
    coeffs
        Coefficients for the basis elements of shape ``(out_channels, in_channels, num_basis_elements)``,
        where ``num_basis_elements`` is 10 (fully connected subgroup) or 5 (full Lorentz group).
    config
        LGATr primitives configuration.

    Returns
    -------
    outputs
        Result of shape ``(..., out_channels, 16)``.
    """
    if config.sparse_linear:
        return _equi_linear_sparse(x, coeffs, config=config)
    return _equi_linear_dense(x, coeffs, config=config)


def grade_project(x: torch.Tensor) -> torch.Tensor:
    """Project a multivector onto its individual grades.

    The result is a single tensor with a new grade dimension.

    Parameters
    ----------
    x
        Input multivector of shape ``(..., 16)``.

    Returns
    -------
    outputs
        Output multivector of shape ``(..., 5, 16)``. The second-to-last dimension indexes grades.
    """
    # Equivalent to ``einsum("g i j, ... j -> ... g i", basis[:5], x)``: the first five basis
    # elements are pure-diagonal grade projectors, so the (5, 16, 16) operand collapses to a
    # (5, 16) mask of their diagonals.
    mask = _compute_grade_projection_mask(device=x.device, dtype=x.dtype)
    return x.unsqueeze(-2) * mask


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Compute the reversal of a multivector.

    The reversal preserves the scalar, vector, and pseudoscalar components and flips the sign of
    the bivector and axialvector components.

    Parameters
    ----------
    x
        Input multivector of shape ``(..., 16)``.

    Returns
    -------
    outputs
        Output multivector of shape ``(..., 16)``.
    """
    return _compute_reversal(device=x.device, dtype=x.dtype) * x


def grade_involute(x: torch.Tensor) -> torch.Tensor:
    """Compute the grade involution of a multivector.

    The grade involution preserves the scalar, bivector, and pseudoscalar components and flips the
    sign of the vector and axialvector components.

    Parameters
    ----------
    x
        Input multivector of shape ``(..., 16)``.

    Returns
    -------
    outputs
        Output multivector of shape ``(..., 16)``.
    """

    return _compute_grade_involution(device=x.device, dtype=x.dtype) * x
