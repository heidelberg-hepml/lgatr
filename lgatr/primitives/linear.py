"""Linear operations on multivectors, in particular linear basis maps."""

import math
from functools import lru_cache
from pathlib import Path

import torch

from ..utils.autocast import minimum_autocast_precision
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


# Grade slice offsets and widths (1, 4, 6, 4, 1) in the 16-component multivector.
_GRADE_SLICES = ((0, 1), (1, 5), (5, 11), (11, 15), (15, 16))


def _component_flat(t: torch.Tensor, channels: int) -> torch.Tensor:
    # (..., channels, 16) -> (batch, 16, channels) with one transpose copy; the five grade
    # slices along dim -2 are then cheap row-block views, and each grade contraction is a
    # single 2-D GEMM (the natural per-batch BMM is several times slower for small channels).
    return t.transpose(-1, -2).reshape(-1, 16, channels)


def _grade_unflat(t: torch.Tensor, batch_shape: torch.Size, dim_g: int) -> torch.Tensor:
    # (batch*dim_g, channels) -> (..., channels, dim_g).
    return t.unflatten(0, (*batch_shape, dim_g)).transpose(-1, -2)


def _pair_coeffs(coeffs: torch.Tensor, subgroup: bool) -> torch.Tensor:
    # (out_c, in_c, 10) -> (5, 2*out_c, in_c): the grade-preserving weight of grade g stacked on
    # the Hodge-dual weight consuming grade g, one GEMM operand per grade. Built once per call
    # (a single cat) instead of five cats inside the autograd.Function, where it would also be
    # rebuilt in backward.
    cp = coeffs.permute(2, 0, 1)
    if subgroup:
        return torch.cat((cp[:5], cp[5:]), dim=1)
    return cp.contiguous()


class _EquiLinearSparse(torch.autograd.Function):
    # Per-grade GEMMs on grade slices of x: avoids materializing the (10, 16, 16) basis
    # and the multi-axis einsum. Takes the weights pre-paired as (5, 2*out_c, in_c) (see
    # _pair_coeffs), so forward and backward index grade weights as views instead of
    # re-assembling them with cats. Saving only (x, weights) and recomputing the flattened
    # grade slices in backward holds 1x of x; autograd through the per-grade GEMMs holds 3x.
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
    def forward(x, weights, subgroup):
        batch_shape = x.shape[:-2]
        in_c = x.shape[-2]
        out_c = weights.shape[-2] // 2 if subgroup else weights.shape[-2]

        xt = _component_flat(x, in_c)
        x0, x1, x2, x3, x4 = (xt[:, a:b].reshape(-1, in_c) for a, b in _GRADE_SLICES)

        z0 = torch.nn.functional.linear(x0, weights[0])
        z1 = torch.nn.functional.linear(x1, weights[1])
        z2 = torch.nn.functional.linear(x2, weights[2])
        z3 = torch.nn.functional.linear(x3, weights[3])
        z4 = torch.nn.functional.linear(x4, weights[4])

        if subgroup:

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
            y0 = _grade_unflat(z0, batch_shape, 1)
            y1 = _grade_unflat(z1, batch_shape, 4)
            y2 = _grade_unflat(z2, batch_shape, 6)
            y3 = _grade_unflat(z3, batch_shape, 4)
            y4 = _grade_unflat(z4, batch_shape, 1)

        return torch.cat((y0, y1, y2, y3, y4), dim=-1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weights, subgroup = inputs
        ctx.save_for_backward(x, weights)
        ctx.subgroup = subgroup

    @staticmethod
    def backward(ctx, grad_out):
        x, weights = ctx.saved_tensors
        batch_shape = x.shape[:-2]
        in_c = x.shape[-2]
        out_c = weights.shape[-2] // 2 if ctx.subgroup else weights.shape[-2]
        # Under naive_amp the forward GEMMs run in the autocast dtype while (x, weights) were
        # saved as-is, so grad_out comes back low precision and weights stays fp32. Reconcile both
        # streams to the saved x dtype; the casts are no-ops in the fp32 path (islands on or no amp).
        grad_out = grad_out.to(x.dtype)
        grad_x = grad_weights = None

        got = _component_flat(grad_out, out_c)

        def flat(a, b):
            return got[:, a:b].reshape(-1, out_c)

        if ctx.subgroup:
            # Per grade g, pair the direct gradient with the dual gradient routed back from
            # output grade 4-g (the sign + reversal are their own transpose), matching the
            # paired forward weights.
            s = _compute_dual_sign(device=x.device, dtype=grad_out.dtype)

            def dual_flat(a, b):
                return (s[a:b, None] * got[:, a:b]).flip(1).reshape(-1, out_c)

            g0 = torch.cat((flat(0, 1), dual_flat(15, 16)), dim=-1)
            g1 = torch.cat((flat(1, 5), dual_flat(11, 15)), dim=-1)
            g2 = torch.cat((flat(5, 11), dual_flat(5, 11)), dim=-1)
            g3 = torch.cat((flat(11, 15), dual_flat(1, 5)), dim=-1)
            g4 = torch.cat((flat(15, 16), dual_flat(0, 1)), dim=-1)
        else:
            g0, g1, g2, g3, g4 = (flat(a, b) for a, b in _GRADE_SLICES)

        if ctx.needs_input_grad[0]:
            W = weights.to(grad_out.dtype)
            grad_x = torch.cat(
                (
                    _grade_unflat(torch.matmul(g0, W[0]), batch_shape, 1),
                    _grade_unflat(torch.matmul(g1, W[1]), batch_shape, 4),
                    _grade_unflat(torch.matmul(g2, W[2]), batch_shape, 6),
                    _grade_unflat(torch.matmul(g3, W[3]), batch_shape, 4),
                    _grade_unflat(torch.matmul(g4, W[4]), batch_shape, 1),
                ),
                dim=-1,
            )

        if ctx.needs_input_grad[1]:
            xt = _component_flat(x, in_c)
            grad_weights = torch.stack(
                [
                    torch.matmul(g.transpose(0, 1), xt[:, a:b].reshape(-1, in_c))
                    for g, (a, b) in zip((g0, g1, g2, g3, g4), _GRADE_SLICES, strict=True)
                ],
                dim=0,
            ).to(weights.dtype)

        return grad_x, grad_weights, None


def _equi_linear_sparse(
    x: torch.Tensor, coeffs: torch.Tensor, *, config: PrimitivesConfig
) -> torch.Tensor:
    weights = _pair_coeffs(coeffs, config.subgroup)
    return _EquiLinearSparse.apply(x, weights, config.subgroup)


@minimum_autocast_precision(torch.float32, output="high")
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
