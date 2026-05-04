"""Triton kernels and autograd integration for :func:`equi_linear`."""

import torch
import triton
import triton.language as tl

from ..config import gatr_config
from ..linear import _compute_pin_equi_linear_basis
from ._indices import (
    GRADE_COMPS,
    GRADE_DG_BWD,
    GRADE_DG_FWD,
    GRADE_DUAL_PERM,
    GRADE_DUAL_SIGN_BWD,
    GRADE_DUAL_SIGN_FWD,
)
from ._utils import DTYPE_TAGS


def _autotune_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_N": 32, "BLOCK_OUT": 32, "BLOCK_IN": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_OUT": 32, "BLOCK_IN": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_OUT": 32, "BLOCK_IN": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_OUT": 32, "BLOCK_IN": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_OUT": 64, "BLOCK_IN": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_OUT": 64, "BLOCK_IN": 32}, num_warps=8, num_stages=3),
    ]


@triton.autotune(
    configs=_autotune_configs(),
    key=["IN_C", "OUT_C", "G_SIZE", "SUBGROUP", "DTYPE_TAG"],
)
@triton.jit
def _equi_linear_grade_kernel(
    x_ptr,
    c_ptr,
    out_ptr,
    N,
    sx_n,
    sx_in,
    sx_d,
    sc_o,
    sc_in,
    sc_b,
    so_n,
    so_o,
    so_d,
    DTYPE_TAG,
    G: tl.constexpr,
    DG: tl.constexpr,
    G_SIZE: tl.constexpr,
    COMPS: tl.constexpr,
    DUALS: tl.constexpr,
    SIGNS: tl.constexpr,
    SUBGROUP: tl.constexpr,
    IN_C: tl.constexpr,
    OUT_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    """One launch processes the d_g components of grade G; coeff slice G is shared across them."""
    pid_n = tl.program_id(0)
    pid_o = tl.program_id(1)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    o_offs = pid_o * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    n_mask = n_offs < N
    o_mask = o_offs < OUT_C

    for di in tl.static_range(G_SIZE):
        i = COMPS[di]
        dp = DUALS[di]
        ds = SIGNS[di]
        acc = tl.zeros((BLOCK_N, BLOCK_OUT), dtype=tl.float32)
        for in_off in range(0, IN_C, BLOCK_IN):
            in_offs = in_off + tl.arange(0, BLOCK_IN)
            in_mask = in_offs < IN_C
            nm_in = n_mask[:, None] & in_mask[None, :]
            cm_in = in_mask[:, None] & o_mask[None, :]
            x_p = x_ptr + n_offs[:, None] * sx_n + in_offs[None, :] * sx_in + i * sx_d
            x_i = tl.load(x_p, mask=nm_in, other=0.0).to(tl.float32)
            c_p = c_ptr + in_offs[:, None] * sc_in + o_offs[None, :] * sc_o + G * sc_b
            c_g = tl.load(c_p, mask=cm_in, other=0.0).to(tl.float32)
            acc = tl.dot(x_i, c_g, acc=acc, out_dtype=tl.float32, input_precision="ieee")
            if SUBGROUP:
                xdp_p = x_ptr + n_offs[:, None] * sx_n + in_offs[None, :] * sx_in + dp * sx_d
                x_dp = tl.load(xdp_p, mask=nm_in, other=0.0).to(tl.float32)
                cdb_p = c_ptr + in_offs[:, None] * sc_in + o_offs[None, :] * sc_o + DG * sc_b
                c_db = tl.load(cdb_p, mask=cm_in, other=0.0).to(tl.float32) * ds
                acc = tl.dot(x_dp, c_db, acc=acc, out_dtype=tl.float32, input_precision="ieee")
        out_p = out_ptr + n_offs[:, None] * so_n + o_offs[None, :] * so_o + i * so_d
        tl.store(out_p, acc.to(out_ptr.dtype.element_ty), mask=n_mask[:, None] & o_mask[None, :])


def _launch_grade_grouped(
    x_flat: torch.Tensor,
    coeffs: torch.Tensor,
    out: torch.Tensor,
    in_c: int,
    out_c: int,
    subgroup: bool,
    signs_table: tuple[tuple[int, ...], ...],
    dg_for_grade: tuple[int, ...],
) -> None:
    n = x_flat.shape[0]

    def grid(meta):
        return (
            triton.cdiv(n, meta["BLOCK_N"]),
            triton.cdiv(out_c, meta["BLOCK_OUT"]),
        )

    dtype_tag = DTYPE_TAGS[x_flat.dtype]
    for g in range(5):
        comps = GRADE_COMPS[g]
        _equi_linear_grade_kernel[grid](
            x_flat,
            coeffs,
            out,
            n,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            coeffs.stride(0),
            coeffs.stride(1),
            coeffs.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            dtype_tag,
            G=g,
            DG=dg_for_grade[g],
            G_SIZE=len(comps),
            COMPS=comps,
            DUALS=GRADE_DUAL_PERM[g],
            SIGNS=signs_table[g],
            SUBGROUP=subgroup,
            IN_C=in_c,
            OUT_C=out_c,
        )


def _equi_linear_fwd(x: torch.Tensor, coeffs: torch.Tensor, subgroup: bool) -> torch.Tensor:
    leading = x.shape[:-2]
    in_c = x.shape[-2]
    out_c = coeffs.shape[0]
    x_flat = x.reshape(-1, in_c, 16).contiguous()
    out = torch.empty(x_flat.shape[0], out_c, 16, device=x.device, dtype=x.dtype)
    _launch_grade_grouped(
        x_flat, coeffs, out, in_c, out_c, subgroup, GRADE_DUAL_SIGN_FWD, GRADE_DG_FWD
    )
    return out.reshape(*leading, out_c, 16)


def _equi_linear_bwd_x(
    grad_out: torch.Tensor, coeffs: torch.Tensor, subgroup: bool
) -> torch.Tensor:
    """Reuse the forward kernel with coeffs.transpose(0,1) (a view) and BWD dual signs/DG."""
    leading = grad_out.shape[:-2]
    out_c = grad_out.shape[-2]
    in_c = coeffs.shape[1]
    g_flat = grad_out.reshape(-1, out_c, 16).contiguous()
    coeffs_t = coeffs.transpose(0, 1)
    grad_x = torch.empty(g_flat.shape[0], in_c, 16, device=grad_out.device, dtype=grad_out.dtype)
    _launch_grade_grouped(
        g_flat, coeffs_t, grad_x, out_c, in_c, subgroup, GRADE_DUAL_SIGN_BWD, GRADE_DG_BWD
    )
    return grad_x.reshape(*leading, in_c, 16)


def _equi_linear_bwd_coeffs(
    grad_out: torch.Tensor, x: torch.Tensor, subgroup: bool
) -> torch.Tensor:
    """grad_coeffs via the dense basis; output is small (out_c × in_c × NB ≤ 10240)."""
    basis = _compute_pin_equi_linear_basis(subgroup, device=x.device, dtype=x.dtype)
    g_flat = grad_out.reshape(-1, grad_out.shape[-2], grad_out.shape[-1])
    x_flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    return torch.einsum("byi,aij,bxj->yxa", g_flat, basis, x_flat)


class _EquiLinearTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeffs, subgroup):
        ctx.save_for_backward(x, coeffs)
        ctx.subgroup = subgroup
        return _equi_linear_fwd(x, coeffs, subgroup)

    @staticmethod
    def backward(ctx, grad_out):
        x, coeffs = ctx.saved_tensors
        grad_x = grad_coeffs = None
        if ctx.needs_input_grad[0]:
            grad_x = _equi_linear_bwd_x(grad_out.contiguous(), coeffs, ctx.subgroup)
        if ctx.needs_input_grad[1]:
            grad_coeffs = _equi_linear_bwd_coeffs(grad_out, x, ctx.subgroup)
        return grad_x, grad_coeffs, None


@torch.compiler.disable()
def equi_linear_triton(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated :func:`lgatr.primitives.equi_linear`."""
    return _EquiLinearTriton.apply(x, coeffs, gatr_config.use_fully_connected_subgroup)
