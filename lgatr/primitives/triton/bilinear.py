"""Triton kernels and autograd integration for :func:`geometric_product`."""

import math

import torch
import triton
import triton.language as tl

from ._indices import (
    GP_AI_BWD_X,
    GP_AI_BWD_Y,
    GP_AI_FWD,
    GP_BI_BWD_X,
    GP_BI_BWD_Y,
    GP_BI_FWD,
    GP_SI_BWD_X,
    GP_SI_BWD_Y,
    GP_SI_FWD,
)
from ._utils import DTYPE_TAGS


def _autotune_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_BC": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_BC": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_BC": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_BC": 1024}, num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=_autotune_configs(), key=["BC", "DTYPE_TAG"])
@triton.jit
def _gp_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    BC,
    sa_bc,
    sa_d,
    sb_bc,
    sb_d,
    so_bc,
    so_d,
    DTYPE_TAG,
    AI: tl.constexpr,
    BI: tl.constexpr,
    SI: tl.constexpr,
    BLOCK_BC: tl.constexpr,
):
    """Half-unrolled fused 16-output GP kernel: out[k] = sum_n SI[k,n] * a[AI[k,n]] * b[BI[k,n]]."""
    pid = tl.program_id(0)
    bc = pid * BLOCK_BC + tl.arange(0, BLOCK_BC)
    m = bc < BC
    a_tup = (
        tl.load(a_ptr + bc * sa_bc + 0 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 1 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 2 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 3 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 4 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 5 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 6 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 7 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 8 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 9 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 10 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 11 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 12 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 13 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 14 * sa_d, mask=m, other=0.0).to(tl.float32),
        tl.load(a_ptr + bc * sa_bc + 15 * sa_d, mask=m, other=0.0).to(tl.float32),
    )
    b_tup = (
        tl.load(b_ptr + bc * sb_bc + 0 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 1 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 2 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 3 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 4 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 5 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 6 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 7 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 8 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 9 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 10 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 11 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 12 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 13 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 14 * sb_d, mask=m, other=0.0).to(tl.float32),
        tl.load(b_ptr + bc * sb_bc + 15 * sb_d, mask=m, other=0.0).to(tl.float32),
    )
    for k in tl.static_range(16):
        acc = tl.zeros((BLOCK_BC,), dtype=tl.float32)
        for n in tl.static_range(16):
            acc = acc + SI[k][n] * a_tup[AI[k][n]] * b_tup[BI[k][n]]
        tl.store(
            out_ptr + bc * so_bc + k * so_d,
            acc.to(out_ptr.dtype.element_ty),
            mask=m,
        )


def _launch(
    a_flat: torch.Tensor,
    b_flat: torch.Tensor,
    out: torch.Tensor,
    ai: tuple[tuple[int, ...], ...],
    bi: tuple[tuple[int, ...], ...],
    si: tuple[tuple[int, ...], ...],
) -> None:
    bc = a_flat.shape[0]

    def grid(meta):
        return (triton.cdiv(bc, meta["BLOCK_BC"]),)

    _gp_kernel[grid](
        a_flat,
        b_flat,
        out,
        bc,
        a_flat.stride(0),
        a_flat.stride(1),
        b_flat.stride(0),
        b_flat.stride(1),
        out.stride(0),
        out.stride(1),
        DTYPE_TAGS[a_flat.dtype],
        AI=ai,
        BI=bi,
        SI=si,
    )


def _broadcast_and_flatten(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    out_shape = torch.broadcast_shapes(x.shape, y.shape)

    def _prep(t: torch.Tensor) -> torch.Tensor:
        v = t if t.shape == out_shape else t.broadcast_to(out_shape)
        return v.contiguous()

    x_b, y_b = _prep(x), _prep(y)
    bc = math.prod(out_shape[:-1])
    return x_b.reshape(bc, 16), y_b.reshape(bc, 16), out_shape


def _gp_fwd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat, y_flat, out_shape = _broadcast_and_flatten(x, y)
    out = torch.empty_like(x_flat)
    _launch(x_flat, y_flat, out, GP_AI_FWD, GP_BI_FWD, GP_SI_FWD)
    return out.reshape(*out_shape)


def _gp_bwd_x(grad_out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    g_flat, y_flat, out_shape = _broadcast_and_flatten(grad_out, y)
    out = torch.empty_like(g_flat)
    _launch(g_flat, y_flat, out, GP_AI_BWD_X, GP_BI_BWD_X, GP_SI_BWD_X)
    return out.reshape(*out_shape)


def _gp_bwd_y(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    g_flat, x_flat, out_shape = _broadcast_and_flatten(grad_out, x)
    out = torch.empty_like(g_flat)
    _launch(g_flat, x_flat, out, GP_AI_BWD_Y, GP_BI_BWD_Y, GP_SI_BWD_Y)
    return out.reshape(*out_shape)


def _reduce_to_input_shape(grad: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Sum-reduce a broadcast gradient back to the original tensor shape."""
    if grad.shape == original_shape:
        return grad
    extra = grad.dim() - len(original_shape)
    if extra > 0:
        grad = grad.sum(dim=tuple(range(extra)))
    for d, sz in enumerate(original_shape):
        if sz == 1 and grad.shape[d] != 1:
            grad = grad.sum(dim=d, keepdim=True)
    return grad


class _GPTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return _gp_fwd(x, y)

    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None
        go = grad_out.contiguous()
        if ctx.needs_input_grad[0]:
            grad_x = _reduce_to_input_shape(_gp_bwd_x(go, y), x.shape)
        if ctx.needs_input_grad[1]:
            grad_y = _reduce_to_input_shape(_gp_bwd_y(go, x), y.shape)
        return grad_x, grad_y


@torch.compiler.disable()
def geometric_product_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated :func:`lgatr.primitives.geometric_product`."""
    return _GPTriton.apply(x, y)
