"""Geometric product."""

from functools import lru_cache
from pathlib import Path

import torch

from ..utils.autocast import minimum_autocast_precision
from .config import PrimitivesConfig
from .linear import DEFAULT_DEVICE, DEFAULT_DTYPE

# Module-level constants loaded once at import. Lru_cache helpers below only do `.to(...)`
# casts on these, keeping them traceable under torch.compile(fullgraph=True).
_GP = (
    torch.load(Path(__file__).parent.resolve() / "geometric_product.pt", weights_only=True)
    .to(DEFAULT_DTYPE)
    .to_dense()
)
# Each row gp[i, j, :] has exactly one nonzero (a +/-1 entry); store its column and sign.
_GP_INDICES = _GP.abs().argmax(dim=-1)
_GP_SIGNS = torch.gather(_GP, -1, _GP_INDICES.unsqueeze(-1)).squeeze(-1)


@lru_cache
def _load_geometric_product_tensor(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    # Geometric-product tensor of shape (16, 16, 16), cast to (device, dtype).
    return _GP.to(device=device, dtype=dtype)


@lru_cache
def _compute_sparse_gp_indices(
    device: torch.device = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    # (indices, signs) of shape (16, 16) each, cast to (device, dtype).
    return _GP_INDICES.to(device=device), _GP_SIGNS.to(device=device, dtype=dtype)


def _geometric_product_dense(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Equation: out[..., i] = sum_{j, k} gp[i, j, k] * x[..., j] * y[..., k]
    gp = _load_geometric_product_tensor(device=x.device, dtype=x.dtype)
    # Build the (..., 16, 16) outer product x_j * y_k, then matmul against the flattened gp[i, j*k]
    outer = x.unsqueeze(-1) * y.unsqueeze(-2)
    return outer.flatten(-2, -1) @ gp.flatten(1, 2).T


def _unbroadcast(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    # Undo broadcasting: sum the batch dims that were expanded (the trailing 16 never is).
    if grad.shape == shape:
        return grad
    extra = grad.dim() - len(shape)
    if extra:
        grad = grad.sum(dim=tuple(range(extra)))
    dims = [i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1]
    return grad.sum(dim=dims, keepdim=True) if dims else grad


class _GeometricProductSparse(torch.autograd.Function):
    # out[..., i] = sum_j signs[i, j] * x[..., j] * y[..., indices[i, j]]. Bilinear, so the
    # gradients are the same sparse contraction; saving only (x, y) keeps this lighter than dense.
    # The setup_context style plus generate_vmap_rule keeps torch.func transforms (vmap, grad,
    # jacrev) working; forward-mode AD (jacfwd/jvp) would additionally need a jvp rule.

    generate_vmap_rule = True

    @staticmethod
    def forward(x, y):
        indices, signs = _compute_sparse_gp_indices(device=x.device, dtype=x.dtype)
        # Fused gather-multiply-sum rather than a batched (..., 16, 16) @ (..., 16, 1) matmul: the
        # matmul must materialize the 16x16 operand, while this fuses to a single kernel under
        # torch.compile (no 16x16 buffer), which is both faster and far lighter on GPU.
        return (signs * y[..., indices] * x.unsqueeze(-2)).sum(-1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensors
        indices, signs = _compute_sparse_gp_indices(device=x.device, dtype=x.dtype)
        # The sign folds below are in-place on freshly gathered tensors, which is safe under
        # double backward (no other node saves them).
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            # grad_x[..., j] = sum_i grad_out[..., i] * signs[i, j] * y[..., indices[i, j]]
            m = signs * y[..., indices]
            grad_x = _unbroadcast((grad_out.unsqueeze(-1) * m).sum(-2), x.shape)
            del m  # free the (..., 16, 16) temp before grad_y allocates its own
        if ctx.needs_input_grad[1]:
            # grad_y[..., k] = sum_{i, j : indices[i, j] = k} grad_out[..., i] * signs[i, j] * x[..., j]
            p = grad_out.unsqueeze(-1) * x.unsqueeze(-2)
            p.mul_(signs)
            grad_y = grad_out.new_zeros(p.shape[:-2] + (16,))
            # index_add_ is CUDA-nondeterministic, but beats the deterministic gather+matmul
            # alternative by ~10% on CPU and ~5% on CUDA.
            grad_y.index_add_(-1, indices.reshape(-1), p.flatten(-2, -1))
            grad_y = _unbroadcast(grad_y, y.shape)
        return grad_x, grad_y


def _geometric_product_sparse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _GeometricProductSparse.apply(x, y)


@minimum_autocast_precision(torch.float32, output="high")
def geometric_product(
    x: torch.Tensor, y: torch.Tensor, *, config: PrimitivesConfig
) -> torch.Tensor:
    """Compute the geometric product ``f(x, y) = x * y``.

    Parameters
    ----------
    x
        First input multivector of shape ``(..., 16)``.
        Batch dimensions must be broadcastable between ``x`` and ``y``.
    y
        Second input multivector of shape ``(..., 16)``.
        Batch dimensions must be broadcastable between ``x`` and ``y``.
    config
        LGATr primitives configuration.

    Returns
    -------
    outputs
        Result of shape ``(..., 16)``. Batch dimensions are the broadcast of ``x`` and ``y``.
    """
    if config.sparse_gp:
        return _geometric_product_sparse(x, y)
    return _geometric_product_dense(x, y)
