"""Random SO(1, 3) transformations for equivariance tests.

Local reimplementation of ``lloca.utils.rand_transforms.rand_lorentz`` to keep the test layer
independent of ``lloca`` (whose top-level ``__init__`` eagerly imports ``torch_geometric``).
"""

import math

import torch


def rand_lorentz(
    shape: torch.Size | tuple[int, ...],
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample random SO(1, 3) (rotation × boost) transformations.

    Parameters
    ----------
    shape
        Batch shape of the transformation matrices.
    std_eta
        Standard deviation of the rapidity used for the boost.
    n_max_std_eta
        Truncation in units of ``std_eta`` for the rapidity.
    device
        Device for the output tensor.
    dtype
        Dtype for the output tensor.

    Returns
    -------
    trafo
        Lorentz transformation matrices of shape ``(*shape, 4, 4)``.
    """
    boost = _rand_boost(shape, std_eta, n_max_std_eta, device=device, dtype=dtype)
    rotation = _rand_rotation(shape, device=device, dtype=dtype)
    return torch.einsum("...ij,...jk->...ik", rotation, boost)


def _rand_boost(
    shape: torch.Size | tuple[int, ...],
    std_eta: float,
    n_max_std_eta: float,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    eta = torch.randn((*shape, 3), device=device, dtype=dtype)
    beta = (eta * std_eta).clamp(min=-std_eta * n_max_std_eta, max=std_eta * n_max_std_eta)
    beta2 = (beta**2).sum(dim=-1, keepdim=True)
    gamma = 1 / (1 - beta2).clamp(min=1e-10).sqrt()
    fourmomenta = torch.cat([gamma, beta], dim=-1)
    return _restframe_boost(fourmomenta)


def _restframe_boost(fourmomenta: torch.Tensor) -> torch.Tensor:
    t0 = fourmomenta.narrow(-1, 0, 1)
    beta = fourmomenta[..., 1:] / t0.clamp_min(1e-10)
    beta2 = beta.square().sum(dim=-1, keepdim=True)
    gamma = torch.rsqrt(torch.clamp_min(1 - beta2, min=1e-10))
    boost = -gamma * beta

    eye3 = torch.eye(3, device=fourmomenta.device, dtype=fourmomenta.dtype)
    eye3 = eye3.reshape(*(1,) * (fourmomenta.dim() - 1), 3, 3).expand(*fourmomenta.shape[:-1], 3, 3)
    scale = (gamma - 1) / torch.clamp_min(beta2, min=1e-10)
    outer = beta.unsqueeze(-1) * beta.unsqueeze(-2)
    rot = eye3 + scale.unsqueeze(-1) * outer

    row0 = torch.cat((gamma, boost), dim=-1)
    lower = torch.cat((boost.unsqueeze(-1), rot), dim=-1)
    return torch.cat((row0.unsqueeze(-2), lower), dim=-2)


def _rand_rotation(
    shape: torch.Size | tuple[int, ...],
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Uniform random rotation via Shoemake's quaternion sampling.
    u = torch.rand((*shape, 3), device=device, dtype=dtype)
    q1 = torch.sqrt(1 - u[..., 0]) * torch.sin(2 * math.pi * u[..., 1])
    q2 = torch.sqrt(1 - u[..., 0]) * torch.cos(2 * math.pi * u[..., 1])
    q3 = torch.sqrt(u[..., 0]) * torch.sin(2 * math.pi * u[..., 2])
    q0 = torch.sqrt(u[..., 0]) * torch.cos(2 * math.pi * u[..., 2])

    R1 = torch.stack(
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)], dim=-1
    )
    R2 = torch.stack(
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)], dim=-1
    )
    R3 = torch.stack(
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)], dim=-1
    )
    R = torch.stack([R1, R2, R3], dim=-2)

    trafo = torch.eye(4, device=device, dtype=dtype).expand(*shape, 4, 4).clone()
    trafo[..., 1:, 1:] = R
    return trafo
