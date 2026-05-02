"""Gated nonlinearities on multivectors."""

import torch


def gated_relu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated ReLU nonlinearity.

    Computes ``ReLU(gates) * x`` (broadcasting ``gates`` across the multivector components).

    Parameters
    ----------
    x
        Multivector input of shape ``(..., 16)``.
    gates
        Pin-invariant gates of shape ``(..., 1)``.

    Returns
    -------
    outputs
        Result of shape ``(..., 16)``.
    """

    weights = torch.nn.functional.relu(gates)
    outputs = weights * x
    return outputs


def gated_sigmoid(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated sigmoid nonlinearity.

    Computes ``sigmoid(gates) * x`` (broadcasting ``gates`` across the multivector components).

    Parameters
    ----------
    x
        Multivector input of shape ``(..., 16)``.
    gates
        Pin-invariant gates of shape ``(..., 1)``.

    Returns
    -------
    outputs
        Result of shape ``(..., 16)``.
    """

    weights = torch.nn.functional.sigmoid(gates)
    outputs = weights * x
    return outputs


def gated_gelu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated GeLU nonlinearity (without division).

    Computes ``GeLU(gates) * x`` (broadcasting ``gates`` across the multivector components).

    References
    ----------
    Dan Hendrycks, Kevin Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415

    Parameters
    ----------
    x
        Multivector input of shape ``(..., 16)``.
    gates
        Pin-invariant gates of shape ``(..., 1)``.

    Returns
    -------
    outputs
        Result of shape ``(..., 16)``.
    """

    weights = torch.nn.functional.gelu(gates, approximate="tanh")
    outputs = weights * x
    return outputs


def gated_silu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated SiLU (Swish) nonlinearity (without division).

    Computes ``Swish(gates) * x`` (broadcasting ``gates`` across the multivector components).

    References
    ----------
    Stefan Elfwing, Eiji Uchibe, Kenji Doya, "Sigmoid-Weighted Linear Units for Neural Network
    Function Approximation in Reinforcement Learning", arXiv:1702.03118

    Parameters
    ----------
    x
        Multivector input of shape ``(..., 16)``.
    gates
        Pin-invariant gates of shape ``(..., 1)``.

    Returns
    -------
    outputs
        Result of shape ``(..., 16)``.
    """

    weights = torch.nn.functional.silu(gates)
    outputs = weights * x
    return outputs
