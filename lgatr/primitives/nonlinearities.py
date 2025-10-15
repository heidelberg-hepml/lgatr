"""Gated nonlinearities on multivectors."""
import torch


def gated_relu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated ReLU nonlinearity.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    ``ReLU(gates) * x``.

    Parameters
    ----------
    x : torch.Tensor
        Multivector input with shape (..., 16)
    gates : torch.Tensor
        Pin-invariant gates with shape (..., 1).

    Returns
    -------
    outputs : torch.Tensor
        Computes ReLU(gates) * x, with broadcasting along the last dimension.
        Result has shape (..., 16)
    """

    weights = torch.nn.functional.relu(gates)
    outputs = weights * x
    return outputs


def gated_sigmoid(x: torch.Tensor, gates: torch.Tensor):
    """Pin-equivariant gated sigmoid nonlinearity.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    ``sigmoid(gates) * x``.

    Parameters
    ----------
    x : torch.Tensor
        Multivector input with shape (..., 16)
    gates : torch.Tensor
        Pin-invariant gates with shape (..., 1).

    Returns
    -------
    outputs : torch.Tensor
        Computes Sigmoid(gates) * x, with broadcasting along the last dimension.
        Result has shape (..., 16)
    """

    weights = torch.nn.functional.sigmoid(gates)
    outputs = weights * x
    return outputs


def gated_gelu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated GeLU nonlinearity without division.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    ``GeLU(gates) * x``.

    References
    ----------
    Dan Hendrycks, Kevin Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415

    Parameters
    ----------
    x : torch.Tensor
        Multivector input with shape (..., 16)
    gates : torch.Tensor
        Pin-invariant gates with shape (..., 1).

    Returns
    -------
    outputs : torch.Tensor
        Computes GeLU(gates) * x, with broadcasting along the last dimension.
        Result has shape (..., 16)
    """

    weights = torch.nn.functional.gelu(gates, approximate="tanh")
    outputs = weights * x
    return outputs


def gated_silu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated SiLU or Swish nonlinearity without division.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    ``Swish(gates) * x``.

    References
    ----------
    Stefan Elfwing, Eiji Uchibe, Kenji Doya, "Sigmoid-Weighted Linear Units for Neural Network
    Function Approximation in Reinforcement Learning", arXiv:1702.03118

    Parameters
    ----------
    x : torch.Tensor
        Multivector input with shape (..., 16)
    gates : torch.Tensor
        Pin-invariant gates with shape (..., 1).

    Returns
    -------
    outputs : torch.Tensor
        Computes Swish(gates) * x, with broadcasting along the last dimension.
        Result has shape (..., 16)
    """

    weights = torch.nn.functional.silu(gates)
    outputs = weights * x
    return outputs
