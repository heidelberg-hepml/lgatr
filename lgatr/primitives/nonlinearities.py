import torch


def gated_relu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated ReLU nonlinearity.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    ReLU(gates) * x.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Multivector input
    gates : torch.Tensor with shape (..., 1)
        Pin-invariant gates.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Computes ReLU(gates) * x, with broadcasting along the last dimension.
    """

    weights = torch.nn.functional.relu(gates)
    outputs = weights * x
    return outputs


def gated_sigmoid(x: torch.Tensor, gates: torch.Tensor):
    """Pin-equivariant gated sigmoid nonlinearity.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    sigmoid(gates) * x.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Multivector input
    gates : torch.Tensor with shape (..., 1)
        Pin-invariant gates.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Computes sigmoid(gates) * x, with broadcasting along the last dimension.
    """

    weights = torch.nn.functional.sigmoid(gates)
    outputs = weights * x
    return outputs


def gated_gelu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated GeLU nonlinearity without division.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    GeLU(gates) * x.

    References
    ----------
    Dan Hendrycks, Kevin Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Multivector input
    gates : torch.Tensor with shape (..., 1)
        Pin-invariant gates.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Computes GeLU(gates) * x, with broadcasting along the last dimension.
    """

    weights = torch.nn.functional.gelu(gates, approximate="tanh")
    outputs = weights * x
    return outputs
