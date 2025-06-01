"""Embedding and extracting vectors into multivectors."""
import torch


def embed_vector(vector: torch.Tensor) -> torch.Tensor:
    """Embeds Lorentz vectors in multivectors.

    Parameters
    ----------
    vector : torch.Tensor
        Lorentz vector with shape (..., 4)

    Returns
    -------
    multivector : torch.Tensor
        Embedding into multivector with shape (..., 16).
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = vector.shape[:-1]
    multivector = torch.zeros(
        *batch_shape, 16, dtype=vector.dtype, device=vector.device
    )

    # Embedding into Lorentz vectors
    multivector[..., 1:5] = vector

    return multivector


def extract_vector(multivector: torch.Tensor) -> torch.Tensor:
    """Given a multivector, extract a Lorentz vector.

    Parameters
    ----------
    multivector : torch.Tensor
        Multivector with shape (..., 16).

    Returns
    -------
    vector : torch.Tensor
        Lorentz vector with shape (..., 4)
    """

    vector = multivector[..., 1:5]

    return vector
