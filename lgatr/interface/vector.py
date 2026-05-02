"""Embedding and extracting vectors into multivectors."""

import torch


def embed_vector(vector: torch.Tensor) -> torch.Tensor:
    """Embed a Lorentz vector into a multivector.

    Parameters
    ----------
    vector
        Lorentz vector of shape ``(..., 4)``.

    Returns
    -------
    multivector
        Multivector embedding of shape ``(..., 16)``.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = vector.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=vector.dtype, device=vector.device)

    # Embedding into Lorentz vectors
    multivector[..., 1:5] = vector

    return multivector


def extract_vector(multivector: torch.Tensor) -> torch.Tensor:
    """Extract a Lorentz vector from a multivector.

    Parameters
    ----------
    multivector
        Multivector of shape ``(..., 16)``.

    Returns
    -------
    vector
        Lorentz vector of shape ``(..., 4)``.
    """

    vector = multivector[..., 1:5]

    return vector
