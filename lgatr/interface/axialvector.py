"""Embedding and extracting axial vectors into multivectors."""

import torch


def embed_axialvector(axialvector: torch.Tensor) -> torch.Tensor:
    """Embed an axial vector into a multivector.

    Parameters
    ----------
    axialvector
        Axial vector of shape ``(..., 4)``.

    Returns
    -------
    multivector
        Multivector embedding of shape ``(..., 16)``.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = axialvector.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=axialvector.dtype, device=axialvector.device)

    # Embedding into Lorentz vectors
    multivector[..., 11:15] = axialvector.flip(-1)

    return multivector


def extract_axialvector(multivector: torch.Tensor) -> torch.Tensor:
    """Extract an axial vector from a multivector.

    Parameters
    ----------
    multivector
        Multivector of shape ``(..., 16)``.

    Returns
    -------
    axialvector
        Axial vector of shape ``(..., 4)``.
    """

    axialvector = multivector[..., 11:15].flip(-1)

    return axialvector
