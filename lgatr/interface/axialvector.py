"""Embedding and extracting axial vectors into multivectors."""

import torch


def embed_axialvector(axialvector: torch.Tensor) -> torch.Tensor:
    """Embeds axial vectors in multivectors.

    Parameters
    ----------
    axialvector : torch.Tensor
        Axial vector with shape (..., 4)

    Returns
    -------
    multivector : torch.Tensor
        Embedding into multivector with shape (..., 16).
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = axialvector.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=axialvector.dtype, device=axialvector.device)

    # Embedding into Lorentz vectors
    multivector[..., 11:15] = axialvector.flip(-1)

    return multivector


def extract_axialvector(multivector: torch.Tensor) -> torch.Tensor:
    """Given a multivector, extract a axial vector.

    Parameters
    ----------
    multivector : torch.Tensor
        Multivector with shape (..., 16).

    Returns
    -------
    axialvector : torch.Tensor
        Axial vector with shape (..., 4)
    """

    axialvector = multivector[..., 11:15].flip(-1)

    return axialvector
