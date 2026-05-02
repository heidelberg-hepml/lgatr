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

    # F.pad(x, (11, 1)) zero-pads 11 entries on the left and 1 on the right of the last dim,
    # placing the (flipped) input at indices 11..15 (the trivector slots) with zeros elsewhere.
    return torch.nn.functional.pad(axialvector.flip(-1), (11, 1))


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
