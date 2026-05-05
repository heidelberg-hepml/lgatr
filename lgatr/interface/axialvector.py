"""Embedding and extracting axial vectors into multivectors."""

import torch


def embed_axialvector(axialvectors: torch.Tensor) -> torch.Tensor:
    """Embed axial vectors into multivectors.

    Parameters
    ----------
    axialvectors
        Axial vectors of shape ``(..., 4)``.

    Returns
    -------
    multivectors
        Multivectors of shape ``(..., 16)``.
    """

    # F.pad(x, (11, 1)) zero-pads 11 entries on the left and 1 on the right of the last dim,
    # placing the (flipped) input at indices 11..15 (the axialvector slots) with zeros elsewhere.
    return torch.nn.functional.pad(axialvectors.flip(-1), (11, 1))


def extract_axialvector(multivectors: torch.Tensor) -> torch.Tensor:
    """Extract axial vectors from multivectors.

    Parameters
    ----------
    multivectors
        Multivectors of shape ``(..., 16)``.

    Returns
    -------
    axialvectors
        Axial vectors of shape ``(..., 4)``.
    """

    axialvectors = multivectors[..., 11:15].flip(-1)

    return axialvectors
