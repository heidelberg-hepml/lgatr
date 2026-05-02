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

    # F.pad(x, (1, 11)) zero-pads 1 entry on the left and 11 on the right of the last dim,
    # placing the input at indices 1..5 (the Lorentz-vector slots) with zeros elsewhere.
    return torch.nn.functional.pad(vector, (1, 11))


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
