"""Embedding and extracting vectors into multivectors."""

import torch


def embed_vector(vectors: torch.Tensor) -> torch.Tensor:
    """Embed Lorentz vectors into multivectors.

    Parameters
    ----------
    vectors
        Lorentz vectors of shape ``(..., 4)``.

    Returns
    -------
    multivectors
        Multivectors of shape ``(..., 16)``.
    """

    # F.pad(x, (1, 11)) zero-pads 1 entry on the left and 11 on the right of the last dim,
    # placing the input at indices 1..5 (the Lorentz-vector slots) with zeros elsewhere.
    return torch.nn.functional.pad(vectors, (1, 11))


def extract_vector(multivectors: torch.Tensor) -> torch.Tensor:
    """Extract Lorentz vectors from multivectors.

    Parameters
    ----------
    multivectors
        Multivectors of shape ``(..., 16)``.

    Returns
    -------
    vectors
        Lorentz vectors of shape ``(..., 4)``.
    """

    vectors = multivectors[..., 1:5]

    return vectors
