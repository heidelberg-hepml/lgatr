"""Embedding and extracting pseudoscalars into multivectors."""

import torch


def embed_pseudoscalar(pseudoscalars: torch.Tensor) -> torch.Tensor:
    """Embed pseudoscalars into multivectors.

    Parameters
    ----------
    pseudoscalars
        Pseudoscalars of shape ``(..., 1)``.

    Returns
    -------
    multivectors
        Multivectors of shape ``(..., 16)``. ``multivectors[..., [15]]`` is the same as
        ``pseudoscalars``; the other components are zero.
    """
    assert pseudoscalars.shape[-1] == 1
    return torch.nn.functional.pad(pseudoscalars, (15, 0))


def extract_pseudoscalar(multivectors: torch.Tensor) -> torch.Tensor:
    """Extract pseudoscalars from multivectors.

    Parameters
    ----------
    multivectors
        Multivectors of shape ``(..., 16)``.

    Returns
    -------
    pseudoscalars
        Pseudoscalars of shape ``(..., 1)``.
    """

    return multivectors[..., [15]]
