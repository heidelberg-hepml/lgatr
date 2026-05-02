"""Embedding and extracting pseudoscalars into multivectors."""

import torch


def embed_pseudoscalar(pseudoscalars: torch.Tensor) -> torch.Tensor:
    """Embed a pseudoscalar tensor into a multivector.

    Parameters
    ----------
    pseudoscalars
        Pseudoscalar inputs of shape ``(..., 1)``.

    Returns
    -------
    multivectors
        Multivector outputs of shape ``(..., 16)``. ``multivectors[..., [15]]`` is the same as
        ``pseudoscalars``; the other components are zero.
    """
    assert pseudoscalars.shape[-1] == 1
    return torch.nn.functional.pad(pseudoscalars, (15, 0))


def extract_pseudoscalar(multivectors: torch.Tensor) -> torch.Tensor:
    """Extract the pseudoscalar component from a multivector.

    Parameters
    ----------
    multivectors
        Multivector inputs of shape ``(..., 16)``.

    Returns
    -------
    pseudoscalars
        Pseudoscalar component of shape ``(..., 1)``.
    """

    return multivectors[..., [15]]
