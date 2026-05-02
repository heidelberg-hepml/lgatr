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
    non_scalar_shape = list(pseudoscalars.shape[:-1]) + [15]
    non_scalar_components = torch.zeros(
        non_scalar_shape, device=pseudoscalars.device, dtype=pseudoscalars.dtype
    )
    embedding = torch.cat((non_scalar_components, pseudoscalars), dim=-1)

    return embedding


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
