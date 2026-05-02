"""Embedding and extracting scalars into multivectors."""

import torch


def embed_scalar(scalars: torch.Tensor) -> torch.Tensor:
    """Embed a scalar tensor into a multivector.

    Parameters
    ----------
    scalars
        Scalar inputs of shape ``(..., 1)``.

    Returns
    -------
    multivectors
        Multivector outputs of shape ``(..., 16)``. ``multivectors[..., [0]]`` is the same as
        ``scalars``; the other components are zero.
    """
    assert scalars.shape[-1] == 1
    return torch.nn.functional.pad(scalars, (0, 15))


def extract_scalar(multivectors: torch.Tensor) -> torch.Tensor:
    """Extract the scalar component from a multivector.

    Parameters
    ----------
    multivectors
        Multivector inputs of shape ``(..., 16)``.

    Returns
    -------
    scalars
        Scalar component of shape ``(..., 1)``.
    """

    return multivectors[..., [0]]
