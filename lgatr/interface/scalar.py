"""Embedding and extracting scalars into multivectors."""

import torch


def embed_scalar(scalars: torch.Tensor) -> torch.Tensor:
    """Embed scalars into multivectors.

    Parameters
    ----------
    scalars
        Scalars of shape ``(..., 1)``.

    Returns
    -------
    multivectors
        Multivectors of shape ``(..., 16)``. ``multivectors[..., [0]]`` is the same as
        ``scalars``; the other components are zero.
    """
    assert scalars.shape[-1] == 1
    return torch.nn.functional.pad(scalars, (0, 15))


def extract_scalar(multivectors: torch.Tensor) -> torch.Tensor:
    """Extract scalars from multivectors.

    Parameters
    ----------
    multivectors
        Multivectors of shape ``(..., 16)``.

    Returns
    -------
    scalars
        Scalars of shape ``(..., 1)``.
    """

    return multivectors[..., [0]]
