"""Embedding and extracting scalars into multivectors."""

import torch


def embed_scalar(scalars: torch.Tensor) -> torch.Tensor:
    """Embeds a scalar tensor into multivectors.

    Parameters
    ----------
    scalars: torch.Tensor
        Scalar inputs with shape (..., 1).

    Returns
    -------
    multivectors: torch.Tensor
        Multivector outputs with shape (..., 16).
        ``multivectors[..., [0]]`` is the same as ``scalars``. The other components are zero.
    """
    assert scalars.shape[-1] == 1
    non_scalar_shape = list(scalars.shape[:-1]) + [15]
    non_scalar_components = torch.zeros(
        non_scalar_shape, device=scalars.device, dtype=scalars.dtype
    )
    embedding = torch.cat((scalars, non_scalar_components), dim=-1)

    return embedding


def extract_scalar(multivectors: torch.Tensor) -> torch.Tensor:
    """Extracts scalar components from multivectors.

    Parameters
    ----------
    multivectors: torch.Tensor
        Multivector inputs with shape (..., 16).

    Returns
    -------
    scalars: torch.Tensor
        Scalar component of multivectors with shape (..., 1).
    """

    return multivectors[..., [0]]
