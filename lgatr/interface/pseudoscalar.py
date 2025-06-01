"""Embedding and extracting pseudoscalars into multivectors."""
import torch


def embed_pseudoscalar(pseudoscalars: torch.Tensor) -> torch.Tensor:
    """Embeds a pseudoscalar tensor into multivectors.

    Parameters
    ----------
    pseudoscalars: torch.Tensor
        Scalar inputs with shape (..., 1).

    Returns
    -------
    multivectors: torch.Tensor
        Multivector outputs with shape (..., 16).
        ``multivectors[..., [15]]`` is the same as ``pseudoscalars``. The other components are zero.
    """
    assert pseudoscalars.shape[-1] == 1
    non_scalar_shape = list(pseudoscalars.shape[:-1]) + [15]
    non_scalar_components = torch.zeros(
        non_scalar_shape, device=pseudoscalars.device, dtype=pseudoscalars.dtype
    )
    embedding = torch.cat((non_scalar_components, pseudoscalars), dim=-1)

    return embedding


def extract_pseudoscalar(multivectors: torch.Tensor) -> torch.Tensor:
    """Extracts pseudoscalar components from multivectors.

    Parameters
    ----------
    multivectors: torch.Tensor
        Multivector inputs with shape (..., 16).

    Returns
    -------
    pseudoscalars: torch.Tensor
        Pseudoscalar component of multivectors with shape (..., 1).
    """

    return multivectors[..., [15]]
