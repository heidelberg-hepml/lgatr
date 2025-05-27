import torch


def embed_axialvector(axialvector: torch.Tensor) -> torch.Tensor:
    """Embeds axial vectors in multivectors.

    Parameters
    ----------
    axialvector : torch.Tensor with shape (..., 4)
        Axial vector

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = axialvector.shape[:-1]
    multivector = torch.zeros(
        *batch_shape, 16, dtype=axialvector.dtype, device=axialvector.device
    )

    # Embedding into Lorentz vectors
    multivector[..., 11:15] = axialvector.flip(-1)

    return multivector


def extract_axialvector(multivector: torch.Tensor) -> torch.Tensor:
    """Given a multivector, extract a axial vector.

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 16)
        Multivector.

    Returns
    -------
    axialvector : torch.Tensor with shape (..., 4)
        Axial vector
    """

    axialvector = multivector[..., 11:15].flip(-1)

    return axialvector
