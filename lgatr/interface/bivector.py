"""Embedding and extracting bivectors into multivectors."""

import torch


def embed_bivector(bivectors: torch.Tensor) -> torch.Tensor:
    """Embed Lorentz bivectors into multivectors.

    Parameters
    ----------
    bivectors
        Lorentz bivectors of shape ``(..., 6)`` using the convention (tx, ty, tz, xy, xz, yz).

    Returns
    -------
    multivectors
        Multivectors of shape ``(..., 16)``.
    """

    assert bivectors.shape[-1] == 6
    # F.pad(x, (5, 5)) zero-pads 5 entries on the left and 5 on the right of the last dim,
    # placing the input at indices 5-10 (the bivector slots) with zeros elsewhere.
    return torch.nn.functional.pad(bivectors, (5, 5))


def extract_bivector(multivectors: torch.Tensor) -> torch.Tensor:
    """Extract Lorentz bivectors from multivectors.

    Parameters
    ----------
    multivectors
        Multivectors of shape ``(..., 16)``.

    Returns
    -------
    bivectors
        Lorentz bivectors of shape ``(..., 6)``.
    """

    assert multivectors.shape[-1] == 16
    bivectors = multivectors[..., 5:11]

    return bivectors
