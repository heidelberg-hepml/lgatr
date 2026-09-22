"""Light-cone coordinates, for well-conditioned Minkowski products in low precision."""

import math

import torch


def get_lightcone_frame(reference: torch.Tensor) -> torch.Tensor:
    """Construct the orthogonal map from Cartesian to light-cone coordinates.

    With ``n`` the spatial direction of ``reference`` and ``e1``, ``e2`` two transverse unit
    vectors, the map sends a Lorentz vector ``v = (t, r)`` to

    .. math::
        x^+ = (t + r \\cdot n)/\\sqrt{2}, \\quad x^- = (t - r \\cdot n)/\\sqrt{2},
        \\quad x^1 = r \\cdot e_1, \\quad x^2 = r \\cdot e_2.

    Nearly massless vectors collinear with ``n`` have a small ``x^-``, which is stored explicitly
    instead of as a difference of large numbers. A good reference is the summed four-momentum of
    the items, e.g. the jet momentum.

    Parameters
    ----------
    reference
        Lorentz vectors of shape ``(..., 4)`` using the convention (t, x, y, z).

    Returns
    -------
    frame
        Orthogonal maps of shape ``(..., 4, 4)``.
    """
    assert reference.shape[-1] == 4
    px, py, pz = reference[..., 1:].unbind(-1)
    phi = torch.atan2(py, px)
    theta = torch.atan2(torch.hypot(px, py), pz)
    cos_phi, sin_phi, cos_theta, sin_theta = phi.cos(), phi.sin(), theta.cos(), theta.sin()
    one, zero = torch.ones_like(phi), torch.zeros_like(phi)
    nx, ny, nz = sin_theta * cos_phi, sin_theta * sin_phi, cos_theta

    # rows x+, x-, then the transverse e1 = (-sin phi, cos phi, 0) and e2 = n x e1
    rows = [
        torch.stack([one, nx, ny, nz], dim=-1) / math.sqrt(2),
        torch.stack([one, -nx, -ny, -nz], dim=-1) / math.sqrt(2),
        torch.stack([zero, -sin_phi, cos_phi, zero], dim=-1),
        torch.stack([zero, -cos_theta * cos_phi, -cos_theta * sin_phi, sin_theta], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def to_lightcone(vectors: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Map Lorentz vectors into the light-cone coordinates of ``frame``.

    Call this in full precision, outside of autocast.

    Parameters
    ----------
    vectors
        Lorentz vectors of shape ``(..., 4)`` in Cartesian coordinates.
    frame
        Maps of shape ``(..., 4, 4)`` from :func:`get_lightcone_frame`, broadcast against
        ``vectors``.

    Returns
    -------
    vectors
        Lorentz vectors of shape ``(..., 4)`` in light-cone coordinates.
    """
    assert vectors.shape[-1] == 4 and frame.shape[-2:] == (4, 4)
    return (frame @ vectors.unsqueeze(-1)).squeeze(-1)


def from_lightcone(vectors: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Map Lorentz vectors back from the light-cone coordinates of ``frame``.

    Parameters
    ----------
    vectors
        Lorentz vectors of shape ``(..., 4)`` in light-cone coordinates.
    frame
        Maps of shape ``(..., 4, 4)`` from :func:`get_lightcone_frame`, broadcast against
        ``vectors``.

    Returns
    -------
    vectors
        Lorentz vectors of shape ``(..., 4)`` in Cartesian coordinates.
    """
    return to_lightcone(vectors, frame.mT)
