"""Light-cone coordinates, for well-conditioned Minkowski products in low precision."""

import itertools
import math

import torch

from ..utils.autocast import minimum_autocast_precision

# Basis blades of each grade, in the order of the 16 multivector components.
_BLADES = [
    torch.tensor(list(itertools.combinations(range(4), grade)), dtype=torch.long)
    for grade in range(5)
]
_GRADE_SLICES = ((0, 1), (1, 5), (5, 11), (11, 15), (15, 16))


def get_lightcone_frame(reference: torch.Tensor) -> torch.Tensor:
    """Construct the orthogonal map from Cartesian to light-cone coordinates.

    With ``n`` the spatial direction of ``reference`` and ``e1``, ``e2`` two transverse unit
    vectors, the map sends a Lorentz vector ``v = (t, r)`` to

    .. math::
        x^+ = (t + r \\cdot n)/\\sqrt{2}, \\quad x^- = (t - r \\cdot n)/\\sqrt{2},
        \\quad x^1 = r \\cdot e_1, \\quad x^2 = r \\cdot e_2.

    Nearly massless vectors collinear with ``n`` have a small ``x^-``, which is stored explicitly
    instead of as a difference of large numbers. A good reference is the summed four-momentum of
    the items, e.g. the jet momentum. The frame cancels in the outputs and its gradient is singular
    along the z axis and at rest, so pass a detached ``reference``.

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

    # rows x+, x-, then the transverse e1 = (-sin phi, cos phi, 0) and e2 = e1 x n (det = 1)
    rows = [
        torch.stack([one, nx, ny, nz], dim=-1) / math.sqrt(2),
        torch.stack([one, -nx, -ny, -nz], dim=-1) / math.sqrt(2),
        torch.stack([zero, -sin_phi, cos_phi, zero], dim=-1),
        torch.stack([zero, cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def get_lightcone_frame_mv(frame: torch.Tensor) -> torch.Tensor:
    """Lift a light-cone frame to the map on multivectors.

    Multivectors transform with the outermorphism of ``frame``: the grade-``k`` block of the map
    collects the ``k x k`` minors of ``frame``.

    Parameters
    ----------
    frame
        Maps of shape ``(..., 4, 4)`` from :func:`get_lightcone_frame`.

    Returns
    -------
    frame_mv
        Maps of shape ``(..., 16, 16)``.
    """
    assert frame.shape[-2:] == (4, 4)
    frame_mv = frame.new_zeros(*frame.shape[:-2], 16, 16)
    for (start, stop), blade in zip(_GRADE_SLICES, _BLADES, strict=True):
        blade = blade.to(frame.device)
        minors = frame[..., blade[:, None, :, None], blade[None, :, None, :]]
        frame_mv[..., start:stop, start:stop] = torch.linalg.det(minors)
    return frame_mv


@minimum_autocast_precision(torch.float32)
def to_lightcone_mv(multivectors: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Map multivectors into the light-cone coordinates of ``frame``.

    Every multivector input has to be mapped, including spurions. Runs in float32 even inside
    autocast.

    Parameters
    ----------
    multivectors
        Multivectors of shape ``(..., 16)`` in Cartesian coordinates.
    frame
        Maps of shape ``(..., 4, 4)`` from :func:`get_lightcone_frame`, broadcast against
        ``multivectors``.

    Returns
    -------
    multivectors
        Multivectors of shape ``(..., 16)`` in light-cone coordinates.
    """
    assert multivectors.shape[-1] == 16
    # einsum instead of matmul, which would materialize the broadcast frame per multivector
    return torch.einsum("...ij,...j->...i", get_lightcone_frame_mv(frame), multivectors)


@minimum_autocast_precision(torch.float32)
def from_lightcone_mv(multivectors: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Map multivectors back from the light-cone coordinates of ``frame``.

    Parameters
    ----------
    multivectors
        Multivectors of shape ``(..., 16)`` in light-cone coordinates.
    frame
        Maps of shape ``(..., 4, 4)`` from :func:`get_lightcone_frame`, broadcast against
        ``multivectors``.

    Returns
    -------
    multivectors
        Multivectors of shape ``(..., 16)`` in Cartesian coordinates.
    """
    return to_lightcone_mv(multivectors, frame.mT)


@minimum_autocast_precision(torch.float32)
def to_lightcone(vectors: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Map Lorentz vectors into the light-cone coordinates of ``frame``.

    Every vector input has to be mapped, including spurions. Runs in float32 even inside
    autocast.

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
    return torch.einsum("...ij,...j->...i", frame, vectors)


@minimum_autocast_precision(torch.float32)
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
