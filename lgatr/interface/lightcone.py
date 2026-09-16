"""Light-cone coordinates, for well-conditioned Lorentz products in low precision."""

import torch

# guards against a vanishing spatial reference direction; see get_lightcone_frame
_EPS = 1e-12


def get_lightcone_frame(reference: torch.Tensor) -> torch.Tensor:
    """Construct the orthogonal map from Cartesian to light-cone coordinates.

    With ``n`` the spatial direction of ``reference``, ``e1 = z x n / |z x n|`` and
    ``e2 = n x e1``, the map sends a Lorentz vector ``v = (t, r)`` to

    .. math::
        x^+ = (t + r \\cdot n)/\\sqrt{2}, \\quad x^- = (t - r \\cdot n)/\\sqrt{2},
        \\quad x^1 = r \\cdot e_1, \\quad x^2 = r \\cdot e_2.

    The map ``T`` is orthogonal and ``T eta T^T`` is the light-cone metric that the slim layers
    use when the network is constructed with ``lightcone=True``. A network fed ``T v`` for every
    vector, spurions included, therefore computes the same function as one fed ``v``, up to
    floating-point rounding.

    What changes is the conditioning. For two nearly collinear, nearly massless vectors the
    Cartesian product ``t t' - r . r'`` is a small difference of large numbers, so rounding the
    components to bf16/fp16 destroys it. Here the small ``t - r . n`` is formed once, in float64,
    and then stored as a number of its own, which is what makes half-precision attention and
    half-precision vector GEMMs as accurate as float32 (see ``lightcone`` in
    :class:`~lgatr.nets.slim.LGATrSlim`).

    A good reference is the summed four-momentum of the items the network sees, e.g. the jet
    four-momentum for a jet-level task, because the map helps precisely for those items that are
    collinear with it. The frame degenerates when the spatial part of ``reference`` vanishes, and
    ``e1`` degenerates when ``reference`` points exactly along the z axis; neither is meaningful
    for a jet, and no fallback is applied.

    Parameters
    ----------
    reference
        Lorentz vectors of shape ``(..., 4)`` using the convention (t, x, y, z).

    Returns
    -------
    frame
        Orthogonal maps of shape ``(..., 4, 4)``, in float64.
    """
    assert reference.shape[-1] == 4
    spatial = reference.to(torch.float64)[..., 1:]
    n = spatial / spatial.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    nx, ny, nz = n.unbind(-1)
    rho = torch.sqrt(nx**2 + ny**2).clamp_min(_EPS)
    zero, s = torch.zeros_like(nx), torch.full_like(nx, 2**-0.5)
    rows = [
        torch.stack([s, s * nx, s * ny, s * nz], dim=-1),
        torch.stack([s, -s * nx, -s * ny, -s * nz], dim=-1),
        torch.stack([zero, -ny / rho, nx / rho, zero], dim=-1),
        torch.stack([zero, -nz * nx / rho, -nz * ny / rho, rho], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def to_lightcone(vectors: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Map Lorentz vectors into the light-cone coordinates of ``frame``.

    Apply this to every vector the network sees, spurions included, and construct the network
    with ``lightcone=True``.

    Parameters
    ----------
    vectors
        Lorentz vectors of shape ``(..., 4)`` in Cartesian coordinates.
    frame
        Maps of shape ``(..., 4, 4)`` from :func:`get_lightcone_frame`, broadcast against
        ``vectors``; a per-event frame of shape ``(batch, 4, 4)`` applied to vectors of shape
        ``(batch, items, channels, 4)`` is passed as ``frame[:, None, None]``.

    Returns
    -------
    vectors
        Lorentz vectors of shape ``(..., 4)`` in light-cone coordinates, in the dtype of the
        inputs.
    """
    return _apply_frame(vectors, frame)


def from_lightcone(vectors: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Map Lorentz vectors back from the light-cone coordinates of ``frame``.

    The inverse of :func:`to_lightcone`, for network outputs with vector channels.

    Parameters
    ----------
    vectors
        Lorentz vectors of shape ``(..., 4)`` in light-cone coordinates.
    frame
        Maps of shape ``(..., 4, 4)`` from :func:`get_lightcone_frame`, broadcast as in
        :func:`to_lightcone`.

    Returns
    -------
    vectors
        Lorentz vectors of shape ``(..., 4)`` in Cartesian coordinates, in the dtype of the
        inputs.
    """
    return _apply_frame(vectors, frame.transpose(-1, -2))


def _apply_frame(vectors: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Apply ``frame`` to the last dim of ``vectors``, in float64."""
    assert vectors.shape[-1] == 4 and frame.shape[-2:] == (4, 4)
    outputs = frame.to(torch.float64) @ vectors.to(torch.float64).unsqueeze(-1)
    return outputs.squeeze(-1).to(vectors.dtype)
