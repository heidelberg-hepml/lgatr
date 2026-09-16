import torch

from lgatr.interface import from_lightcone, get_lightcone_frame, to_lightcone
from tests.helpers import BATCH_DIMS, STRICT_TOLERANCES

METRIC = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64))
# metric in light-cone coordinates: x+ pairs with x-, the transverse components are negated
METRIC_LIGHTCONE = torch.tensor(
    [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, -1.0]],
    dtype=torch.float64,
)


def _random_reference(*batch_dims: int) -> torch.Tensor:
    # timelike reference with a generic spatial direction, like a jet four-momentum
    spatial = torch.randn(*batch_dims, 3, dtype=torch.float64)
    scale = 1 + torch.rand(*batch_dims, 1, dtype=torch.float64)
    time = spatial.norm(dim=-1, keepdim=True) * scale
    return torch.cat([time, spatial], dim=-1)


def test_frame_is_orthogonal_with_lightcone_metric() -> None:
    # T T^T = 1 and T eta T^T = eta_lightcone, so the map preserves Minkowski products
    frame = get_lightcone_frame(_random_reference(*BATCH_DIMS))
    assert frame.shape == (*BATCH_DIMS, 4, 4)

    eye = torch.eye(4, dtype=torch.float64).expand_as(frame)
    torch.testing.assert_close(frame @ frame.transpose(-1, -2), eye, **STRICT_TOLERANCES)
    torch.testing.assert_close(
        frame @ METRIC @ frame.transpose(-1, -2),
        METRIC_LIGHTCONE.expand_as(frame),
        **STRICT_TOLERANCES,
    )


def test_products_are_preserved() -> None:
    # Minkowski products are unchanged, which is why the networks compute the same function
    frame = get_lightcone_frame(_random_reference(BATCH_DIMS[0]))
    v = torch.randn(*BATCH_DIMS, 4, dtype=torch.float64)
    w = torch.randn(*BATCH_DIMS, 4, dtype=torch.float64)

    v_lc = to_lightcone(v, frame[:, None])
    w_lc = to_lightcone(w, frame[:, None])
    products = torch.einsum("...i,ij,...j->...", v, METRIC, w)
    products_lc = torch.einsum("...i,ij,...j->...", v_lc, METRIC_LIGHTCONE, w_lc)
    torch.testing.assert_close(products_lc, products, **STRICT_TOLERANCES)


def test_roundtrip_and_dtype() -> None:
    # from_lightcone inverts to_lightcone, and both keep the dtype of their inputs
    frame = get_lightcone_frame(_random_reference(BATCH_DIMS[0]))
    v = torch.randn(*BATCH_DIMS, 4, dtype=torch.float64)

    torch.testing.assert_close(
        from_lightcone(to_lightcone(v, frame[:, None]), frame[:, None]), v, **STRICT_TOLERANCES
    )

    v32 = v.to(torch.float32)
    assert to_lightcone(v32, frame[:, None]).dtype == torch.float32
    assert from_lightcone(v32, frame[:, None]).dtype == torch.float32


def test_lightcone_components() -> None:
    # a vector collinear with the reference has a vanishing x- component, which is the small
    # number the light-cone coordinates keep explicit
    reference = torch.tensor([[2.0, 0.0, 0.0, 2.0]], dtype=torch.float64)
    frame = get_lightcone_frame(reference)
    collinear = to_lightcone(torch.tensor([[3.0, 0.0, 0.0, 3.0]], dtype=torch.float64), frame)

    torch.testing.assert_close(collinear[..., 1], torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(collinear[..., 0], torch.tensor([3.0 * 2**0.5], dtype=torch.float64))
    torch.testing.assert_close(collinear[..., 2:], torch.zeros(1, 2, dtype=torch.float64))
