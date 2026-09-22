import math

import pytest
import torch

from lgatr.interface import from_lightcone, get_lightcone_frame, to_lightcone
from tests.helpers import BATCH_DIMS, TOLERANCES

METRIC = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0]))
# metric in light-cone coordinates: x+ pairs with x-, the transverse components are negated
METRIC_LIGHTCONE = torch.tensor(
    [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, -1.0]]
)


def _random_reference(*batch_dims: int) -> torch.Tensor:
    # timelike reference with a generic spatial direction, like a jet momentum
    spatial = torch.randn(*batch_dims, 3)
    time = spatial.norm(dim=-1, keepdim=True) * (1 + torch.rand(*batch_dims, 1))
    return torch.cat([time, spatial], dim=-1)


# references along the z axis or at rest, where the transverse directions are not unique
DEGENERATE_REFERENCES = torch.tensor(
    [[2.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, -1.0], [2.0, 0.0, 0.0, 0.0]]
)


@pytest.mark.parametrize("reference", [_random_reference(*BATCH_DIMS), DEGENERATE_REFERENCES])
def test_frame_is_orthogonal_with_lightcone_metric(reference: torch.Tensor) -> None:
    # T T^T = 1 and T eta T^T = eta_lightcone, so the map preserves Minkowski products
    frame = get_lightcone_frame(reference)
    assert frame.shape == (*reference.shape, 4)
    torch.testing.assert_close(frame @ frame.mT, torch.eye(4).expand_as(frame), **TOLERANCES)
    torch.testing.assert_close(
        frame @ METRIC @ frame.mT, METRIC_LIGHTCONE.expand_as(frame), **TOLERANCES
    )


def test_roundtrip_and_products() -> None:
    # from_lightcone inverts to_lightcone, and Minkowski products are unchanged
    frame = get_lightcone_frame(_random_reference(BATCH_DIMS[0]))[:, None]
    v = torch.randn(*BATCH_DIMS, 4)
    w = torch.randn(*BATCH_DIMS, 4)
    v_lc, w_lc = to_lightcone(v, frame), to_lightcone(w, frame)

    torch.testing.assert_close(from_lightcone(v_lc, frame), v, **TOLERANCES)
    torch.testing.assert_close(
        torch.einsum("...i,ij,...j->...", v_lc, METRIC_LIGHTCONE, w_lc),
        torch.einsum("...i,ij,...j->...", v, METRIC, w),
        **TOLERANCES,
    )


def test_lightcone_components() -> None:
    # a vector collinear with the reference has a vanishing x- component, which is the small
    # number the light-cone coordinates keep explicit
    frame = get_lightcone_frame(torch.tensor([2.0, 2.0, 0.0, 0.0]))
    collinear = to_lightcone(torch.tensor([3.0, 3.0, 0.0, 0.0]), frame)
    torch.testing.assert_close(collinear, torch.tensor([3.0 * math.sqrt(2), 0.0, 0.0, 0.0]))
