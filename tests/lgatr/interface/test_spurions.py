import pytest
import torch

from lgatr.interface import embed_vector
from lgatr.interface.spurions import get_num_spurions, get_spurions
from tests.helpers import TOLERANCES

# Beam 4-momenta documented in get_spurions, with their pz-mirrored partners.
BEAMS = {
    "lightlike": [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0]],
    "timelike": [[2**0.5, 0.0, 0.0, 1.0], [2**0.5, 0.0, 0.0, -1.0]],
    "spacelike": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]],
}
TIME = [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "beam_spurion, add_time_spurion, beam_mirror, expected_num",
    [
        # beam_spurion = "xyplane"
        ("xyplane", True, True, 2),  # +1 for xyplane +1 for time
        ("xyplane", False, True, 1),
        ("xyplane", True, False, 2),
        ("xyplane", False, False, 1),
        # beam_spurion = "lightlike"
        ("lightlike", True, True, 3),  # +2 for mirror +1 for time
        ("lightlike", True, False, 2),
        ("lightlike", False, True, 2),
        ("lightlike", False, False, 1),
        # beam_spurion = "spacelike"
        ("spacelike", True, True, 3),
        ("spacelike", False, True, 2),
        ("spacelike", True, False, 2),
        ("spacelike", False, False, 1),
        # beam_spurion = "timelike"
        ("timelike", True, True, 3),
        ("timelike", False, True, 2),
        ("timelike", True, False, 2),
        ("timelike", False, False, 1),
        # beam_spurion = None
        (None, True, True, 1),  # +0 for beam +1 for time
        (None, False, True, 0),
        (None, True, False, 1),
        (None, False, False, 0),
    ],
)
def test_get_num_spurions(
    beam_spurion: str | None,
    add_time_spurion: bool,
    beam_mirror: bool,
    expected_num: int,
) -> None:
    # get_num_spurions returns the expected count for each combination.
    result = get_num_spurions(
        beam_spurion=beam_spurion,
        add_time_spurion=add_time_spurion,
        beam_mirror=beam_mirror,
    )
    assert result == expected_num


@pytest.mark.parametrize("beam_spurion", list(BEAMS))
@pytest.mark.parametrize("beam_mirror", [False, True])
def test_get_spurions_beam_values(beam_spurion: str, beam_mirror: bool) -> None:
    # The beam spurions are the documented 4-momenta embedded as vectors, followed by the time
    # direction; beam_mirror appends the pz-flipped partner.
    beams = BEAMS[beam_spurion][: 2 if beam_mirror else 1]
    expected = embed_vector(torch.tensor(beams + [TIME]))

    spurions = get_spurions(beam_spurion=beam_spurion, beam_mirror=beam_mirror)
    torch.testing.assert_close(spurions, expected, **TOLERANCES)


def test_get_spurions_xyplane_value() -> None:
    # The xy-plane spurion is the xy bivector (slot 8) and nothing else.
    expected = torch.zeros(1, 16)
    expected[0, 8] = 1.0

    spurions = get_spurions(beam_spurion="xyplane", add_time_spurion=False)
    torch.testing.assert_close(spurions, expected, **TOLERANCES)


def test_get_spurions_time_only() -> None:
    # Without a beam spurion, only the time direction is returned.
    spurions = get_spurions(beam_spurion=None)
    torch.testing.assert_close(spurions, embed_vector(torch.tensor([TIME])), **TOLERANCES)
