import pytest
import torch

from lgatr.interface.spurions import get_num_spurions, get_spurions


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
def test_get_num_spurions(beam_spurion, add_time_spurion, beam_mirror, expected_num):
    """Check that get_num_spurions returns the correct integer for each config."""
    result = get_num_spurions(
        beam_spurion=beam_spurion,
        add_time_spurion=add_time_spurion,
        beam_mirror=beam_mirror,
    )
    assert result == expected_num


@pytest.mark.parametrize(
    "beam_spurion, add_time_spurion, beam_mirror",
    [
        ("xyplane", True, True),
        ("xyplane", False, False),
        ("lightlike", True, True),
        ("lightlike", False, False),
        ("spacelike", True, True),
        ("spacelike", False, False),
        ("timelike", True, True),
        ("timelike", False, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_get_spurions(beam_spurion, add_time_spurion, beam_mirror):
    """
    Verify that get_spurions returns a tensor of shape (n_spurions, 16),
    where n_spurions = get_num_spurions(...).
    """
    expected_num = get_num_spurions(
        beam_spurion=beam_spurion,
        add_time_spurion=add_time_spurion,
        beam_mirror=beam_mirror,
    )
    result = get_spurions(
        beam_spurion=beam_spurion,
        add_time_spurion=add_time_spurion,
        beam_mirror=beam_mirror,
        device="cpu",
        dtype=torch.float32,
    )

    # Check type
    assert isinstance(result, torch.Tensor), "Output should be a torch.Tensor."

    # Check shape
    assert result.shape == (expected_num, 16)
