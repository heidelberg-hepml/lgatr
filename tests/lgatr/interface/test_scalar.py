import pytest
import torch

from lgatr.interface import embed_scalar, extract_scalar
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_embed_scalar(batch_dims: list[int]) -> None:
    # embed_scalar puts the input in slot 0 and zeros out the rest; extract_scalar inverts it.
    scalar = torch.randn(*batch_dims, 1)
    mv = embed_scalar(scalar)

    # Check that scalar part of MV contains the data we wanted to embed
    mv_scalar = mv[..., [0]]
    torch.testing.assert_close(mv_scalar, scalar, **TOLERANCES)

    # Check that other components of MV are empty
    mv_other = mv[..., 1:]
    torch.testing.assert_close(mv_other, torch.zeros_like(mv_other), **TOLERANCES)

    # Check that we can extract the scalar back from the multivector
    other_scalar = extract_scalar(embed_scalar(scalar))
    torch.testing.assert_close(scalar, other_scalar, **TOLERANCES)
