import pytest
import torch

from lgatr.interface import embed_pseudoscalar, extract_pseudoscalar
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_embed_pseudoscalar(batch_dims: list[int]) -> None:
    # embed_pseudoscalar puts the input in slot 15 and zeros out the rest; extract inverts it.
    pseudoscalars = torch.randn(*batch_dims, 1)
    mv = embed_pseudoscalar(pseudoscalars)

    # Check that scalar part of MV contains the data we wanted to embed
    mv_pseudoscalars = mv[..., [15]]
    torch.testing.assert_close(mv_pseudoscalars, pseudoscalars, **TOLERANCES)

    # Check that other components of MV are empty
    mv_other = mv[..., :-1]
    torch.testing.assert_close(mv_other, torch.zeros_like(mv_other), **TOLERANCES)

    # Check that we can extract the pseudoscalars back from the multivector
    other_pseudoscalars = extract_pseudoscalar(embed_pseudoscalar(pseudoscalars))
    torch.testing.assert_close(pseudoscalars, other_pseudoscalars, **TOLERANCES)
