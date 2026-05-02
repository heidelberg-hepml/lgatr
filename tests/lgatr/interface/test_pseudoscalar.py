import pytest
import torch

from lgatr.interface import embed_pseudoscalar, extract_pseudoscalar
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_embed_pseudoscalar(batch_dims: list[int]) -> None:
    # embed_pseudoscalar puts the input in slot 15 and zeros out the rest; extract inverts it.
    pseudoscalar = torch.randn(*batch_dims, 1)
    mv = embed_pseudoscalar(pseudoscalar)

    # Check that scalar part of MV contains the data we wanted to embed
    mv_pseudoscalar = mv[..., [15]]
    torch.testing.assert_close(mv_pseudoscalar, pseudoscalar, **TOLERANCES)

    # Check that other components of MV are empty
    mv_other = mv[..., :-1]
    torch.testing.assert_close(mv_other, torch.zeros_like(mv_other), **TOLERANCES)

    # Check that we can extract the pseudoscalar back from the multivector
    other_pseudoscalar = extract_pseudoscalar(embed_pseudoscalar(pseudoscalar))
    torch.testing.assert_close(pseudoscalar, other_pseudoscalar, **TOLERANCES)
