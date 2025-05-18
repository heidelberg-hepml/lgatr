import pytest
import torch

from lgatr.interface import embed_pseudoscalar
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_embed_pseudoscalar(batch_dims):
    """Tests that embed_pseudoscalar() embeds pseudoscalars into multivectors correctly."""

    pseudoscalar = torch.randn(*batch_dims, 1)
    mv = embed_pseudoscalar(pseudoscalar)

    # Check that scalar part of MV contains the data we wanted to embed
    mv_pseudoscalar = mv[..., [15]]
    torch.testing.assert_close(mv_pseudoscalar, pseudoscalar, **TOLERANCES)

    # Check that other components of MV are empty
    mv_other = mv[..., :-1]
    torch.testing.assert_close(mv_other, torch.zeros_like(mv_other), **TOLERANCES)
