import pytest
import torch

from lgatr.interface import embed_axialvector, extract_axialvector
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_axialvector_embedding_consistency(batch_dims):
    """Tests whether Lorentz vector embeddings into multivectors are cycle consistent."""
    axialvectors = torch.randn(*batch_dims, 4)
    multivectors = embed_axialvector(axialvectors)
    axialvectors_reencoded = extract_axialvector(multivectors)
    torch.testing.assert_close(axialvectors, axialvectors_reencoded, **TOLERANCES)
