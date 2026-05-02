import pytest
import torch

from lgatr.interface import embed_axialvector, extract_axialvector
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_axialvector_embedding_consistency(batch_dims: list[int]) -> None:
    # embed_axialvector and extract_axialvector are inverses of each other.
    axialvectors = torch.randn(*batch_dims, 4)
    multivectors = embed_axialvector(axialvectors)
    axialvectors_reencoded = extract_axialvector(multivectors)
    torch.testing.assert_close(axialvectors, axialvectors_reencoded, **TOLERANCES)
