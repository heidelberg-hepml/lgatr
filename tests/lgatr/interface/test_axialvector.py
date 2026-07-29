import pytest
import torch

from lgatr.interface import (
    embed_axialvector,
    embed_pseudoscalar,
    embed_vector,
    extract_axialvector,
)
from lgatr.primitives.bilinear import geometric_product
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, TOLERANCES


def test_axialvector_embedding_consistency() -> None:
    # embed_axialvector puts the input in the axialvector slots 11-14 and zeros out the rest;
    # extract_axialvector inverts it.
    axialvectors = torch.randn(*BATCH_DIMS, 4)
    multivectors = embed_axialvector(axialvectors)

    assert (multivectors[..., :11] == 0).all()
    assert (multivectors[..., 15:] == 0).all()
    torch.testing.assert_close(extract_axialvector(multivectors), axialvectors, **TOLERANCES)


@pytest.mark.parametrize("component", range(4))
def test_axialvector_embedding_convention(component: int) -> None:
    # Axial vectors use the same (t, x, y, z) component order as vectors: the dual of the i-th
    # vector component populates the i-th axialvector component and no other.
    vector = torch.zeros(4)
    vector[component] = 1.0
    dual = geometric_product(
        embed_vector(vector), embed_pseudoscalar(torch.ones(1)), config=PrimitivesConfig()
    )
    axialvector = extract_axialvector(dual)
    assert axialvector[component].abs() == pytest.approx(1.0)
    assert (axialvector[torch.arange(4) != component] == 0).all()
