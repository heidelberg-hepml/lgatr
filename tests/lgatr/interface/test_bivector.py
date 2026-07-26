import torch

from lgatr.interface import embed_bivector, extract_bivector
from tests.helpers import BATCH_DIMS, TOLERANCES


def test_bivector_embedding() -> None:
    # embed_bivector occupies the bivector slots 5-10 and leaves the other grades at zero;
    # extract_bivector inverts it.
    bivectors = torch.randn(*BATCH_DIMS, 6)
    multivectors = embed_bivector(bivectors)

    assert multivectors.shape == (*BATCH_DIMS, 16)
    torch.testing.assert_close(multivectors[..., 5:11], bivectors, **TOLERANCES)
    assert (multivectors[..., :5] == 0).all()
    assert (multivectors[..., 11:] == 0).all()
    torch.testing.assert_close(extract_bivector(multivectors), bivectors, **TOLERANCES)
