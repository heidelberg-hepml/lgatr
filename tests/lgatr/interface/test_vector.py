import torch

from lgatr.interface import embed_vector, extract_vector
from tests.helpers import BATCH_DIMS, TOLERANCES


def test_embed_vector() -> None:
    # embed_vector puts the input in the vector slots 1-4 and zeros out the rest; extract inverts it.
    vectors = torch.randn(*BATCH_DIMS, 4)
    mv = embed_vector(vectors)

    torch.testing.assert_close(mv[..., 1:5], vectors, **TOLERANCES)
    assert (mv[..., :1] == 0).all()
    assert (mv[..., 5:] == 0).all()
    torch.testing.assert_close(extract_vector(mv), vectors, **TOLERANCES)
