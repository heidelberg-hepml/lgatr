import torch

from lgatr.interface import embed_scalar, extract_scalar
from tests.helpers import BATCH_DIMS, TOLERANCES


def test_embed_scalar() -> None:
    # embed_scalar puts the input in slot 0 and zeros out the rest; extract_scalar inverts it.
    scalars = torch.randn(*BATCH_DIMS, 1)
    mv = embed_scalar(scalars)

    torch.testing.assert_close(mv[..., [0]], scalars, **TOLERANCES)
    assert (mv[..., 1:] == 0).all()
    torch.testing.assert_close(extract_scalar(mv), scalars, **TOLERANCES)
