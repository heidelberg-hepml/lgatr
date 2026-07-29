import torch

from lgatr.interface import embed_pseudoscalar, extract_pseudoscalar
from tests.helpers import BATCH_DIMS, TOLERANCES


def test_embed_pseudoscalar() -> None:
    # embed_pseudoscalar puts the input in slot 15 and zeros out the rest; extract inverts it.
    pseudoscalars = torch.randn(*BATCH_DIMS, 1)
    mv = embed_pseudoscalar(pseudoscalars)

    torch.testing.assert_close(mv[..., [15]], pseudoscalars, **TOLERANCES)
    assert (mv[..., :15] == 0).all()
    torch.testing.assert_close(extract_pseudoscalar(mv), pseudoscalars, **TOLERANCES)
