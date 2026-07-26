import torch

from lgatr.layers.dropout import GradeDropout

BATCH_DIMS = (10,)


def test_dropout_layer() -> None:
    # GradeDropout wraps grade_dropout (tested in tests/lgatr/primitives/test_dropout.py) and adds
    # scalar dropout. Check the shapes it produces, the p=0 identity, and scalars=None handling.
    layer = GradeDropout(p=0.2)
    layer.train()

    mv = torch.randn(*BATCH_DIMS, 16)
    s = torch.randn(*BATCH_DIMS)
    outputs_mv, outputs_s = layer(mv, s)
    assert outputs_mv.shape == mv.shape
    assert outputs_s.shape == s.shape

    outputs_mv, outputs_s = layer(mv, None)
    assert outputs_mv.shape == mv.shape
    assert outputs_s is None

    identity = GradeDropout(p=0.0)
    identity.train()
    outputs_mv, outputs_s = identity(mv, s)
    torch.testing.assert_close(outputs_mv, mv)
    torch.testing.assert_close(outputs_s, s)
