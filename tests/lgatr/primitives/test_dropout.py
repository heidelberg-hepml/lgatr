import pytest
import torch

from lgatr.primitives.dropout import grade_dropout
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("p", [0.0, 0.2])
@pytest.mark.parametrize("training", [True, False])
def test_dropout_shape(training: bool, p: float) -> None:
    # grade_dropout preserves the input shape, and is the identity whenever it is switched off.
    x = torch.randn(*BATCH_DIMS, 16)
    y = grade_dropout(x, p=p, training=training)

    assert y.shape == x.shape
    if p == 0.0 or not training:
        torch.testing.assert_close(y, x, **TOLERANCES)


def test_dropout_expectation(num_trials: int = 10000) -> None:
    # grade_dropout's train-time output matches its test-time output in expectation.
    x = torch.randn(*BATCH_DIMS, 16)
    y_train = grade_dropout(x.unsqueeze(0).expand(num_trials, *x.shape), p=0.2, training=True)

    # Over 10k trials we won't get perfect agreement
    torch.testing.assert_close(y_train.mean(dim=0), x, **MILD_TOLERANCES)


def test_dropout_equivariance() -> None:
    # grade_dropout is Pin-equivariant at test time.
    check_pin_equivariance(
        grade_dropout,
        1,
        batch_dims=BATCH_DIMS,
        fn_kwargs=dict(training=False, p=0.2),
        spin=False,
        **TOLERANCES,
    )
