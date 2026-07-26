"""Self-tests of the equivariance harness in tests/helpers."""

import pytest
import torch

from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("spin", [True, False])
def test_equivariance_harness_detects_violations(spin: bool) -> None:
    # The harness must accept the identity and reject a non-equivariant function (torch.square is
    # not even linear). Without the negative half, a degenerate transformation -- e.g. one that
    # samples the identity -- would silently make every equivariance test in the suite pass.
    check_pin_equivariance(lambda x: x, 1, batch_dims=BATCH_DIMS, spin=spin, **TOLERANCES)

    with pytest.raises(AssertionError):
        check_pin_equivariance(torch.square, 1, batch_dims=BATCH_DIMS, spin=spin, **TOLERANCES)
