import pytest
import torch

from lgatr.layers.mlp.nonlinearities import ScalarGatedNonlinearity
from lgatr.utils.misc import get_nonlinearity
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance

ACTIVATIONS = ["gelu", "relu", "sigmoid", "silu", "tanh"]
NUM_SCALARS = 9


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_scalar_gated_nonlinearity_layer_correctness(activation: str) -> None:
    # The multivectors are gated by the activation of their own scalar component, while the
    # auxiliary scalars are passed through the activation directly.
    layer = ScalarGatedNonlinearity(nonlinearity=activation)
    multivectors = torch.randn(*BATCH_DIMS, 16)
    scalars = torch.randn(*BATCH_DIMS, NUM_SCALARS)

    outputs_mv, outputs_s = layer(multivectors, scalars=scalars)

    gate = get_nonlinearity(activation)(multivectors[..., [0]])
    torch.testing.assert_close(outputs_mv, gate * multivectors, **TOLERANCES)
    torch.testing.assert_close(outputs_s, get_nonlinearity(activation)(scalars), **TOLERANCES)


def test_scalar_gated_nonlinearity_layer_equivariance() -> None:
    # ScalarGatedNonlinearity is Pin-equivariant: the gate is built from the invariant scalar part.
    layer = ScalarGatedNonlinearity(nonlinearity="gelu")
    scalars = torch.randn(*BATCH_DIMS, NUM_SCALARS)
    check_pin_equivariance(
        layer, 1, batch_dims=BATCH_DIMS, fn_kwargs=dict(scalars=scalars), spin=False, **TOLERANCES
    )
