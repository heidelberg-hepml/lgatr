import pytest
import torch
from torch import nn

from lgatr.layers.layer_norm import EquiLayerNorm
from tests.helpers import TOLERANCES, check_pin_equivariance

BATCH_DIMS = (7, 9)
NUM_SCALARS = 9


@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_equi_layer_norm_layer_equivariance(elementwise_affine: bool) -> None:
    # EquiLayerNorm is Pin-equivariant, including with a non-trivial per-grade affine gain.
    # The normalization itself is covered by tests/lgatr/primitives/test_normalization.py.
    mv_channels = BATCH_DIMS[-1]
    layer = EquiLayerNorm(mv_channels, NUM_SCALARS, elementwise_affine=elementwise_affine)
    if elementwise_affine:
        # default init is all-ones, which is indistinguishable from off, so randomize
        nn.init.normal_(layer.weight_mv)
        nn.init.normal_(layer.weight_s)
    scalars = torch.randn(*BATCH_DIMS, NUM_SCALARS)
    check_pin_equivariance(
        layer, 1, batch_dims=BATCH_DIMS, fn_kwargs=dict(scalars=scalars), spin=False, **TOLERANCES
    )


def test_equi_layer_norm_none_scalars() -> None:
    # EquiLayerNorm propagates scalars=None.
    layer = EquiLayerNorm()
    inputs = torch.randn(4, 5, 16)
    outputs_mv, outputs_s = layer(inputs, scalars=None)
    assert outputs_mv.shape == inputs.shape
    assert outputs_s is None
