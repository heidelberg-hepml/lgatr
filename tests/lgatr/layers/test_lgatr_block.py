import pytest
import torch

from lgatr.layers import LGATrBlock
from lgatr.layers.attention.config import SelfAttentionConfig
from lgatr.layers.mlp.config import MLPConfig
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

NUM_ITEMS, MV_CHANNELS = 8, 6

# GeometricBilinear needs a scalar stream, so s_channels=0 requires geometric_product=False.
S_CHANNELS = [(0, False), (2, True), (6, True)]


def _block(
    s_channels: int, geometric_product: bool, num_heads: int, multi_query: bool, **kwargs
) -> LGATrBlock:
    """An LGATrBlock with the standard test channel counts."""
    return LGATrBlock(
        MV_CHANNELS,
        s_channels=s_channels,
        attention=SelfAttentionConfig(num_heads=num_heads, multi_query=multi_query),
        mlp=MLPConfig(),
        primitives=PrimitivesConfig(geometric_product=geometric_product),
        **kwargs,
    )


@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("s_channels,geometric_product", S_CHANNELS)
@pytest.mark.parametrize("multi_query", [False, True])
def test_lgatr_block_shape(
    num_heads: int, s_channels: int, geometric_product: bool, multi_query: bool
) -> None:
    # LGATrBlock outputs match the input shape for both multivector and scalar streams.
    net = _block(s_channels, geometric_product, num_heads, multi_query, dropout_prob=0.3)

    inputs = torch.randn(*BATCH_DIMS, NUM_ITEMS, MV_CHANNELS, 16)
    scalars = torch.randn(*BATCH_DIMS, NUM_ITEMS, s_channels) if s_channels else None
    outputs, output_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (*BATCH_DIMS, NUM_ITEMS, MV_CHANNELS, 16)
    if s_channels:
        assert output_scalars.shape == (*BATCH_DIMS, NUM_ITEMS, s_channels)
    else:
        assert output_scalars is None


@pytest.mark.parametrize("s_channels,geometric_product", [(0, False), (6, True)])
@pytest.mark.parametrize("multi_query", [False, True])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
def test_lgatr_block_equivariance(
    s_channels: int, geometric_product: bool, multi_query: bool, norm_elementwise_affine: bool
) -> None:
    # LGATrBlock is Spin-equivariant. Only Spin, because of the fixed reference multivector in its
    # bilinear layers.
    net = _block(
        s_channels,
        geometric_product,
        num_heads=4,
        multi_query=multi_query,
        norm_elementwise_affine=norm_elementwise_affine,
    )

    scalars = torch.randn(*BATCH_DIMS, NUM_ITEMS, s_channels) if s_channels else None
    data_dims = (*BATCH_DIMS, NUM_ITEMS, MV_CHANNELS)
    check_pin_equivariance(
        net, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **MILD_TOLERANCES
    )


def test_lgatr_block_rejects_none_scalars() -> None:
    # An LGATrBlock built with scalar channels rejects scalars=None at runtime.
    net = _block(s_channels=2, geometric_product=True, num_heads=2, multi_query=False)
    with pytest.raises(ValueError):
        net(torch.randn(3, 5, MV_CHANNELS, 16), scalars=None)
