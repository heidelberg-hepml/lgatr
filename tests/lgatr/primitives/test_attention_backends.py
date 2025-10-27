import importlib.util

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

from lgatr.primitives.attention_backends import get_attention_backend
from tests.helpers.constants import STRICT_TOLERANCES as TOLERANCES

SHAPES = [
    (32, 8, 5, 32),
    (9, 3, 7, 13),
]


_xformers_available = importlib.util.find_spec("xformers") is not None
_torch_version = torch.__version__.split("+")[0]
_flex_available = tuple(int(x) for x in _torch_version.split(".")[:2]) >= (2, 7)


def _random_qkv(shape, dtype=torch.float32):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    q = torch.randn(*shape, dtype=dtype, device=device)
    k = torch.randn(*shape, dtype=dtype, device=device)
    v = torch.randn(*shape, dtype=dtype, device=device)
    return q, k, v


@pytest.mark.parametrize("shape", SHAPES)
def test_default_backend_selection(shape):
    backend_fn = get_attention_backend()
    assert backend_fn is torch_sdpa

    qkv = _random_qkv(shape)
    out = backend_fn(*qkv)
    assert out.shape == shape


@pytest.mark.skipif(not _xformers_available, reason="xformers not installed")
@pytest.mark.parametrize("shape", SHAPES)
def test_xformers_backend_selection(shape):
    # check that backend exists
    backend_fn = get_attention_backend(backend="xformers_attention")

    # check that result has correct shape
    qkv = _random_qkv(shape)
    out = backend_fn(*qkv)
    assert out.shape == shape

    # check agreement with default attention
    default_backend_fn = get_attention_backend()
    out_default = default_backend_fn(*qkv)
    torch.testing.assert_close(out, out_default, **TOLERANCES)


@pytest.mark.skipif(not _flex_available, reason="flex_attention requires torch>=2.7")
@pytest.mark.parametrize("shape", SHAPES)
def test_flex_backend_selection(shape):
    # check that backend exists
    backend_fn = get_attention_backend(backend="flex_attention")

    # check that result has correct shape
    qkv = _random_qkv(shape)
    out = backend_fn(*qkv)
    assert out.shape == shape

    # check agreement with default attention
    default_backend_fn = get_attention_backend()
    out_default = default_backend_fn(*qkv)
    torch.testing.assert_close(out, out_default, **TOLERANCES)
