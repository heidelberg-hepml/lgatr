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
_flash_available = importlib.util.find_spec("flash-attn") is not None
_torch_version = torch.__version__.split("+")[0]
_flex_available = tuple(int(x) for x in _torch_version.split(".")[:2]) >= (2, 7)
_varlen_available = tuple(int(x) for x in _torch_version.split(".")[:2]) >= (2, 10)


def _random_qkv(shape, dtype=torch.float32, device=None):
    if device is None:
        device = torch.device("cpu")

    q = torch.randn(*shape, dtype=dtype, device=device)
    k = torch.randn(*shape, dtype=dtype, device=device)
    v = torch.randn(*shape, dtype=dtype, device=device)
    return q, k, v


def _sparsify_shape(shape_dense, device=None):
    if device is None:
        device = torch.device("cpu")

    shape_sparse = (1, shape_dense[1], shape_dense[0] * shape_dense[2], shape_dense[3])
    cu_seq = [shape_dense[2] * i for i in range(shape_dense[0] + 1)]
    cu_seq = torch.tensor(cu_seq, dtype=torch.int32, device=device)
    max_seq = shape_dense[2]
    return shape_sparse, cu_seq, max_seq


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
    from lgatr.primitives.attention_backends.xformers import attention

    # check that backend exists
    backend_fn = get_attention_backend(backend="xformers")
    assert backend_fn is attention

    # check that result has correct shape
    qkv = _random_qkv(shape)
    out = backend_fn(*qkv)
    assert out.shape == shape

    # check agreement with default attention
    default_backend_fn = get_attention_backend()
    out_default = default_backend_fn(*qkv)
    torch.testing.assert_close(out, out_default, **TOLERANCES)


@pytest.mark.skipif(not _flex_available, reason="flex requires torch>=2.7")
@pytest.mark.parametrize("shape", SHAPES)
def test_flex_backend_selection(shape):
    from lgatr.primitives.attention_backends.flex import attention

    # check that backend exists
    backend_fn = get_attention_backend(backend="flex")
    assert backend_fn is attention

    # check that result has correct shape
    qkv = _random_qkv(shape)
    out = backend_fn(*qkv)
    assert out.shape == shape

    # check agreement with default attention
    default_backend_fn = get_attention_backend()
    out_default = default_backend_fn(*qkv)
    torch.testing.assert_close(out, out_default, **TOLERANCES)


@pytest.mark.skipif(
    not _flash_available or not torch.cuda.is_available(),
    reason="flash requires flash-attn package and CUDA",
)
@pytest.mark.parametrize("shape", [SHAPES[0]])
def test_flash_backend_selection(shape):
    from lgatr.primitives.attention_backends.flash import attention

    # check that backend exists
    backend_fn = get_attention_backend(backend="flash")
    assert backend_fn is attention

    # check that result has correct shape
    device = torch.device("cuda")
    shape, cu_seq, max_seq = _sparsify_shape(shape, device=device)
    kwargs = {
        "cu_seqlens_q": cu_seq,
        "cu_seqlens_k": cu_seq,
        "max_seqlen_q": max_seq,
        "max_seqlen_k": max_seq,
    }
    qkv = _random_qkv(shape, device=device)
    out = backend_fn(*qkv, **kwargs)
    assert out.shape == shape


@pytest.mark.skipif(
    not _varlen_available or not torch.cuda.is_available(),
    reason="varlen requires torch>=2.10 and CUDA",
)
@pytest.mark.parametrize("shape", [SHAPES[0]])
def test_varlen_backend_selection(shape):
    from lgatr.primitives.attention_backends.varlen import attention

    # check that backend exists
    backend_fn = get_attention_backend(backend="varlen")
    assert backend_fn is attention

    # check that result has correct shape
    device = torch.device("cuda")
    shape, cu_seq, max_seq = _sparsify_shape(shape, device=device)
    kwargs = {"cu_seq_q": cu_seq, "cu_seq_k": cu_seq, "max_q": max_seq, "max_k": max_seq}
    qkv = _random_qkv(shape, device=device)
    out = backend_fn(*qkv, **kwargs)
    assert out.shape == shape
