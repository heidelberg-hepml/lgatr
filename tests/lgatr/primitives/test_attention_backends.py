import importlib.util

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

from lgatr.primitives.attention_backends import get_attention_backend
from tests.helpers.constants import STRICT_TOLERANCES

SHAPES = [
    (32, 8, 5, 32),
    (9, 3, 7, 13),
]


_xformers_available = importlib.util.find_spec("xformers") is not None
_flash_available = importlib.util.find_spec("flash_attn") is not None
_torch_version = torch.__version__.split("+")[0]
_flex_available = tuple(int(x) for x in _torch_version.split(".")[:2]) >= (2, 7)
_varlen_available = tuple(int(x) for x in _torch_version.split(".")[:2]) >= (2, 10)


def _random_qkv(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random ``(query, key, value)`` tensors with the given shape."""
    if device is None:
        device = torch.device("cpu")

    q = torch.randn(*shape, dtype=dtype, device=device)
    k = torch.randn(*shape, dtype=dtype, device=device)
    v = torch.randn(*shape, dtype=dtype, device=device)
    return q, k, v


def _sparsify_shape(
    shape_dense: tuple[int, ...],
    device: torch.device | None = None,
) -> tuple[tuple[int, int, int, int], torch.Tensor, int]:
    """Convert a dense ``(B, H, T, D)`` shape into ``(1, H, B*T, D)`` plus cu-seqlens for varlen."""
    if device is None:
        device = torch.device("cpu")

    shape_sparse = (1, shape_dense[1], shape_dense[0] * shape_dense[2], shape_dense[3])
    cu_seq = [shape_dense[2] * i for i in range(shape_dense[0] + 1)]
    cu_seq = torch.tensor(cu_seq, dtype=torch.int32, device=device)
    max_seq = shape_dense[2]
    return shape_sparse, cu_seq, max_seq


@pytest.mark.parametrize("shape", SHAPES)
def test_default_backend_selection(shape: tuple[int, ...]) -> None:
    # The default attention backend is torch's native scaled_dot_product_attention.
    backend_fn = get_attention_backend()
    assert backend_fn is torch_sdpa

    qkv = _random_qkv(shape)
    out = backend_fn(*qkv)
    assert out.shape == shape


@pytest.mark.skipif(not _xformers_available, reason="xformers not installed")
@pytest.mark.parametrize("shape", SHAPES)
def test_xformers_backend_selection(shape: tuple[int, ...]) -> None:
    # Selecting backend="xformers" routes to the xformers wrapper and matches the default backend.
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
    torch.testing.assert_close(out, out_default, **STRICT_TOLERANCES)


@pytest.mark.skipif(not _flex_available, reason="flex requires torch>=2.7")
@pytest.mark.parametrize("shape", SHAPES)
def test_flex_backend_selection(shape: tuple[int, ...]) -> None:
    # Selecting backend="flex" routes to flex_attention and matches the default backend.
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
    torch.testing.assert_close(out, out_default, **STRICT_TOLERANCES)


def _dense_to_sparse(t: torch.Tensor, shape_dense: tuple[int, int, int, int]) -> torch.Tensor:
    """Reshape ``(B, H, T, D)`` to ``(1, H, B*T, D)`` so segments line up with cu_seqlens."""
    b, h, seq, d = shape_dense
    return t.permute(1, 0, 2, 3).reshape(1, h, b * seq, d).contiguous()


def _sparse_to_dense(t: torch.Tensor, shape_dense: tuple[int, int, int, int]) -> torch.Tensor:
    """Inverse of ``_dense_to_sparse``: ``(1, H, B*T, D)`` back to ``(B, H, T, D)``."""
    b, h, seq, d = shape_dense
    return t.reshape(h, b, seq, d).permute(1, 0, 2, 3).contiguous()


@pytest.mark.skipif(
    not _flash_available or not torch.cuda.is_available(),
    reason="flash requires flash-attn package and CUDA",
)
@pytest.mark.parametrize("shape", SHAPES)
def test_flash_backend_selection(shape: tuple[int, ...]) -> None:
    # Selecting backend="flash" routes to the flash-attention wrapper and matches the default
    # backend run independently per segment.
    from lgatr.primitives.attention_backends.flash import attention

    # check that backend exists
    backend_fn = get_attention_backend(backend="flash")
    assert backend_fn is attention

    # check that result has correct shape
    device = torch.device("cuda")
    shape_sparse, cu_seq, max_seq = _sparsify_shape(shape, device=device)
    qkv_dense = _random_qkv(shape, device=device)
    qkv_sparse = tuple(_dense_to_sparse(t, shape) for t in qkv_dense)
    kwargs = {
        "cu_seqlens_q": cu_seq,
        "cu_seqlens_k": cu_seq,
        "max_seqlen_q": max_seq,
        "max_seqlen_k": max_seq,
    }
    out = backend_fn(*qkv_sparse, **kwargs)
    assert out.shape == shape_sparse

    # check agreement with default attention applied per segment (i.e. dense (B, H, T, D) sdpa)
    default_backend_fn = get_attention_backend()
    out_default = default_backend_fn(*qkv_dense)
    out_dense = _sparse_to_dense(out, shape)
    torch.testing.assert_close(out_dense, out_default, **STRICT_TOLERANCES)


@pytest.mark.skipif(
    not _varlen_available or not torch.cuda.is_available(),
    reason="varlen requires torch>=2.10 and CUDA",
)
@pytest.mark.parametrize("shape", SHAPES)
def test_varlen_backend_selection(shape: tuple[int, ...]) -> None:
    # Selecting backend="varlen" routes to the varlen wrapper and matches the default backend
    # run independently per segment.
    from lgatr.primitives.attention_backends.varlen import attention

    # check that backend exists
    backend_fn = get_attention_backend(backend="varlen")
    assert backend_fn is attention

    # check that result has correct shape
    device = torch.device("cuda")
    shape_sparse, cu_seq, max_seq = _sparsify_shape(shape, device=device)
    qkv_dense = _random_qkv(shape, device=device)
    qkv_sparse = tuple(_dense_to_sparse(t, shape) for t in qkv_dense)
    kwargs = {"cu_seq_q": cu_seq, "cu_seq_k": cu_seq, "max_q": max_seq, "max_k": max_seq}
    out = backend_fn(*qkv_sparse, **kwargs)
    assert out.shape == shape_sparse

    # check agreement with default attention applied per segment (i.e. dense (B, H, T, D) sdpa)
    default_backend_fn = get_attention_backend()
    out_default = default_backend_fn(*qkv_dense)
    out_dense = _sparse_to_dense(out, shape)
    torch.testing.assert_close(out_dense, out_default, **STRICT_TOLERANCES)
