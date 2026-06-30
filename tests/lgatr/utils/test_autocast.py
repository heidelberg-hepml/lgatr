import pytest
import torch

import lgatr.utils.autocast as autocast_mod
from lgatr.utils.autocast import minimum_autocast_precision, naive_amp


# Choose dtypes to work on most devices -- torch.bfloat16 is not available on some GPUs
@pytest.mark.parametrize("device,amp_dtype", [("cpu", torch.bfloat16)])
def test_minimum_autocast_precision_inputs(device: str, amp_dtype: torch.dtype) -> None:
    # Low-precision floats cast up to min_dtype; others unchanged; no-op outside autocast.
    @minimum_autocast_precision(torch.float32)
    def return_input_dtypes(*args, **kwargs):
        dtypes = [arg.dtype if isinstance(arg, torch.Tensor) else None for arg in args]
        dtypes += [arg.dtype if isinstance(arg, torch.Tensor) else None for arg in kwargs.values()]
        return dtypes

    # Inputs
    input_dtypes = [
        amp_dtype,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int32,
        torch.bool,
        None,
    ]
    inputs = [
        "banana" if dtype is None else torch.empty(3, 5, device=device, dtype=dtype)
        for dtype in input_dtypes
    ]
    expected_dtypes = [
        torch.float32,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int32,
        torch.bool,
        None,
    ]

    # Test that without autocast, nothing happens
    dtypes0 = return_input_dtypes(*inputs)
    for got, expected in zip(dtypes0, input_dtypes, strict=False):
        assert got == expected

    # Test that when autocasting, inputs are correctly casted
    with torch.autocast(device, amp_dtype, enabled=True):
        dtypes1 = return_input_dtypes(*inputs)

    for got, expected in zip(dtypes1, expected_dtypes, strict=False):
        assert got == expected


# Choose dtypes to work on most devices -- torch.bfloat16 is not available on some GPUs
@pytest.mark.parametrize(
    "output_mode,expected_dtype",
    [
        (None, torch.float64),
        (torch.float64, torch.float64),
        ("low", torch.bfloat16),
        ("high", torch.float64),
    ],
)
def test_minimum_autocast_precision_outputs(
    output_mode,
    expected_dtype: torch.dtype,
    device: str = "cpu",
    amp_dtype: torch.dtype = torch.bfloat16,
) -> None:
    # minimum_autocast_precision honors the ``output`` mode: None / dtype / "low" / "high".
    @minimum_autocast_precision(torch.float32, output=output_mode)
    def sum_(*args):
        outputs = 0.0
        for arg in args:
            outputs = outputs + arg
        return outputs

    # Inputs
    input_dtypes = [torch.bfloat16, torch.float32, torch.float64]
    inputs = [torch.randn((3, 5), device=device, dtype=dtype) for dtype in input_dtypes]

    # Check output dtype
    with torch.autocast(device, amp_dtype, enabled=True):
        outputs = sum_(*inputs)
    assert outputs.dtype == expected_dtype


@pytest.mark.parametrize("device,amp_dtype", [("cpu", torch.bfloat16)])
def test_naive_amp_disables_islands(device: str, amp_dtype: torch.dtype) -> None:
    # Inside `with naive_amp()`, the decorator is a no-op so a low-precision input stays low.
    @minimum_autocast_precision(torch.float32)
    def input_dtype(x):
        return x.dtype

    x = torch.empty(3, 5, device=device, dtype=amp_dtype)
    with torch.autocast(device, amp_dtype, enabled=True):
        assert input_dtype(x) == torch.float32  # island on: upcast to fp32
        with naive_amp():
            assert input_dtype(x) == amp_dtype  # island bypassed: stays low precision
            assert torch.is_autocast_enabled(device)  # autocast left enabled (unlike the decorator)
        assert input_dtype(x) == torch.float32  # restored after the block


def test_naive_amp_false_is_noop() -> None:
    # naive_amp(False) leaves the global state untouched (never overrides an outer naive_amp).
    assert autocast_mod._NAIVE_AMP is False
    with naive_amp(False):
        assert autocast_mod._NAIVE_AMP is False
    with naive_amp():
        with naive_amp(False):
            assert autocast_mod._NAIVE_AMP is True
        assert autocast_mod._NAIVE_AMP is True
    assert autocast_mod._NAIVE_AMP is False


def test_naive_amp_nesting_and_exception_restore() -> None:
    # Nesting restores the prior value at each level; an exception still restores it.
    with naive_amp():
        assert autocast_mod._NAIVE_AMP is True
        with naive_amp():
            assert autocast_mod._NAIVE_AMP is True
        assert autocast_mod._NAIVE_AMP is True
    assert autocast_mod._NAIVE_AMP is False

    with pytest.raises(RuntimeError, match="boom"):
        with naive_amp():
            raise RuntimeError("boom")
    assert autocast_mod._NAIVE_AMP is False
