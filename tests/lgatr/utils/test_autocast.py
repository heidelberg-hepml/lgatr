import pytest
import torch

from lgatr.utils.autocast import minimum_autocast_precision


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
def test_minimum_autocast_precision_context_manager(device: str, amp_dtype: torch.dtype) -> None:
    # Inside the `with`: autocast disabled; `cast()` upcasts low-precision floats only.
    bf16_tensor = torch.empty(3, 5, device=device, dtype=amp_dtype)
    fp64_tensor = torch.empty(3, 5, device=device, dtype=torch.float64)
    int_tensor = torch.empty(3, 5, device=device, dtype=torch.int32)

    with torch.autocast(device, amp_dtype, enabled=True):
        assert torch.is_autocast_enabled(device)
        with minimum_autocast_precision(torch.float32) as mp:
            assert not torch.is_autocast_enabled(device)
            assert mp.cast(bf16_tensor).dtype == torch.float32
            assert mp.cast(fp64_tensor).dtype == torch.float64
            assert mp.cast(int_tensor).dtype == torch.int32
            assert mp.cast("not a tensor") == "not a tensor"
        assert torch.is_autocast_enabled(device)


def test_minimum_autocast_precision_context_manager_outside_autocast() -> None:
    # Without outer autocast: CM still safe; `cast()` still upcasts unconditionally.
    bf16_tensor = torch.empty(3, 5, dtype=torch.bfloat16)

    assert not torch.is_autocast_enabled("cpu")
    with minimum_autocast_precision(torch.float32) as mp:
        assert not torch.is_autocast_enabled("cpu")
        assert mp.cast(bf16_tensor).dtype == torch.float32
    assert not torch.is_autocast_enabled("cpu")


def test_minimum_autocast_precision_context_manager_exception_propagates() -> None:
    # Exceptions raised inside the `with` block propagate, and outer autocast state is restored.
    with torch.autocast("cpu", torch.bfloat16, enabled=True):
        with pytest.raises(RuntimeError, match="boom"):
            with minimum_autocast_precision(torch.float32):
                assert not torch.is_autocast_enabled("cpu")
                raise RuntimeError("boom")
        assert torch.is_autocast_enabled("cpu")


def test_minimum_autocast_precision_context_manager_nested_same_instance() -> None:
    # Nesting `with mp: with mp:` on the same instance works (stack-based __enter__/__exit__).
    mp = minimum_autocast_precision(torch.float32)
    with torch.autocast("cpu", torch.bfloat16, enabled=True):
        with mp:
            assert not torch.is_autocast_enabled("cpu")
            with mp:
                assert not torch.is_autocast_enabled("cpu")
            assert not torch.is_autocast_enabled("cpu")
        assert torch.is_autocast_enabled("cpu")
