"""Misc utilities: residual addition and an autocast-precision decorator."""

from collections.abc import Callable
from functools import wraps
from itertools import chain
from typing import Any, Literal

import torch


def residual_add(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    """Residual addition that propagates None.

    Returns None if both operands are None, the non-None operand if exactly one is None, and
    ``a + b`` otherwise. Used by transformer blocks to add residual connections on the optional
    scalar stream without special-casing each call site.
    """
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def minimum_autocast_precision(
    min_dtype: torch.dtype = torch.float32,
    output: Literal["low", "high"] | torch.dtype | None = None,
    which_args: list[int] | None = None,
    which_kwargs: list[str] | None = None,
) -> Callable:
    """Decorator that ensures input tensors are autocast to a minimum precision.

    Only has an effect inside autocast-enabled regions; otherwise the decorated function is
    unchanged. Only floating-point inputs are modified — non-tensors, integer tensors, and boolean
    tensors are left alone.

    Note: AMP is enabled separately for CPU and CUDA. This decorator may behave unexpectedly when
    both devices are used and only one of them has AMP enabled.

    Parameters
    ----------
    min_dtype
        Minimum dtype.
    output
        Specifies which dtype the outputs should be cast to. Only floating-point tensor outputs
        are affected. If None, outputs are not modified. If ``"low"``, the lowest-precision input
        dtype is used. If ``"high"``, ``min_dtype`` or the highest-precision input dtype is used
        (whichever is higher). If a ``torch.dtype``, that dtype is used.
    which_args
        Positional argument indices to modify. If None, all positional arguments are modified
        (subject to the type / dtype filter above).
    which_kwargs
        Keyword argument names to modify. If None, all keyword arguments are modified (subject to
        the type / dtype filter above).

    Returns
    -------
    decorator
        Decorator that casts tensor inputs to the given minimum precision.
    """

    def decorator(func: Callable) -> Callable:
        """Decorator that casts input tensors to minimum precision."""

        def _cast_in(var: Any) -> Any:
            """Cast a single input to at least ``min_dtype`` precision."""
            if not isinstance(var, torch.Tensor):
                return var
            if not var.dtype.is_floating_point:
                return var
            if torch.finfo(var.dtype).bits >= torch.finfo(min_dtype).bits:
                return var
            return var.to(min_dtype)

        def _cast_out(var: Any, dtype: torch.dtype) -> Any:
            """Cast a single output to the requested dtype."""
            if not isinstance(var, torch.Tensor):
                return var
            if not var.dtype.is_floating_point:
                return var
            return var.to(dtype)

        @wraps(func)
        def decorated_func(*args: Any, **kwargs: Any):
            # Only change dtypes in autocast-enabled regions
            if not (torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()):
                # NB: torch.is_autocast_enabled() only checks for GPU autocast
                # See https://github.com/pytorch/pytorch/issues/110966
                return func(*args, **kwargs)
            # Cast inputs to at least 32 bit
            mod_args = [
                _cast_in(arg) for i, arg in enumerate(args) if which_args is None or i in which_args
            ]
            mod_kwargs = {
                key: _cast_in(val)
                for key, val in kwargs.items()
                if which_kwargs is None or key in which_kwargs
            }
            # Call function w/o autocast enabled
            with (
                torch.autocast(device_type="cuda", enabled=False),
                torch.autocast(device_type="cpu", enabled=False),
            ):
                outputs = func(*mod_args, **mod_kwargs)
            # Cast outputs to correct dtype
            if output is None:
                return outputs
            if output in ["low", "high"]:
                in_dtypes = [
                    arg.dtype
                    for arg in chain(args, kwargs.values())
                    if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point
                ]
                assert len(in_dtypes)
                if output == "low":
                    out_dtype = min([min_dtype] + in_dtypes, key=lambda dt: torch.finfo(dt).bits)
                else:
                    out_dtype = max(in_dtypes, key=lambda dt: torch.finfo(dt).bits)
            else:
                out_dtype = output
            if isinstance(outputs, tuple):
                return (_cast_out(val, out_dtype) for val in outputs)
            else:
                return _cast_out(outputs, out_dtype)

        return decorated_func

    return decorator
