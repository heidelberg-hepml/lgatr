"""Pin inputs to a minimum autocast precision; usable as decorator or context manager."""

from collections.abc import Callable
from contextlib import ExitStack
from functools import wraps
from itertools import chain
from typing import Any, Literal

import torch


class minimum_autocast_precision:
    """Pin tensors to a minimum precision inside autocast regions.

    Usable in two forms:

    - As a decorator: ``@minimum_autocast_precision(torch.float32)`` on a function definition.
      Inside autocast-enabled regions, floating-point inputs below ``min_dtype`` are cast up to
      ``min_dtype``, autocast is disabled for the call, and outputs are optionally cast per the
      ``output`` argument. Outside autocast regions the decorator is a no-op.
    - As a context manager: ``with minimum_autocast_precision(torch.float32) as mp:`` disables
      both CUDA and CPU autocast inside the block. Use ``mp.cast(tensor)`` to upcast individual
      tensors to at least ``min_dtype``.

    Only floating-point tensors are modified — non-tensors, integer tensors, and boolean tensors
    are left alone.

    Note: AMP is enabled separately for CPU and CUDA. This may behave unexpectedly when both
    devices are used and only one of them has AMP enabled. Instances are not thread-safe; share
    one per thread if used concurrently.

    Parameters
    ----------
    min_dtype
        Minimum dtype.
    output
        Decorator-only. Specifies which dtype the outputs should be cast to. Only floating-point
        tensor outputs are affected. If None, outputs are not modified. If ``"low"``, the
        lowest-precision input dtype is used. If ``"high"``, ``min_dtype`` or the highest-precision
        input dtype is used (whichever is higher). If a ``torch.dtype``, that dtype is used.
    which_args
        Decorator-only. Positional argument indices to modify. If None, all positional arguments
        are modified (subject to the type / dtype filter above).
    which_kwargs
        Decorator-only. Keyword argument names to modify. If None, all keyword arguments are
        modified (subject to the type / dtype filter above).
    """

    def __init__(
        self,
        min_dtype: torch.dtype = torch.float32,
        output: Literal["low", "high"] | torch.dtype | None = None,
        which_args: list[int] | None = None,
        which_kwargs: list[str] | None = None,
    ) -> None:
        self.min_dtype = min_dtype
        self.output = output
        self.which_args = which_args
        self.which_kwargs = which_kwargs
        self._stack: list[ExitStack] = []

    def cast(self, var: Any) -> Any:
        """Upcast a floating-point tensor to ``min_dtype`` (regardless of autocast state)."""
        if not isinstance(var, torch.Tensor):
            return var
        if not var.dtype.is_floating_point:
            return var
        if torch.finfo(var.dtype).bits >= torch.finfo(self.min_dtype).bits:
            return var
        return var.to(self.min_dtype)

    def _cast_out(self, var: Any, dtype: torch.dtype) -> Any:
        """Cast a single output to the requested dtype."""
        if not isinstance(var, torch.Tensor):
            return var
        if not var.dtype.is_floating_point:
            return var
        return var.to(dtype)

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def decorated_func(*args: Any, **kwargs: Any):
            # Only change dtypes in autocast-enabled regions
            if not (torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")):
                return func(*args, **kwargs)
            # Cast inputs to at least min_dtype
            mod_args = [
                self.cast(arg)
                for i, arg in enumerate(args)
                if self.which_args is None or i in self.which_args
            ]
            mod_kwargs = {
                key: self.cast(val)
                for key, val in kwargs.items()
                if self.which_kwargs is None or key in self.which_kwargs
            }
            # Fresh contexts (not `with self:`) — keeps the decorator re-entrant-safe.
            with (
                torch.autocast(device_type="cuda", enabled=False),
                torch.autocast(device_type="cpu", enabled=False),
            ):
                outputs = func(*mod_args, **mod_kwargs)
            return self._apply_output_dtype(outputs, args, kwargs)

        return decorated_func

    def _apply_output_dtype(self, outputs: Any, args: tuple, kwargs: dict) -> Any:
        """Cast outputs per the ``output`` mode; see class docstring."""
        if self.output is None:
            return outputs
        if self.output in ["low", "high"]:
            in_dtypes = [
                arg.dtype
                for arg in chain(args, kwargs.values())
                if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point
            ]
            assert len(in_dtypes)
            if self.output == "low":
                out_dtype = min([self.min_dtype] + in_dtypes, key=lambda dt: torch.finfo(dt).bits)
            else:
                out_dtype = max(in_dtypes, key=lambda dt: torch.finfo(dt).bits)
        else:
            out_dtype = self.output
        if isinstance(outputs, tuple):
            return (self._cast_out(val, out_dtype) for val in outputs)
        return self._cast_out(outputs, out_dtype)

    def __enter__(self) -> "minimum_autocast_precision":
        # Stacked so the same instance can be re-entered (nested `with`); pop_all transfers
        # ownership only after both inner contexts entered cleanly.
        with ExitStack() as guard:
            guard.enter_context(torch.autocast(device_type="cuda", enabled=False))
            guard.enter_context(torch.autocast(device_type="cpu", enabled=False))
            self._stack.append(guard.pop_all())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self._stack.pop().__exit__(exc_type, exc_val, exc_tb)
