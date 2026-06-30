"""Pin inputs to a minimum autocast precision; usable as a decorator."""

from collections.abc import Callable
from functools import wraps
from itertools import chain
from typing import Any, Literal

import torch

# Toggled by the naive_amp context manager; read at call time so torch.compile constant-folds it.
_NAIVE_AMP = False


class naive_amp:
    """Disable all :class:`minimum_autocast_precision` pinning inside the block.

    While active, the fp32 precision islands created by the :class:`minimum_autocast_precision`
    decorator are bypassed and the wrapped ops run in the surrounding autocast dtype (e.g. bf16).
    Restores the previous state on exit; safe to nest.

    Parameters
    ----------
    enabled
        Whether to enable naive-AMP mode. ``False`` leaves the current state untouched, making
        ``naive_amp(False)`` a no-op that still nests cleanly (it never overrides an outer
        ``naive_amp``).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._prev: list[bool] = []

    def __enter__(self) -> "naive_amp":
        global _NAIVE_AMP
        if self.enabled:
            self._prev.append(_NAIVE_AMP)
            _NAIVE_AMP = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        global _NAIVE_AMP
        if self.enabled:
            _NAIVE_AMP = self._prev.pop()
        return False


class minimum_autocast_precision:
    """Pin tensors to a minimum precision inside autocast regions.

    Used as a decorator: ``@minimum_autocast_precision(torch.float32)`` on a function definition.
    Inside autocast-enabled regions, floating-point inputs below ``min_dtype`` are cast up to
    ``min_dtype``, autocast is disabled for the call, and outputs are optionally cast per the
    ``output`` argument. Outside autocast regions the decorator is a no-op.

    The :class:`naive_amp` context manager turns the decorator into a no-op, letting the wrapped
    ops run in the surrounding autocast dtype instead.

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
        Specifies which dtype the outputs should be cast to. Only floating-point tensor outputs
        are affected. If ``"low"`` (default), the lowest-precision input dtype is used. If
        ``None``, outputs are not modified. If ``"high"``, ``min_dtype`` or the highest-precision
        input dtype is used (whichever is higher). If a ``torch.dtype``, that dtype is used.
    which_args
        Positional argument indices to modify. If None, all positional arguments are modified
        (subject to the type / dtype filter above).
    which_kwargs
        Keyword argument names to modify. If None, all keyword arguments are modified (subject to
        the type / dtype filter above).
    """

    def __init__(
        self,
        min_dtype: torch.dtype = torch.float32,
        output: Literal["low", "high"] | torch.dtype | None = "low",
        which_args: list[int] | None = None,
        which_kwargs: list[str] | None = None,
    ) -> None:
        self.min_dtype = min_dtype
        self.output = output
        self.which_args = which_args
        self.which_kwargs = which_kwargs

    def cast(self, var: Any) -> Any:
        """Upcast a floating-point tensor to at least ``min_dtype``."""
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
            # Skip in naive-AMP mode (run in the autocast dtype), or outside autocast regions.
            if _NAIVE_AMP or not (
                torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")
            ):
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
            if not in_dtypes:
                # No floating-point inputs to derive "low"/"high" from; nothing to cast back to.
                return outputs
            # Plain loop instead of min/max(..., key=lambda) to avoid graph breaks in torch.compile
            if self.output == "low":
                candidates = [self.min_dtype] + in_dtypes
                out_dtype = candidates[0]
                for dt in candidates[1:]:
                    if torch.finfo(dt).bits < torch.finfo(out_dtype).bits:
                        out_dtype = dt
            else:
                out_dtype = in_dtypes[0]
                for dt in in_dtypes[1:]:
                    if torch.finfo(dt).bits > torch.finfo(out_dtype).bits:
                        out_dtype = dt
        else:
            out_dtype = self.output
        if isinstance(outputs, tuple):
            return tuple(self._cast_out(val, out_dtype) for val in outputs)
        return self._cast_out(outputs, out_dtype)
