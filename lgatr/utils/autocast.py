"""Pin inputs to a minimum autocast precision; usable as a decorator."""

from collections.abc import Callable
from functools import wraps
from itertools import chain
from typing import Any, Literal

import torch

try:
    torch.is_autocast_enabled("cpu")

    def _autocast_active() -> bool:
        """Whether CPU or CUDA autocast is enabled."""
        return torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")

    def autocast_dtype(device_type: str = "cuda") -> torch.dtype:
        """Dtype that autocast would cast to on ``device_type``."""
        return torch.get_autocast_dtype(device_type)

except TypeError:  # pragma: no cover - torch<2.4 has no device_type argument

    def _autocast_active() -> bool:
        """Whether CPU or CUDA autocast is enabled."""
        return torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()

    def autocast_dtype(device_type: str = "cuda") -> torch.dtype:
        """Dtype that autocast would cast to on ``device_type``."""
        if device_type == "cpu":
            return torch.get_autocast_cpu_dtype()
        return torch.get_autocast_gpu_dtype()


class minimum_autocast_precision:
    """Pin tensors to a minimum precision inside autocast regions.

    Used as a decorator: ``@minimum_autocast_precision(torch.float32)`` on a function definition.
    Inside autocast-enabled regions, floating-point inputs below ``min_dtype`` are cast up to
    ``min_dtype``, autocast is disabled for the call, and outputs are optionally cast per the
    ``output`` argument. Outside autocast regions the decorator is a no-op.

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
        are affected. If ``"low"`` (default), the lowest precision among ``min_dtype`` and the
        input dtypes is used. If ``"high"``, the highest-precision input dtype is used. If
        ``None``, outputs are not modified. If a ``torch.dtype``, that dtype is used. In the
        ``"low"`` and ``"high"`` modes, outputs are left alone when there are no floating-point
        inputs to derive a dtype from.
    """

    def __init__(
        self,
        min_dtype: torch.dtype = torch.float32,
        output: Literal["low", "high"] | torch.dtype | None = "low",
    ) -> None:
        self.min_dtype = min_dtype
        self.output = output

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
            # Skip outside autocast regions.
            if not _autocast_active():
                return func(*args, **kwargs)
            # Cast inputs to at least min_dtype
            mod_args = [self.cast(arg) for arg in args]
            mod_kwargs = {key: self.cast(val) for key, val in kwargs.items()}
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
