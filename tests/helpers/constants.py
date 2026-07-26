"""Settings used for multiple tests."""

import torch


def _compile_supported() -> bool:
    """Whether torch.compile works here; torch 2.0 rejects python>=3.11, for example."""
    try:
        torch.compile(lambda x: x)
    except RuntimeError:
        return False
    return True


COMPILE_SUPPORTED = _compile_supported()
TORCH_VERSION = tuple(int(part) for part in torch.__version__.split(".")[:2])

# Default tolerances
TOLERANCES = dict(atol=1e-3, rtol=1e-4)
MILD_TOLERANCES = dict(atol=0.05, rtol=0.05)
STRICT_TOLERANCES = dict(atol=1e-6, rtol=1e-6)

# Batch dimensions that are typically checked
BATCH_DIMS = [[3, 5]]
