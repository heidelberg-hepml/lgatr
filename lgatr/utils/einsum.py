"""Einsum with explicit, hardcoded contraction paths."""

import torch


def custom_einsum(equation: str, *operands: torch.Tensor, path: list[int]) -> torch.Tensor:
    """Computes einsum with a custom contraction order."""

    # Justification: For the sake of performance, we need direct access to torch's private methods.

    return torch._VF.einsum(equation, operands, path=path)
