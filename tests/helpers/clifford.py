"""Geometric-algebra operations based on the clifford library."""

import clifford
import numpy as np
import torch

LAYOUT, _ = clifford.Cl(1, 3)


def _to_mv(array: np.ndarray) -> clifford.MultiVector:
    """Wrap a length-16 numpy array as a Pin(1, 3) multivector."""
    return clifford.MultiVector(LAYOUT, value=array)


def mv_list_to_tensor(
    multivectors: list[clifford.MultiVector],
    batch_shape: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Stack a list of multivectors into a torch tensor."""
    tensor = torch.from_numpy(np.array([mv.value for mv in multivectors])).to(torch.float32)
    if batch_shape is not None:
        tensor = tensor.reshape(*batch_shape, 16)

    return tensor


def _sample_reflection() -> clifford.MultiVector:
    """Sample a normalized Lorentz vector, i.e. a single reflection in Pin(1, 3)."""
    vector = np.zeros(16)
    vector[2:5] = np.random.normal(size=3) * 2
    norm = np.linalg.norm(vector[2:5])
    vector[1] = (np.random.uniform() - 0.5) * norm

    mv = _to_mv(vector)
    return mv / abs(mv.mag2()) ** 0.5


def _sample_pin_multivector(odd: bool) -> tuple[clifford.MultiVector, clifford.MultiVector]:
    """Sample a non-identity element of Pin(1, 3), and its inverse, as a product of reflections.

    An odd number of reflections gives an element outside Spin(1, 3), i.e. one that involves a
    parity flip; an even number gives a Spin element. The identity is never returned, so every
    equivariance check applies a non-trivial transformation.
    """
    num_reflections = np.random.choice([1, 3] if odd else [2, 4])

    multivector, inverse = 1.0, 1.0
    for _ in range(num_reflections):
        v = _sample_reflection()
        # v is normalized, so v * v == +-1 and the inverse of the product just reverses the order.
        multivector = multivector * v
        inverse = v / (v * v).value[0] * inverse

    return multivector, inverse


class RandomPinTransform:
    """Random Pin(1, 3) transform on multivector tensors of shape ``(..., 16)``.

    The action is the twisted sandwich product: ``u x u^-1`` for even ``u``, and
    ``u grade_involute(x) u^-1`` for odd ``u``. It is linear in ``x``, so it is applied as a single
    16x16 matmul; building that matrix means sandwiching the 16 basis blades, which costs the same
    no matter how large the inputs are.
    """

    def __init__(self, odd: bool = False) -> None:
        u, u_inverse = _sample_pin_multivector(odd)
        basis = [_to_mv(row) for row in np.eye(16)]
        if odd:
            basis = [e.gradeInvol() for e in basis]
        self._matrix = mv_list_to_tensor([u * e * u_inverse for e in basis])

    def sample(self, batch_dims: tuple[int, ...] | list[int]) -> torch.Tensor:
        """Draw random multivector inputs of shape ``(*batch_dims, 16)``."""
        return torch.randn(*batch_dims, 16)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[-1] == 16
        return inputs @ self._matrix
