"""Geometric-algebra operations based on the clifford library."""

import clifford
import numpy as np
import torch

LAYOUT, BLADES = clifford.Cl(1, 3)


def np_to_mv(array: np.ndarray) -> clifford.MultiVector:
    """Convert a numpy array to a Pin(1, 3) multivector."""
    return clifford.MultiVector(LAYOUT, value=array)


def tensor_to_mv(tensor: torch.Tensor) -> clifford.MultiVector:
    """Convert a torch tensor to a Pin(1, 3) multivector."""
    return np_to_mv(tensor.detach().cpu().numpy())


def tensor_to_mv_list(tensor: torch.Tensor) -> list[clifford.MultiVector]:
    """Flatten a torch tensor of shape ``(..., 16)`` to a list of multivectors."""
    tensor = tensor.reshape((-1, 16))
    mv_list = [tensor_to_mv(x) for x in tensor]

    return mv_list


def mv_list_to_tensor(
    multivectors: list[clifford.MultiVector],
    batch_shape: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Stack a list of multivectors into a torch tensor."""
    tensor = torch.from_numpy(np.array([mv.value for mv in multivectors])).to(torch.float32)
    if batch_shape is not None:
        tensor = tensor.reshape(*batch_shape, 16)

    return tensor


def sample_pin_multivector(
    spin: bool = False, rng: np.random.Generator | None = None
) -> clifford.MultiVector:
    """Sample from the Pin(1, 3) group as a product of reflections."""

    if rng is None:
        rng = np.random.default_rng()

    # Sample number of reflections we want to multiply
    if spin:
        i = np.random.randint(3) * 2
    else:
        i = np.random.randint(5)

    # If no reflections, just return unit scalar
    if i == 0:
        return BLADES[""]

    multivector = 1.0
    for _ in range(i):
        # Sample reflection vector
        vector = np.zeros(16)
        vector[2:5] = rng.normal(size=3) * 2
        norm = np.linalg.norm(vector[2:5])
        vector[1] = (rng.uniform(size=1)[0] - 0.5) * norm

        vector_mv = np_to_mv(vector)
        vector_mv = vector_mv / abs(vector_mv.mag2()) ** 0.5

        # Multiply together (geometric product)
        multivector = multivector * vector_mv

    return multivector


def get_parity(mv: clifford.MultiVector) -> bool:
    """Return True if ``mv`` is pure-odd-grade, False if pure-even-grade.

    Raises
    ------
    RuntimeError
        If ``mv`` is mixed-grade.
    """
    if mv == mv.even:
        return False
    if mv == mv.odd:
        return True
    raise RuntimeError(f"Mixed-grade multivector: {mv}")


def sandwich(u: clifford.MultiVector, x: clifford.MultiVector) -> clifford.MultiVector:
    """Compute the sandwich product ``(-1)^(grade(u) * grade(x)) u x u^{-1}``.

    If ``u`` has odd grades, this equals ``u * grade_involute(x) * u^{-1}``; if ``u`` has even
    grades, this equals ``u * x * u^{-1}``.
    """

    if get_parity(u):
        return u * x.gradeInvol() * u.shirokov_inverse()

    return u * x * u.shirokov_inverse()


class SlowRandomPinTransform:
    """Random Pin transform on a multivector torch tensor.

    Slow; only used for testing. Breaks the computational graph.
    """

    def __init__(self, spin: bool = False, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        self._u = sample_pin_multivector(spin, rng)
        self._u_inverse = self._u.shirokov_inverse()

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the Pin transformation to multivector inputs of shape ``(..., 16)``."""
        # Input shape
        assert inputs.shape[-1] == 16
        batch_dims = inputs.shape[:-1]

        # Convert inputs to list of multivectors
        inputs_mv = tensor_to_mv_list(inputs)

        # Transform
        outputs_mv = [sandwich(self._u, x) for x in inputs_mv]

        # Back to tensor
        outputs = mv_list_to_tensor(outputs_mv, batch_shape=batch_dims)

        return outputs
