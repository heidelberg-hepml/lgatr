"""Global configuration for the L-GATr primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LGATrConfig:
    """Global configuration for the symmetry group and bilinear-layer toggles.

    Parameters
    ----------
    use_fully_connected_subgroup
        If True, model is only equivariant with respect to
        the fully connected subgroup of the Lorentz group,
        the proper orthochronous Lorentz group :math:`SO^+(1,3)`,
        which does not include parity and time reversal.
        This setting affects how the EquiLinear maps work:
        For :math:`SO^+(1,3)`, they include transitions scalars/pseudoscalars
        vectors/axialvectors and among bivectors, effectively
        treating the pseudoscalar/axialvector representations
        like another scalar/vector.
        Defaults to True, because parity-odd representations
        are usually not important in high-energy physics simulations.
    use_bivector
        If False, the bivector components are set to zero after they are created in the
        :class:`GeometricBilinear` layer. This is a toy switch to explore the effect of
        higher-order representations.
    use_geometric_product
        If False, the :class:`GeometricBilinear` layer is replaced by a sequence of
        :class:`EquiLinear` and :class:`ScalarGatedNonlinearity` layers. This is a toy switch to
        explore the effect of the geometric product.
    sparse
        If True, route :func:`equi_linear` and :func:`geometric_product` through gather-and-reduce
        kernels that exploit the basis sparsity (basis tensors are ~1% / ~6% nonzero, so the
        dense path otherwise spends most of its FLOPs on zero entries). The sparse path uses
        less optimized kernels (no fused BLAS GEMM), so it can be slower than dense for small
        batches/channels despite the lower flop count. Outputs match the dense path within
        standard test tolerances but are not bit-identical. Defaults to False for backward
        compatibility (existing checkpoints expect dense numerics).
    triton
        If True, route :func:`equi_linear` and :func:`geometric_product` through fused Triton
        kernels when the inputs live on a CUDA device with a supported dtype, falling through to
        the ``sparse`` dispatch otherwise. Defaults to False for backward compatibility.
    """

    use_fully_connected_subgroup: bool = True

    use_bivector: bool = True
    use_geometric_product: bool = True

    sparse: bool = False
    triton: bool = False

    @property
    def num_pin_linear_basis_elements(self) -> int:
        return 10 if self.use_fully_connected_subgroup else 5


gatr_config = LGATrConfig()
