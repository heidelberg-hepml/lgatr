"""LGATr primitives configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.config import cast_config


@dataclass
class PrimitivesConfig:
    """Symmetry-group and bilinear-layer toggles for an L-GATr model.

    A :class:`PrimitivesConfig` is passed to :class:`~lgatr.LGATr` (and to the layers and
    primitive functions it contains) at construction time. Multiple models with different
    configs can coexist in the same process.

    Parameters
    ----------
    subgroup
        If True, the model is only equivariant with respect to the connected subgroup of the
        Lorentz group, the proper orthochronous Lorentz group :math:`SO^+(1,3)`, which excludes
        parity and time reversal. This setting affects how the EquiLinear maps work: for
        :math:`SO^+(1,3)` they additionally mix scalars with pseudoscalars, vectors with
        axialvectors, and among bivectors, effectively treating the pseudoscalar and axialvector
        representations like another scalar and vector. Defaults to True, because parity-odd
        representations are usually not important in high-energy physics simulations.
    bivector
        If False, the bivector components are set to zero after they are created in the
        :class:`GeometricBilinear` layer. This is a toy switch to explore the effect of
        higher-order representations.
    geometric_product
        If False, the :class:`GeometricBilinear` layer is replaced by a
        :class:`ScalarGatedNonlinearity` followed by an :class:`EquiLinear` layer. This is a toy
        switch to explore the effect of the geometric product.
    sparse_gp
        If True, route :func:`geometric_product` through the gather-and-reduce kernel that exploits
        the basis sparsity (the dense path otherwise spends most of its FLOPs on zero entries).
        Under ``torch.compile`` this is both faster and far lighter than the dense product.
    sparse_linear
        If True, route :func:`equi_linear` through the per-grade kernel that exploits the basis
        sparsity. This has fewer FLOPs than the dense path but uses less optimized kernels (no
        single fused BLAS GEMM), so on FLOP-rich GPUs (e.g. H100) it is typically slower and
        heavier than dense; it mainly helps on FLOP-bound hardware. Sparse outputs match the dense
        path within standard test tolerances but are not bit-identical.
    """

    subgroup: bool = True

    bivector: bool = True
    geometric_product: bool = True

    sparse_gp: bool = True
    sparse_linear: bool = False

    @property
    def num_pin_linear_basis_elements(self) -> int:
        """Number of equivariant linear basis elements (10 for the subgroup, 5 for full Lorentz)."""
        return 10 if self.subgroup else 5

    @classmethod
    def cast(cls, config: Any) -> PrimitivesConfig:
        """Cast a :class:`PrimitivesConfig` or mapping to a :class:`PrimitivesConfig`."""
        return cast_config(cls, config)
