from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LGATrConfig:
    """Configuration for global settings like the symmetry group.

    Parameters
    ----------
    use_fully_connected_subgroup : bool
        If True, model is only equivariant with respect to
        the fully connected subgroup of the Lorentz group,
        the proper orthochronous Lorentz group :math:`SO^+(1,3)`,
        which does not include parity and time reversal.
        This setting affects how the EquiLinear maps work:
        For :math:`SO^+(1,3)`, they include transitions scalars/pseudoscalars
        vectors/axialvectors and among bivectors, effectively
        treating the pseudoscalar/axialvector representations
        like another scalar/vector.
        Defaults to False, because parity-odd representations
        are usually not important in high-energy physics simulations.
    use_bivector : bool
        If False, the bivector components are set to zero after they
        are created in the GeometricBilinear layer.
        This is a toy switch to explore the effect of higher-order
        representations.
    use_geometric_product : bool
        If False, the GeometricBilinear layer is replaced
        by a EquiLinear + ScalarGatedNonlinearity layer.
        This is a toy switch to explore the effect of the geometric product.
    """

    use_fully_connected_subgroup: bool = True

    use_bivector: bool = True
    use_geometric_product: bool = True

    @property
    def num_pin_linear_basis_elements(self):
        return 10 if self.use_fully_connected_subgroup else 5


gatr_config = LGATrConfig()
