from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class GATrConfig:
    """
    Parameters
    ----------
    use_fully_connected_subgroup : bool
        If True, model is only equivariant with respect to
        the fully connected subgroup of the Lorentz group,
        the proper orthochronous Lorentz group SO^+(1,3),
        which does not include parity and time reversal.
        This setting affects how the EquiLinear maps work:
        For SO^+(1,3), they include transitions scalars/pseudoscalars
        vectors/axialvectors and among bivectors, effectively
        treating the pseudoscalar/axialvector representations
        like another scalar/vector.
        Defaults to False, because parity-odd representations
        are usually not important in high-energy physics simulations.
    mix_pseudoscalar_into_scalar : bool
        If True, the pseudoscalar part of the multivector mixes
        with the pure-scalar channels in the EquiLinear layer.
        This is a technical aspect of how EquiLinear maps work,
        and only makes sense it use_fully_connected_subgroup=True.
        Attention: The combination use_fully_connected_subgroup=False
        and mix_pseudoscalar_into_scalar=True does not make sense,
        you are only equivariant w.r.t. the fully connected subgroup
        if you choose these settings.
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
    mix_pseudoscalar_into_scalar: bool = True

    use_bivector: bool = True
    use_geometric_product: bool = True

    @property
    def num_pin_linear_basis_elements(self):
        return 10 if self.use_fully_connected_subgroup else 5

    @classmethod
    def cast(cls, config: Any):
        """Casts an object as GATrConfig."""
        if isinstance(config, GATrConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")


gatr_config = GATrConfig()
