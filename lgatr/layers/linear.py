"""Pin-equivariant linear layers between multivector tensors (torch.nn.Modules)."""

import math

import torch
from torch import nn

from ..interface import embed_scalar
from ..primitives.config import gatr_config
from ..primitives.linear import equi_linear


class EquiLinear(nn.Module):
    """Linear layer.

    The forward pass maps multivector inputs with shape (..., in_channels, 16) to multivector
    outputs with shape (..., out_channels, 16) as


        .. code-block::

            outputs[..., j, y] = sum_{i, b, x} weights[j, i, b] basis_map[b, x, y] inputs[..., i, x]
            = sum_i linear(inputs[..., i, :], weights[j, i, :])

    plus an optional bias term for ``outputs[..., :, 0]`` (biases in other multivector components would
    break equivariance). Here ``basis_map`` are precomputed (see ``gatr.primitives.linear``) and weights are the
    learnable weights of this layer. The ``basis_map`` includes 5 elements if the full Lorentz group is
    considered, and 10 elements if only the fully-connected subgroup is considered.
    See the ``use_fully_connected_subgroup`` parameter in ``lgatr.primitives.config.LGATrConfig`` for details.

    If there are auxiliary input scalars, they transform under a linear layer, and mix with the
    scalar components the multivector data. Note that in this layer (and only here) the auxiliary
    scalars are optional.

    This layer supports four initialization schemes:

    - "default": preserves (or actually slightly reducing) the variance of the data in the forward pass
    - "small": variance of outputs is approximately one order of magnitude smaller than for "default"
    - "unit_scalar": outputs will be close to (1, 0, 0, ..., 0)
    - "almost_unit_scalar": similar to "unit_scalar", but with more stochasticity

    We use the "almost_unit_scalar" initialization to preprocess the second
    argument in the ``GeometricBilinears`` layer,
    and the "small" initialization to combine the different attention heads.
    All other linear layers in L-GATr use the "default" initialization.

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels
    out_mv_channels : int
        Output multivector channels
    bias : bool
        Whether a bias term is added to the scalar component of the multivector outputs
    in_s_channels : int or None
        Input scalar channels. If None, no scalars are expected nor returned.
    out_s_channels : int or None
        Output scalar channels. If None, no scalars are expected nor returned.
    initialization : {"default", "small", "unit_scalar", "almost_unit_scalar"}
            Initialization scheme, see ``EquiLinear`` description for more information.
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        in_s_channels: int | None = None,
        out_s_channels: int | None = None,
        bias: bool = True,
        initialization: str = "default",
    ) -> None:
        super().__init__()

        # Check inputs
        if initialization in ["unit_scalar", "almost_unit_scalar"]:
            assert bias, "unit_scalar and almost_unit_scalar initialization requires bias"
            if in_s_channels is None:
                raise NotImplementedError(
                    "unit_scalar and almost_unit_scalar initialization is currently only implemented for scalar inputs"
                )

        self._in_mv_channels = in_mv_channels

        # MV -> MV
        self.weight = nn.Parameter(
            torch.empty(
                (
                    out_mv_channels,
                    in_mv_channels,
                    gatr_config.num_pin_linear_basis_elements,
                )
            )
        )

        # We only need a separate bias here if that isn't already covered by the linear map from
        # scalar inputs
        self.bias = (
            nn.Parameter(torch.zeros((out_mv_channels, 1)))
            if bias and in_s_channels is None
            else None
        )

        # Scalars -> MV scalars
        self.s2mvs: nn.Linear | None
        mix_factor = 2 if gatr_config.use_fully_connected_subgroup else 1
        if in_s_channels:
            self.s2mvs = nn.Linear(in_s_channels, mix_factor * out_mv_channels, bias=bias)
        else:
            self.s2mvs = None

        # MV scalars -> scalars
        if out_s_channels:
            self.mvs2s = nn.Linear(mix_factor * in_mv_channels, out_s_channels, bias=bias)
        else:
            self.mvs2s = None

        # Scalars -> scalars
        if in_s_channels is not None and out_s_channels is not None:
            self.s2s = nn.Linear(
                in_s_channels, out_s_channels, bias=False
            )  # Bias would be duplicate
        else:
            self.s2s = None

        # Initialization
        self.reset_parameters(initialization)

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Maps input multivectors and scalars using the most general equivariant linear map.

        Parameters
        ----------
        multivectors : torch.Tensor
            Input multivectors with shape (..., in_mv_channels, 16)
        scalars : None or torch.Tensor
            Optional input scalars with shape (..., in_s_channels)

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors with shape (..., out_mv_channels, 16)
        outputs_s : None or torch.Tensor
            Output scalars with shape (..., out_s_channels)
        """

        outputs_mv = equi_linear(multivectors, self.weight)  # (..., out_channels, 16)

        if self.bias is not None:
            bias = embed_scalar(self.bias)
            outputs_mv = outputs_mv + bias

        if self.s2mvs is not None and scalars is not None:
            if gatr_config.use_fully_connected_subgroup:
                outputs_mv[..., [0, -1]] += self.s2mvs(scalars).view(
                    *outputs_mv.shape[:-2], outputs_mv.shape[-2], 2
                )
            else:
                outputs_mv[..., 0] += self.s2mvs(scalars)

        if self.mvs2s is not None:
            if gatr_config.use_fully_connected_subgroup:
                outputs_s = self.mvs2s(multivectors[..., [0, -1]].flatten(start_dim=-2))
            else:
                outputs_s = self.mvs2s(multivectors[..., 0])
            if self.s2s is not None and scalars is not None:
                outputs_s = outputs_s + self.s2s(scalars)
        else:
            outputs_s = None

        return outputs_mv, outputs_s

    def reset_parameters(
        self,
        initialization: str,
        gain: float = 1.0,
        additional_factor=None,
    ) -> None:
        """Initializes the weights of the linear layer.

        We following the initialization philosophy of `Kaiming <https://arxiv.org/pdf/1502.01852>`_,
        ensuring that the variance of the activations during the forward pass is preserved.
        Our implementation deviates from the `torch.nn.Linear default initialization <https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/linear.py#L114>`_
        to take the communication between scalar and multivector channels in our linear layer into account.
        For more information, see inline comments in the code.

        Parameters
        ----------
        initialization : {"default", "small", "unit_scalar", "almost_unit_scalar"}
            Initialization scheme, see ``EquiLinear`` description for more information.
        gain : float
            Gain factor for the activations. Should be 1.0 if previous layer has no activation,
            sqrt(2) if it has a ReLU activation, and so on. Can be computed with
            `torch.nn.init.calculate_gain()`.
        additional_factor : float
            Empirically, it has been found that slightly *decreasing* the data variance at each
            layer gives a better performance. In particular, the PyTorch default initialization uses
            an additional factor of 1/sqrt(3) (cancelling the factor of sqrt(3) that naturally
            arises when computing the bounds of a uniform initialization). A discussion of this was
            (to the best of our knowledge) never published, but see
            https://github.com/pytorch/pytorch/issues/57109 and
            https://soumith.ch/files/20141213_gplus_nninit_discussion.htm.
        """
        additional_factor = 1 / math.sqrt(3.0) if additional_factor is None else additional_factor

        # Prefactors depending on initialization scheme
        (
            mv_component_factors,
            mv_factor,
            mvs_bias_shift,
            s_factor,
        ) = self._compute_init_factors(
            initialization,
            gain,
            additional_factor,
        )

        # Following He et al, 1502.01852, we aim to preserve the variance in the forward pass.
        # A sufficient criterion for this is that the variance of the weights is given by
        # `Var[w] = gain^2 / fan`.
        # Here `gain^2` is 2 if the previous layer has a ReLU nonlinearity, 1 for the initial layer,
        # and some other value in other situations (we may not care about this too much).
        # More importantly, `fan` is the number of connections: the number of input elements that
        # get summed over to compute each output element.

        # Let us fist consider the multivector outputs.
        self._init_multivectors(mv_component_factors, mv_factor, mvs_bias_shift)

        # Then let's consider the maps to scalars.
        self._init_scalars(s_factor)

    @staticmethod
    def _compute_init_factors(
        initialization,
        gain,
        additional_factor,
    ):
        """Computes prefactors for the initialization.

        See self.reset_parameters().
        """

        assert initialization in [
            "default",
            "small",
            "unit_scalar",
            "almost_unit_scalar",
        ]

        if initialization == "default":
            mv_factor = gain * additional_factor * math.sqrt(3)
            s_factor = gain * additional_factor * math.sqrt(3)
            mvs_bias_shift = 0.0
        elif initialization == "small":
            # Change scale by a factor of 0.1 in this layer
            mv_factor = 0.1 * gain * additional_factor * math.sqrt(3)
            s_factor = 0.1 * gain * additional_factor * math.sqrt(3)
            mvs_bias_shift = 0.0
        elif initialization == "unit_scalar":
            # Change scale by a factor of 0.1 for MV outputs, and initialize bias around 1
            mv_factor = 0.1 * gain * additional_factor * math.sqrt(3)
            s_factor = gain * additional_factor * math.sqrt(3)
            mvs_bias_shift = 1.0
        elif initialization == "almost_unit_scalar":
            # Change scale by a factor of 0.5 for MV outputs, and initialize bias around 1
            mv_factor = 0.5 * gain * additional_factor * math.sqrt(3)
            s_factor = gain * additional_factor * math.sqrt(3)
            mvs_bias_shift = 1.0

        mv_component_factors = torch.ones(gatr_config.num_pin_linear_basis_elements)
        return mv_component_factors, mv_factor, mvs_bias_shift, s_factor

    def _init_multivectors(self, mv_component_factors, mv_factor, mvs_bias_shift):
        """Weight initialization for maps to multivector outputs."""

        # We have
        # `outputs[..., j, y] = sum_{i, b, x} weights[j, i, b] basis_map[b, x, y] inputs[..., i, x]`
        # The basis maps are more or less grade projections, summing over all basis elements
        # corresponds to (almost) an identity map in the GA space. The sum over `b` and `x` thus
        # does not contribute to `fan` substantially. (We may add a small ad-hoc factor later to
        # make up for this approximation.) However, there is still the sum over incoming channels,
        # and thus `fan ~ mv_in_channels`. Assuming (for now) that the previous layer contained a
        # ReLU activation, we finally have the condition `Var[w] = 2 / mv_in_channels`.
        # Since the variance of a uniform distribution between -a and a is given by
        # `Var[Uniform(-a, a)] = a^2/3`, we should set `a = gain * sqrt(3 / mv_in_channels)`.
        # In theory (see docstring).
        fan_in = self._in_mv_channels
        bound = mv_factor / math.sqrt(fan_in)
        for i, factor in enumerate(mv_component_factors):
            nn.init.uniform_(self.weight[..., i], a=-factor * bound, b=factor * bound)

        # Now let's focus on the scalar components of the multivector outputs.
        # If there are only multivector inputs, all is good. But if scalar inputs contribute them as
        # well, they contribute to the output variance as well.
        # In this case, we initialize such that the multivector inputs and the scalar inputs each
        # contribute half to the output variance.
        # We can achieve this by inspecting the basis maps and seeing that only basis element 0
        # contributes to the scalar output. Thus, we can reduce the variance of the corresponding
        # weights to give a variance of 0.5, not 1.
        if self.s2mvs is not None:
            # contribution from scalar -> mv scalar
            bound = mv_component_factors[0] * mv_factor / math.sqrt(fan_in) / math.sqrt(2)
            nn.init.uniform_(self.weight[..., [0]], a=-bound, b=bound)
            if gatr_config.use_fully_connected_subgroup:
                # contribution from scalar -> mv pseudoscalar
                bound = mv_component_factors[-1] * mv_factor / math.sqrt(fan_in) / math.sqrt(2)
                nn.init.uniform_(self.weight[..., [-1]], a=-bound, b=bound)

        # The same holds for the scalar-to-MV map, where we also just want a variance of 0.5.
        # Note: This is not properly extended to scalar and pseudoscalar outputs yet
        if self.s2mvs is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.s2mvs.weight)  # pylint:disable=protected-access
            fan_in = max(fan_in, 1)  # Since in theory we could have 0-channel scalar "data"
            bound = mv_component_factors[0] * mv_factor / math.sqrt(fan_in) / math.sqrt(2)
            nn.init.uniform_(self.s2mvs.weight, a=-bound, b=bound)

            # Bias needs to be adapted, as the overall fan in is different (need to account for MV
            # and s inputs) and we may need to account for the unit_scalar initialization scheme
            if self.s2mvs.bias is not None:
                fan_in = (
                    nn.init._calculate_fan_in_and_fan_out(self.s2mvs.weight)[0]
                    + self._in_mv_channels
                )
                bound = mv_component_factors[0] / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.s2mvs.bias, mvs_bias_shift - bound, mvs_bias_shift + bound)

    def _init_scalars(self, s_factor):
        """Weight initialization for maps to multivector outputs."""

        # If both exist, we need to account for overcounting again, and assign each a target a
        # variance of 0.5.
        # Note: This is not properly extended to scalar and pseudoscalar outputs yet
        models = []
        if self.s2s:
            models.append(self.s2s)
        if self.mvs2s:
            models.append(self.mvs2s)
        for model in models:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.weight)  # pylint:disable=protected-access
            fan_in = max(fan_in, 1)  # Since in theory we could have 0-channel scalar "data"
            bound = s_factor / math.sqrt(fan_in) / math.sqrt(len(models))
            nn.init.uniform_(model.weight, a=-bound, b=bound)
        # Bias needs to be adapted, as the overall fan in is different (need to account for MV and
        # s inputs)
        if self.mvs2s and self.mvs2s.bias is not None:
            fan_in = nn.init._calculate_fan_in_and_fan_out(self.mvs2s.weight)[0]  # pylint:disable=protected-access
            if self.s2s:
                fan_in += nn.init._calculate_fan_in_and_fan_out(self.s2s.weight)[0]  # pylint:disable=protected-access
            bound = s_factor / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.mvs2s.bias, -bound, bound)
