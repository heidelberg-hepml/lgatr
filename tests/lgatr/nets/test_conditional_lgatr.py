import pytest
import torch

from lgatr.layers import CrossAttentionConfig, MLPConfig, SelfAttentionConfig
from lgatr.nets import ConditionalLGATr
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

BATCH_DIMS = BATCH_DIMS[:-1]
NUM_ITEMS, NUM_ITEMS_COND = 2, 9
IN_MV, MV_COND, HIDDEN_MV, OUT_MV = 7, 11, 9, 8
HIDDEN_S, OUT_S = 4, 5

# GeometricBilinear needs a scalar stream, so the scalar-free configuration is not constructible
# here; see tests/lgatr/nets/test_lgatr.py::test_lgatr_no_scalar_stream.
S_CHANNELS = [(3, 5), (2, 2)]


def _net(in_s: int, s_cond: int, num_heads: int = 4, **kwargs) -> ConditionalLGATr:
    """A ConditionalLGATr with the standard test channel counts."""
    return ConditionalLGATr(
        num_blocks=1,
        in_mv_channels=IN_MV,
        out_mv_channels=OUT_MV,
        hidden_mv_channels=HIDDEN_MV,
        mv_channels_cond=MV_COND,
        in_s_channels=in_s,
        out_s_channels=OUT_S,
        hidden_s_channels=HIDDEN_S,
        s_channels_cond=s_cond,
        mlp=MLPConfig(),
        **kwargs,
    )


@pytest.mark.parametrize("in_s_channels,s_channels_cond", S_CHANNELS)
@pytest.mark.parametrize("multi_query", [False, True])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_conditional_gatr_shape(
    in_s_channels: int, s_channels_cond: int, multi_query: bool, checkpoint_blocks: bool
) -> None:
    # ConditionalLGATr's outputs match the expected shapes. The dict form of the layer configs is
    # used here to exercise Config.cast.
    net = _net(
        in_s_channels,
        s_channels_cond,
        attention=dict(num_heads=4, multi_query=multi_query),
        crossattention=dict(num_heads=4, multi_query=multi_query),
        checkpoint_blocks=checkpoint_blocks,
    )

    inputs = torch.randn(*BATCH_DIMS, NUM_ITEMS, IN_MV, 16)
    scalars = torch.randn(*BATCH_DIMS, NUM_ITEMS, in_s_channels)
    mv_cond = torch.randn(*BATCH_DIMS, NUM_ITEMS_COND, MV_COND, 16)
    s_cond = torch.randn(*BATCH_DIMS, NUM_ITEMS_COND, s_channels_cond)

    outputs, output_scalars = net(
        inputs, scalars=scalars, multivectors_cond=mv_cond, scalars_cond=s_cond
    )

    assert outputs.shape == (*BATCH_DIMS, NUM_ITEMS, OUT_MV, 16)
    assert output_scalars.shape == (*BATCH_DIMS, NUM_ITEMS, OUT_S)


@pytest.mark.parametrize("in_s_channels,s_channels_cond", S_CHANNELS)
@pytest.mark.parametrize("multi_query", [False, True])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
@pytest.mark.parametrize("subgroup,spin", [(True, True), (False, False)])
def test_conditional_gatr_equivariance(
    in_s_channels: int,
    s_channels_cond: int,
    multi_query: bool,
    norm_elementwise_affine: bool,
    subgroup: bool,
    spin: bool,
) -> None:
    # ConditionalLGATr is equivariant in both inputs and condition, under the group it is built
    # for: Spin for the proper-orthochronous subgroup, the full Pin group otherwise.
    net = _net(
        in_s_channels,
        s_channels_cond,
        attention=SelfAttentionConfig(num_heads=4, multi_query=multi_query),
        crossattention=CrossAttentionConfig(num_heads=4, multi_query=multi_query),
        primitives=PrimitivesConfig(subgroup=subgroup),
        norm_elementwise_affine=norm_elementwise_affine,
    )

    scalars = torch.randn(*BATCH_DIMS, NUM_ITEMS, in_s_channels)
    scalars_cond = torch.randn(*BATCH_DIMS, NUM_ITEMS_COND, s_channels_cond)
    data_dims = [
        (*BATCH_DIMS, NUM_ITEMS, IN_MV),
        (*BATCH_DIMS, NUM_ITEMS_COND, MV_COND),
    ]
    check_pin_equivariance(
        net,
        2,
        batch_dims=data_dims,
        fn_kwargs=dict(scalars=scalars, scalars_cond=scalars_cond),
        spin=spin,
        **MILD_TOLERANCES,
    )


def test_conditional_lgatr_apply_casts_and_runs() -> None:
    # The _apply override forwards to nn.Module and warms the caches, so .to() casts and still runs.
    net = ConditionalLGATr(
        num_blocks=1,
        in_mv_channels=2,
        out_mv_channels=1,
        hidden_mv_channels=4,
        mv_channels_cond=3,
        in_s_channels=2,
        out_s_channels=2,
        hidden_s_channels=4,
        s_channels_cond=2,
        attention=SelfAttentionConfig(num_heads=2),
        crossattention=CrossAttentionConfig(num_heads=2),
        mlp=MLPConfig(),
    )

    assert net.to(torch.float64) is net
    assert net.linear_in.weight.dtype == torch.float64

    inputs = torch.randn(2, 3, 2, 16, dtype=torch.float64)
    inputs_cond = torch.randn(2, 5, 3, 16, dtype=torch.float64)
    scalars = torch.randn(2, 3, 2, dtype=torch.float64)
    scalars_cond = torch.randn(2, 5, 2, dtype=torch.float64)
    outputs, output_scalars = net(inputs, inputs_cond, scalars=scalars, scalars_cond=scalars_cond)
    assert outputs.dtype == output_scalars.dtype == torch.float64
