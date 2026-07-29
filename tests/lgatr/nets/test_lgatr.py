import pytest
import torch

from lgatr.layers.attention.config import SelfAttentionConfig
from lgatr.layers.mlp.config import MLPConfig
from lgatr.nets import LGATr
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

BATCH_DIMS = BATCH_DIMS[:-1]
NUM_ITEMS, IN_MV, OUT_MV, HIDDEN_MV = 8, 3, 4, 6

# GeometricBilinear needs a scalar stream, so hidden_s_channels=0 only works with
# geometric_product=False; that configuration has its own test below.
S_CHANNELS = [(0, 0, 7), (4, 5, 6)]


def _net(in_s: int, out_s: int, hidden_s: int, **kwargs) -> LGATr:
    """An LGATr with the standard test channel counts."""
    return LGATr(
        num_blocks=1,
        in_mv_channels=IN_MV,
        out_mv_channels=OUT_MV,
        hidden_mv_channels=HIDDEN_MV,
        in_s_channels=in_s,
        out_s_channels=out_s,
        hidden_s_channels=hidden_s,
        mlp=MLPConfig(),
        **kwargs,
    )


@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", S_CHANNELS)
@pytest.mark.parametrize("multi_query", [False, True])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_lgatr_shape(
    in_s_channels: int,
    out_s_channels: int,
    hidden_s_channels: int,
    multi_query: bool,
    checkpoint_blocks: bool,
) -> None:
    # LGATr's outputs match the expected shapes. The dict form of the layer configs is used here
    # to exercise Config.cast.
    net = _net(
        in_s_channels,
        out_s_channels,
        hidden_s_channels,
        attention=dict(num_heads=4, multi_query=multi_query),
        dropout_prob=0.3,
        checkpoint_blocks=checkpoint_blocks,
    )

    inputs = torch.randn(*BATCH_DIMS, NUM_ITEMS, IN_MV, 16)
    scalars = torch.randn(*BATCH_DIMS, NUM_ITEMS, in_s_channels) if in_s_channels else None
    outputs, output_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (*BATCH_DIMS, NUM_ITEMS, OUT_MV, 16)
    if out_s_channels:
        assert output_scalars.shape == (*BATCH_DIMS, NUM_ITEMS, out_s_channels)


def test_lgatr_no_scalar_stream() -> None:
    # A scalar-free LGATr runs and stays equivariant, as documented for hidden_s_channels=0.
    net = _net(
        0,
        0,
        0,
        attention=SelfAttentionConfig(num_heads=4),
        primitives=PrimitivesConfig(geometric_product=False),
    )

    inputs = torch.randn(*BATCH_DIMS, NUM_ITEMS, IN_MV, 16)
    outputs, output_scalars = net(inputs)
    assert outputs.shape == (*BATCH_DIMS, NUM_ITEMS, OUT_MV, 16)
    assert output_scalars is None

    check_pin_equivariance(net, 1, batch_dims=(*BATCH_DIMS, NUM_ITEMS, IN_MV), **MILD_TOLERANCES)


@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", S_CHANNELS)
@pytest.mark.parametrize("multi_query", [False, True])
@pytest.mark.parametrize("norm_elementwise_affine", [False, True])
@pytest.mark.parametrize("subgroup,spin", [(True, True), (False, False)])
def test_lgatr_equivariance(
    in_s_channels: int,
    out_s_channels: int,
    hidden_s_channels: int,
    multi_query: bool,
    norm_elementwise_affine: bool,
    subgroup: bool,
    spin: bool,
) -> None:
    # LGATr is equivariant under the group it is built for: Spin for the proper-orthochronous
    # subgroup, the full Pin group (including reflections) otherwise.
    net = _net(
        in_s_channels,
        out_s_channels,
        hidden_s_channels,
        attention=SelfAttentionConfig(num_heads=4, multi_query=multi_query),
        primitives=PrimitivesConfig(subgroup=subgroup),
        norm_elementwise_affine=norm_elementwise_affine,
    )

    scalars = torch.randn(*BATCH_DIMS, NUM_ITEMS, in_s_channels) if in_s_channels else None
    check_pin_equivariance(
        net,
        1,
        batch_dims=(*BATCH_DIMS, NUM_ITEMS, IN_MV),
        fn_kwargs=dict(scalars=scalars),
        spin=spin,
        **MILD_TOLERANCES,
    )


@pytest.mark.parametrize("multi_query", [False, True])
def test_lgatr_reinsert_channels(multi_query: bool) -> None:
    # reinsert_mv_channels/reinsert_s_channels reinsert input channels as additional query/key
    # features in every attention layer (multi-head and multi-query variants).
    net = LGATr(
        num_blocks=2,
        in_mv_channels=3,
        out_mv_channels=2,
        hidden_mv_channels=4,
        in_s_channels=5,
        out_s_channels=2,
        hidden_s_channels=4,
        attention=SelfAttentionConfig(num_heads=2, multi_query=multi_query),
        mlp=MLPConfig(),
        reinsert_mv_channels=(0, 2),
        reinsert_s_channels=(1, 3, 4),
    )

    inputs = torch.randn(2, 7, 3, 16)
    scalars = torch.randn(2, 7, 5)
    outputs, output_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (2, 7, 2, 16)
    assert output_scalars.shape == (2, 7, 2)


def test_lgatr_apply_casts_and_runs() -> None:
    # The _apply override forwards to nn.Module and warms the caches, so .to() casts and still runs.
    net = LGATr(
        num_blocks=1,
        in_mv_channels=2,
        out_mv_channels=1,
        hidden_mv_channels=4,
        in_s_channels=2,
        out_s_channels=2,
        hidden_s_channels=4,
        attention=SelfAttentionConfig(num_heads=2),
        mlp=MLPConfig(),
    )

    assert net.to(torch.float64) is net
    assert net.linear_in.weight.dtype == torch.float64

    inputs = torch.randn(2, 3, 2, 16, dtype=torch.float64)
    scalars = torch.randn(2, 3, 2, dtype=torch.float64)
    outputs, output_scalars = net(inputs, scalars=scalars)
    assert outputs.dtype == output_scalars.dtype == torch.float64


def test_two_lgatr_configs_coexist() -> None:
    # Two LGATr models with different PrimitivesConfig instances must coexist in one process,
    # with parameter shapes and forward outputs reflecting their respective configs.
    common = dict(
        num_blocks=1,
        in_mv_channels=2,
        out_mv_channels=1,
        hidden_mv_channels=4,
        in_s_channels=2,
        out_s_channels=2,
        hidden_s_channels=4,
        attention=SelfAttentionConfig(num_heads=2),
        mlp=MLPConfig(),
    )
    cfg_subgroup = PrimitivesConfig(subgroup=True)
    cfg_full = PrimitivesConfig(subgroup=False)
    m_sub = LGATr(primitives=cfg_subgroup, **common)
    m_full = LGATr(primitives=cfg_full, **common)

    # Linear basis count differs between the two groups.
    assert m_sub.linear_in.weight.shape[-1] == 10
    assert m_full.linear_in.weight.shape[-1] == 5

    x = torch.randn(2, 3, 2, 16)
    s = torch.randn(2, 3, 2)
    out_sub, _ = m_sub(x, s)
    out_full, _ = m_full(x, s)
    assert out_sub.shape == out_full.shape == (2, 3, 1, 16)

    # The configs are independent objects on each model.
    assert m_sub.primitives is cfg_subgroup
    assert m_full.primitives is cfg_full
