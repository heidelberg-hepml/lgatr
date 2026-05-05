import pytest
import torch

from lgatr.layers.attention.config import SelfAttentionConfig
from lgatr.layers.mlp.config import MLPConfig
from lgatr.nets import LGATr
from lgatr.primitives.config import PrimitivesConfig
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

S_CHANNELS = [(0, 0, 7), (0, 0, 0), (4, 5, 6)]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "num_items,in_mv_channels,out_mv_channels,hidden_mv_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_heads,num_blocks", [(4, 1)])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", S_CHANNELS)
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.3])
@pytest.mark.parametrize("multi_query_attention", [False, True])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
@pytest.mark.parametrize("use_fully_connected_subgroup", [True, False])
def test_lgatr_shape(
    batch_dims: list[int],
    num_items: int,
    in_mv_channels: int,
    out_mv_channels: int,
    hidden_mv_channels: int,
    num_blocks: int,
    num_heads: int,
    in_s_channels: int,
    out_s_channels: int,
    hidden_s_channels: int,
    multi_query_attention: bool,
    dropout_prob: float | None,
    checkpoint_blocks: bool,
    use_fully_connected_subgroup: bool,
) -> None:
    # LGATr's outputs match the expected shapes for all combinations of channels and config.
    inputs = torch.randn(*batch_dims, num_items, in_mv_channels, 16)
    scalars = torch.randn(*batch_dims, num_items, in_s_channels) if in_s_channels else None

    try:
        net = LGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            attention=dict(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            num_blocks=num_blocks,
            mlp=dict(),
            primitives=PrimitivesConfig(use_fully_connected_subgroup=use_fully_connected_subgroup),
            dropout_prob=dropout_prob,
            checkpoint_blocks=checkpoint_blocks,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    outputs, output_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (*batch_dims, num_items, out_mv_channels, 16)
    if out_s_channels:
        assert output_scalars.shape == (*batch_dims, num_items, out_s_channels)


@pytest.mark.parametrize("batch_dims", [(64,)])
@pytest.mark.parametrize(
    "num_items,in_mv_channels,out_mv_channels,hidden_mv_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_heads,num_blocks", [(4, 1)])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", S_CHANNELS)
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_lgatr_equivariance(
    batch_dims: tuple[int, ...],
    num_items: int,
    in_mv_channels: int,
    out_mv_channels: int,
    hidden_mv_channels: int,
    num_blocks: int,
    num_heads: int,
    in_s_channels: int,
    out_s_channels: int,
    hidden_s_channels: int,
    multi_query_attention: bool,
) -> None:
    # LGATr (full network) is Pin-equivariant.
    try:
        net = LGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            num_blocks=num_blocks,
            mlp=MLPConfig(),
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    scalars = torch.randn(*batch_dims, num_items, in_s_channels) if in_s_channels else None
    data_dims = tuple(list(batch_dims) + [num_items, in_mv_channels])
    check_pin_equivariance(
        net, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **MILD_TOLERANCES
    )


@pytest.mark.parametrize("batch_dims", [(64,)])
@pytest.mark.parametrize(
    "num_items,in_mv_channels,out_mv_channels,hidden_mv_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_heads,num_blocks", [(4, 1)])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", [(4, 5, 6)])
def test_lgatr_equivariance_compiled(
    batch_dims: tuple[int, ...],
    num_items: int,
    in_mv_channels: int,
    out_mv_channels: int,
    hidden_mv_channels: int,
    num_blocks: int,
    num_heads: int,
    in_s_channels: int,
    out_s_channels: int,
    hidden_s_channels: int,
    compile: bool = True,
) -> None:
    # torch.compile-wrapped LGATr still preserves shapes and Pin-equivariance.
    net = LGATr(
        in_mv_channels=in_mv_channels,
        out_mv_channels=out_mv_channels,
        hidden_mv_channels=hidden_mv_channels,
        in_s_channels=in_s_channels,
        out_s_channels=out_s_channels,
        hidden_s_channels=hidden_s_channels,
        attention=SelfAttentionConfig(num_heads=num_heads),
        num_blocks=num_blocks,
        mlp=MLPConfig(),
        compile=compile,
    )

    scalars = torch.randn(*batch_dims, num_items, in_s_channels)
    data_dims = tuple(list(batch_dims) + [num_items, in_mv_channels])
    check_pin_equivariance(
        net, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **MILD_TOLERANCES
    )


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
    cfg_subgroup = PrimitivesConfig(use_fully_connected_subgroup=True)
    cfg_full = PrimitivesConfig(use_fully_connected_subgroup=False)
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
