"""torch.compile smoke tests for all four networks.

All cases use the same tiny shapes, so after the first compilation the rest hit the dynamo cache.
"""

import pytest
import torch

from lgatr.layers import CrossAttentionConfig, MLPConfig, SelfAttentionConfig
from lgatr.nets import ConditionalLGATr, ConditionalLGATrSlim, LGATr, LGATrSlim
from tests.helpers import COMPILE_SUPPORTED, STRICT_TOLERANCES, TORCH_VERSION

pytestmark = pytest.mark.skipif(
    not COMPILE_SUPPORTED or TORCH_VERSION < (2, 3),
    reason="inductor cannot codegen the dynamic-shape sdpa scale before torch 2.3",
)

BATCH, N, N_COND = 2, 3, 3
IN_C, OUT_C, HIDDEN_C = 2, 2, 4

GA_KWARGS = dict(
    num_blocks=1,
    in_mv_channels=IN_C,
    out_mv_channels=OUT_C,
    hidden_mv_channels=HIDDEN_C,
    in_s_channels=IN_C,
    out_s_channels=OUT_C,
    hidden_s_channels=HIDDEN_C,
    attention=SelfAttentionConfig(num_heads=2),
    mlp=MLPConfig(),
)
SLIM_KWARGS = dict(
    num_blocks=1,
    in_v_channels=IN_C,
    out_v_channels=OUT_C,
    hidden_v_channels=16,
    in_s_channels=IN_C,
    out_s_channels=OUT_C,
    hidden_s_channels=8,
    num_heads=2,
)


def _lgatr(compile: bool, **kwargs):
    return LGATr(compile=compile, **GA_KWARGS, **kwargs), (
        torch.randn(BATCH, N, IN_C, 16),
        torch.randn(BATCH, N, IN_C),
    )


def _conditional_lgatr(compile: bool, **kwargs):
    net = ConditionalLGATr(
        compile=compile,
        mv_channels_cond=IN_C,
        s_channels_cond=IN_C,
        crossattention=CrossAttentionConfig(num_heads=2),
        **GA_KWARGS,
        **kwargs,
    )
    return net, (
        torch.randn(BATCH, N, IN_C, 16),
        torch.randn(BATCH, N_COND, IN_C, 16),
        torch.randn(BATCH, N, IN_C),
        torch.randn(BATCH, N_COND, IN_C),
    )


def _slim(compile: bool, **kwargs):
    return LGATrSlim(compile=compile, **SLIM_KWARGS, **kwargs), (
        torch.randn(BATCH, N, IN_C, 4),
        torch.randn(BATCH, N, IN_C),
    )


def _conditional_slim(compile: bool, **kwargs):
    net = ConditionalLGATrSlim(
        compile=compile, v_channels_cond=IN_C, s_channels_cond=IN_C, **SLIM_KWARGS, **kwargs
    )
    return net, (
        torch.randn(BATCH, N, IN_C, 4),
        torch.randn(BATCH, N_COND, IN_C, 4),
        torch.randn(BATCH, N, IN_C),
        torch.randn(BATCH, N_COND, IN_C),
    )


BUILDERS = {
    "lgatr": _lgatr,
    "conditional_lgatr": _conditional_lgatr,
    "slim": _slim,
    "conditional_slim": _conditional_slim,
}


@pytest.mark.parametrize("name", list(BUILDERS))
def test_compiled_matches_eager(name: str) -> None:
    # A compiled network produces the same outputs as the same weights run eagerly. Equivariance
    # itself is covered eagerly in the per-network test modules.
    compiled, inputs = BUILDERS[name](compile=True)
    eager, _ = BUILDERS[name](compile=False)
    eager.load_state_dict(compiled.state_dict())
    compiled.eval()
    eager.eval()

    out_v, out_s = compiled(*inputs)
    ref_v, ref_s = eager(*inputs)

    torch.testing.assert_close(out_v, ref_v, **STRICT_TOLERANCES)
    torch.testing.assert_close(out_s, ref_s, **STRICT_TOLERANCES)


def test_compile_kwargs_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    # compile_kwargs reaches torch.compile verbatim, and the resulting static-shape graph still
    # matches eager. Spying on torch.compile is what pins the forwarding down: output parity holds
    # whether or not the kwargs arrive, so on its own it would not test anything here.
    recorded = []
    real_compile = torch.compile

    def spy(fn, **kwargs):
        recorded.append(kwargs)
        return real_compile(fn, **kwargs)

    monkeypatch.setattr(torch, "compile", spy)
    compiled, inputs = BUILDERS["slim"](compile=True, compile_kwargs=dict(dynamic=False))
    assert recorded == [dict(dynamic=False)]

    eager, _ = BUILDERS["slim"](compile=False)
    eager.load_state_dict(compiled.state_dict())
    compiled.eval()
    eager.eval()

    out_v, out_s = compiled(*inputs)
    ref_v, ref_s = eager(*inputs)

    torch.testing.assert_close(out_v, ref_v, **STRICT_TOLERANCES)
    torch.testing.assert_close(out_s, ref_s, **STRICT_TOLERANCES)
