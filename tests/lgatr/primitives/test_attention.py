import pytest
import torch

from lgatr.primitives import sdp_attention
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize(
    "tokens_in,tokens_out,mv_in,mv_out,s_in,s_out",
    [
        (7, 5, 2, 3, 17, 11),  # all dimensions distinct
        (1, 1, 1, 1, 1, 1),  # everything singleton
        (7, 1, 2, 3, 17, 11),  # single output token
        (7, 5, 1, 3, 1, 11),  # single input channel per stream
    ],
)
def test_scalar_attention_shape(
    tokens_in: int,
    tokens_out: int,
    mv_in: int,
    mv_out: int,
    s_in: int,
    s_out: int,
) -> None:
    # sdp_attention outputs have the expected (..., items_out, channels) shapes.
    q_mv = torch.randn(*BATCH_DIMS, tokens_out, mv_in, 16)
    k_mv = torch.randn(*BATCH_DIMS, tokens_in, mv_in, 16)
    v_mv = torch.randn(*BATCH_DIMS, tokens_in, mv_out, 16)
    q_s = torch.randn(*BATCH_DIMS, tokens_out, s_in)
    k_s = torch.randn(*BATCH_DIMS, tokens_in, s_in)
    v_s = torch.randn(*BATCH_DIMS, tokens_in, s_out)

    outputs, outputs_scalar = sdp_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s)

    assert outputs.shape == (*BATCH_DIMS, tokens_out, mv_out, 16)
    assert outputs_scalar.shape == (*BATCH_DIMS, tokens_out, s_out)


def test_sdp_attention_none_scalars() -> None:
    # sdp_attention propagates None when q_s/k_s/v_s are all None.
    q = k = v = torch.randn(2, 3, 4, 16)
    _, outputs_s = sdp_attention(q, k, v, q_s=None, k_s=None, v_s=None)
    assert outputs_s is None


def test_scalar_attention_equivariance() -> None:
    # sdp_attention is Pin-equivariant when scalar Q/K/V are passed.
    item_dim, key_dim, num_scalars = 3, 2, 5
    data_dims = (*BATCH_DIMS, item_dim, key_dim)
    kwargs = dict(
        q_s=torch.randn(*BATCH_DIMS, item_dim, num_scalars),
        k_s=torch.randn(*BATCH_DIMS, item_dim, num_scalars),
        v_s=torch.randn(*BATCH_DIMS, item_dim, num_scalars),
    )
    check_pin_equivariance(
        sdp_attention, 3, batch_dims=[data_dims] * 3, fn_kwargs=kwargs, spin=False, **TOLERANCES
    )
