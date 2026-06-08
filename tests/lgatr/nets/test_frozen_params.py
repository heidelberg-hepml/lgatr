import torch

from lgatr.layers.attention.config import SelfAttentionConfig
from lgatr.layers.mlp.config import MLPConfig
from lgatr.nets.conditional_lgatr_slim import ConditionalLGATrSlim
from lgatr.nets.lgatr import LGATr
from lgatr.nets.lgatr_slim import LGATrSlim


def assert_all_trainable_params_get_grads(net, *inputs):
    outputs = net(*inputs)
    loss = sum(out.float().pow(2).sum() for out in outputs if out is not None and out.numel() > 0)
    loss.backward()
    missing = [name for name, p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing


def test_LGATrSlim_frozen_params() -> None:
    # With out_v_channels=0, exactly the zero-size linear_out.weight_v and the dead vector
    # params of the LAST block (norm2 affine + MLP) are frozen; earlier blocks, the last
    # block's attention, and all scalar params stay trainable and receive grads.
    net = LGATrSlim(
        in_v_channels=1,
        in_s_channels=5,
        out_v_channels=0,
        out_s_channels=3,
        hidden_v_channels=4,
        hidden_s_channels=8,
        num_blocks=3,
        num_heads=2,
    )
    frozen = {name for name, p in net.named_parameters() if not p.requires_grad}
    assert frozen == {
        "linear_out.weight_v",
        "blocks.2.norm2.weight_v",
        "blocks.2.mlp.layers.0.linear.weight_v",
        "blocks.2.mlp.layers.1.weight_v",
    }
    assert_all_trainable_params_get_grads(net, torch.randn(2, 6, 1, 4), torch.randn(2, 6, 5))


def test_ConditionalLGATrSlim_frozen_params() -> None:
    # Same as above, but the pre-MLP norm of the conditional block is norm3; norm2 feeds
    # cross-attention, which stays alive through norm3's joint normalization.
    net = ConditionalLGATrSlim(
        in_v_channels=1,
        v_channels_cond=1,
        out_v_channels=0,
        hidden_v_channels=4,
        in_s_channels=5,
        s_channels_cond=4,
        out_s_channels=3,
        hidden_s_channels=8,
        num_blocks=2,
        num_heads=2,
    )
    frozen = {name for name, p in net.named_parameters() if not p.requires_grad}
    assert frozen == {
        "linear_out.weight_v",
        "blocks.1.norm3.weight_v",
        "blocks.1.mlp.layers.0.linear.weight_v",
        "blocks.1.mlp.layers.1.weight_v",
    }
    assert_all_trainable_params_get_grads(
        net,
        torch.randn(2, 6, 1, 4),
        torch.randn(2, 5, 1, 4),
        torch.randn(2, 6, 5),
        torch.randn(2, 5, 4),
    )


def test_LGATr_frozen_params() -> None:
    # Full LGATr has no dead tail (every EquiLinear mixes mv and s);
    # only the zero-size params of linear_out are frozen.
    net = LGATr(
        num_blocks=2,
        in_mv_channels=1,
        out_mv_channels=0,
        hidden_mv_channels=4,
        in_s_channels=5,
        out_s_channels=3,
        hidden_s_channels=8,
        attention=SelfAttentionConfig(num_heads=2),
        mlp=MLPConfig(),
    )
    frozen = {name for name, p in net.named_parameters() if not p.requires_grad}
    assert frozen == {
        "linear_out.weight",
        "linear_out.s2mvs.weight",
        "linear_out.s2mvs.bias",
    }
    assert_all_trainable_params_get_grads(net, torch.randn(2, 6, 1, 16), torch.randn(2, 6, 5))
