from lgatr.primitives.attention_backends import get_attention_backend


def test_default_backend_selection():
    backend_fn = get_attention_backend()
    from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

    assert backend_fn is torch_sdpa
