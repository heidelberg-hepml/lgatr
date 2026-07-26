"""xformers memory-efficient attention backend."""

import torch

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalCausalMask,
        BlockDiagonalMask,
    )
except ModuleNotFoundError as err:
    raise ImportError(
        "xformers is not installed. Run 'pip install lgatr[xformers-attention]'."
    ) from err

if not hasattr(torch.library, "custom_op"):  # torch.library.custom_op was added in torch 2.4
    raise ImportError("torch>=2.4 is not installed. Run 'pip install lgatr[xformers-attention]'.")


_CUSTOM_MASK_TYPE = {
    BlockDiagonalMask: 0,
    BlockDiagonalCausalMask: 1,
    BlockDiagonalCausalFromBottomRightMask: 2,
}


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype | None = None,
    attn_bias=None,
    **kwargs,
) -> torch.Tensor:
    """Forward to xformers' ``memory_efficient_attention``.

    Parameters
    ----------
    query
        Queries of shape ``(batch, head, items_out, channel)``.
    key
        Keys of shape ``(batch, head, items_in, channel)``.
    value
        Values of shape ``(batch, head, items_in, channel)``.
    dtype
        If specified, cast input tensors to this dtype before passing to attention. Useful to
        trigger flash-attention.
    attn_bias
        Optional attention bias, e.g. an ``xformers.ops.fmha.attn_bias.BlockDiagonalMask``.
    **kwargs
        Additional keyword arguments forwarded to ``memory_efficient_attention``.

    Returns
    -------
    out
        Result of shape ``(batch, head, items_out, channel)``.
    """
    assert query.ndim == 4, (
        "xformers constrains attention input shape to (batch, head, items, channel)."
    )
    # xformers and the attention kernels expect shape (batch, item, head, channel)
    query, key, value = (t.transpose(1, 2) for t in (query, key, value))
    if key.shape[2] != query.shape[2]:
        # broadcast key/value heads for multi-query / grouped-query attention
        key = key.expand(*key.shape[:2], query.shape[2], key.shape[3])
        value = value.expand(*value.shape[:2], query.shape[2], value.shape[3])

    # attention kernels require head_dim aligned to 128 bits (4 elements in fp32, 8 in
    # fp16/bf16); zero-pad to a multiple of 8 to cover every dtype and overwrite scale for correctness.
    head_dim = query.shape[-1]
    pad = -head_dim % 8
    if pad:
        query, key, value = (torch.nn.functional.pad(t, (0, pad)) for t in (query, key, value))

    if torch.compiler.is_compiling() and _fp32_custom_op_supported(query, dtype, attn_bias, kwargs):
        # fp32 uses xformers' cutlass kernel, which torch.compile cannot trace; route it
        # through the custom ops below instead.
        out = _attention_compiled(
            query, key, value, attn_bias, scale=head_dim**-0.5 if pad else None
        )
    else:
        if pad:
            kwargs.setdefault("scale", head_dim**-0.5)
        # fp16/bf16 kernels are torch.compile-traceable, so trace straight through
        # memory_efficient_attention; only the untraceable fp32 cutlass fallback is run under
        # torch.compiler.disable() (a clean graph break rather than a trace failure).
        compute_dtype = dtype if dtype is not None else query.dtype
        traceable = compute_dtype in (torch.float16, torch.bfloat16)
        forward = _attention_xformers if traceable else _attention_disabled
        out = forward(query, key, value, dtype=dtype, attn_bias=attn_bias, **kwargs)

    if pad:
        out = out[..., :head_dim]
    return out.transpose(1, 2).contiguous()


def _fp32_custom_op_supported(query, dtype, attn_bias, kwargs) -> bool:
    """Whether the fp32 custom-op path reproduces ``memory_efficient_attention`` exactly.

    Only fp32 needs it (fp16/bf16 kernels are torch.compile-traceable directly); also requires a
    basic ``attn_bias`` type and no extra kwargs.
    """
    return (
        dtype is None
        and query.dtype == torch.float32
        and type(attn_bias) in _CUSTOM_MASK_TYPE
        and not kwargs
    )


def _attention_compiled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: BlockDiagonalMask,
    scale: float | None = None,
) -> torch.Tensor:
    """Block-diagonal attention through the torch.compile-traceable custom ops."""
    q_seqinfo, k_seqinfo = attn_bias.q_seqinfo, attn_bias.k_seqinfo
    seqstart_q = q_seqinfo.seqstart.to(query.device)
    if k_seqinfo is q_seqinfo:
        # self-attention shares one seqstart tensor; reading it through both
        # attributes fails dynamo's duplicate-input guards
        seqstart_k, max_seqlen_k = seqstart_q, q_seqinfo.max_seqlen
    else:
        seqstart_k, max_seqlen_k = k_seqinfo.seqstart.to(query.device), k_seqinfo.max_seqlen
    compute_lse = torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    )
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    out, _ = _compiled_varlen_fwd(
        query,
        key,
        value,
        seqstart_q,
        seqstart_k,
        q_seqinfo.max_seqlen,
        max_seqlen_k,
        compute_lse,
        _CUSTOM_MASK_TYPE[type(attn_bias)],
        scale,
    )
    return out


def _attention_xformers(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype | None = None,
    attn_bias=None,
    **kwargs,
) -> torch.Tensor:
    """Forward to xformers' ``memory_efficient_attention`` (torch.compile-traceable for fp16/bf16)."""
    if dtype is not None:
        in_dtype = query.dtype
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)

    out = memory_efficient_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_bias=attn_bias,
        **kwargs,
    )

    return out.to(in_dtype) if dtype is not None else out


# fp32 cutlass attention is not torch.compile-traceable; this disabled variant turns it into a
# clean graph break instead of a dynamo trace failure.
_attention_disabled = torch.compiler.disable()(_attention_xformers)


@torch.library.custom_op("lgatr::compiled_varlen_fwd", mutates_args=())
def _compiled_varlen_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seqstart_q: torch.Tensor,
    seqstart_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    compute_lse: bool,
    custom_mask_type: int,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _, _, _ = torch.ops.aten._efficient_attention_forward(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        bias=None,
        cu_seqlens_q=seqstart_q,
        cu_seqlens_k=seqstart_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        custom_mask_type=custom_mask_type,
        compute_log_sumexp=compute_lse,
        scale=scale,
    )
    return out, lse


@_compiled_varlen_fwd.register_fake
def _compiled_varlen_fwd_fake(
    query,
    key,
    value,
    seqstart_q,
    seqstart_k,
    max_seqlen_q,
    max_seqlen_k,
    compute_lse,
    custom_mask_type,
    scale,
):
    num_seqs = seqstart_q.shape[0] - 1
    lse_dim = ((max_seqlen_q + 31) // 32) * 32 if compute_lse else 0
    out = query.new_empty(query.shape)
    lse = query.new_empty((num_seqs, query.shape[2], lse_dim), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("lgatr::compiled_varlen_bwd", mutates_args=())
def _compiled_varlen_bwd(
    grad: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    seqstart_q: torch.Tensor,
    seqstart_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    custom_mask_type: int,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # philox seed/offset are only read for dropout_p > 0
    rng_dummy = torch.zeros((), dtype=torch.int64)
    grad_q, grad_k, grad_v, _ = torch.ops.aten._efficient_attention_backward(
        grad.contiguous(),
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        bias=None,
        out=out.contiguous(),
        cu_seqlens_q=seqstart_q,
        cu_seqlens_k=seqstart_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        logsumexp=lse.contiguous(),
        dropout_p=0.0,
        philox_seed=rng_dummy,
        philox_offset=rng_dummy,
        custom_mask_type=custom_mask_type,
        bias_requires_grad=False,
        scale=scale,
    )
    return grad_q, grad_k, grad_v


@_compiled_varlen_bwd.register_fake
def _compiled_varlen_bwd_fake(
    grad,
    query,
    key,
    value,
    out,
    lse,
    seqstart_q,
    seqstart_k,
    max_seqlen_q,
    max_seqlen_k,
    custom_mask_type,
    scale,
):
    return (
        query.new_empty(query.shape),
        key.new_empty(key.shape),
        value.new_empty(value.shape),
    )


def _compiled_varlen_setup(ctx, inputs, output):
    (
        query,
        key,
        value,
        seqstart_q,
        seqstart_k,
        max_seqlen_q,
        max_seqlen_k,
        compute_lse,
        custom_mask_type,
        scale,
    ) = inputs
    out, lse = output
    ctx.save_for_backward(query, key, value, seqstart_q, seqstart_k, out, lse)
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.custom_mask_type = custom_mask_type
    ctx.scale = scale


def _compiled_varlen_backward(ctx, grad_out, grad_lse):
    query, key, value, seqstart_q, seqstart_k, out, lse = ctx.saved_tensors
    grad_q, grad_k, grad_v = _compiled_varlen_bwd(
        grad_out.contiguous(),
        query,
        key,
        value,
        out,
        lse,
        seqstart_q,
        seqstart_k,
        ctx.max_seqlen_q,
        ctx.max_seqlen_k,
        ctx.custom_mask_type,
        ctx.scale,
    )
    return grad_q, grad_k, grad_v, None, None, None, None, None, None, None


_compiled_varlen_fwd.register_autograd(
    _compiled_varlen_backward, setup_context=_compiled_varlen_setup
)
