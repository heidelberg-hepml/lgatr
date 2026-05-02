"""L-GATr self-attention."""

import torch
from torch import nn

from ..dropout import GradeDropout
from ..linear import EquiLinear
from .attention import GeometricAttention
from .config import SelfAttentionConfig
from .qkv import MultiQueryQKVModule, QKVModule


class SelfAttention(nn.Module):
    """L-GATr self-attention.

    Constructs queries, keys, and values, computes geometric attention, and projects linearly to
    outputs.

    Parameters
    ----------
    config
        Attention configuration.
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()

        # Store settings
        self.config = config

        # QKV computation
        self.qkv_module = MultiQueryQKVModule(config) if config.multi_query else QKVModule(config)

        # Output projection
        self.out_linear = EquiLinear(
            in_mv_channels=config.hidden_mv_channels * config.num_heads,
            out_mv_channels=config.out_mv_channels,
            in_s_channels=config.hidden_s_channels * config.num_heads,
            out_s_channels=config.out_s_channels,
            initialization=config.output_init,
        )

        # Attention
        self.attention = GeometricAttention(config)

        # Dropout
        self.dropout: nn.Module | None
        if config.dropout_prob is not None:
            self.dropout = GradeDropout(config.dropout_prob)
        else:
            self.dropout = None

        # HeadScaleMHA
        self.use_head_scale = config.head_scale
        if self.use_head_scale:
            self.head_scale = nn.Parameter(torch.ones(config.num_heads))

    def forward(
        self,
        multivectors: torch.Tensor,
        additional_qk_features_mv: torch.Tensor | None = None,
        scalars: torch.Tensor | None = None,
        additional_qk_features_s: torch.Tensor | None = None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute self-attention.

        The result is the following:

        .. code-block::

            # For each head
            queries = linear_channels(inputs)
            keys = linear_channels(inputs)
            values = linear_channels(inputs)
            hidden = attention_items(queries, keys, values, biases=biases)
            head_output = linear_channels(hidden)

            # Combine results
            output = concatenate_heads head_output

        Parameters
        ----------
        multivectors
            Input multivectors of shape ``(..., items, mv_channels, 16)``.
        additional_qk_features_mv
            Additional multivector Q/K features of shape ``(..., items, add_qk_mv_channels, 16)``.
        scalars
            Optional input scalars of shape ``(..., items, s_channels)``. If None, the scalar
            stream is bypassed and ``outputs_s`` may be None (or a tensor lifted by ``out_linear``
            if ``out_s_channels`` is configured).
        additional_qk_features_s
            Additional scalar Q/K features of shape ``(..., items, add_qk_s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_mv
            Output multivectors of shape ``(..., items, mv_channels, 16)``.
        outputs_s
            Output scalars of shape ``(..., items, s_channels)``, or None.
        """
        # Compute Q, K, V
        q_mv, k_mv, v_mv, q_s, k_s, v_s = self.qkv_module(
            multivectors, scalars, additional_qk_features_mv, additional_qk_features_s
        )

        # Attention layer
        h_mv, h_s = self.attention(
            q_mv,
            k_mv,
            v_mv,
            q_s,
            k_s,
            v_s,
            **attn_kwargs,
        )
        if self.use_head_scale:
            h_mv = h_mv * self.head_scale.view(
                *[1] * len(h_mv.shape[:-5]), len(self.head_scale), 1, 1, 1
            )
            if h_s is not None:
                h_s = h_s * self.head_scale.view(
                    *[1] * len(h_s.shape[:-4]), len(self.head_scale), 1, 1
                )

        h_mv = h_mv.transpose(-4, -3).flatten(-3, -2)
        if h_s is not None:
            h_s = h_s.transpose(-3, -2).flatten(-2, -1)

        # Transform linearly one more time
        outputs_mv, outputs_s = self.out_linear(h_mv, scalars=h_s)

        # Dropout
        if self.dropout is not None:
            outputs_mv, outputs_s = self.dropout(outputs_mv, outputs_s)

        return outputs_mv, outputs_s
