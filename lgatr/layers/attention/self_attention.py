"""L-GATr self-attention."""

import torch
from einops import rearrange
from torch import nn

from ..dropout import GradeDropout
from ..linear import EquiLinear
from .attention import GeometricAttention
from .config import SelfAttentionConfig
from .qkv import MultiQueryQKVModule, QKVModule


class SelfAttention(nn.Module):
    """L-GATr self-attention.

    Constructs queries, keys, and values, computes attention, and projects linearly to outputs.

    Parameters
    ----------
    config : SelfAttentionConfig
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
            in_s_channels=(
                None
                if config.in_s_channels is None
                else config.hidden_s_channels * config.num_heads
            ),
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes self-attention.

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
        multivectors : torch.Tensor
            Input multivectors with shape (..., items, mv_channels, 16).
        additional_qk_features_mv : None or torch.Tensor
            Additional multivector Q/K features with shape (..., items, add_qk_mv_channels, 16)
        scalars : None or torch.Tensor
            Optional input scalars with shape (..., items, num_items, s_channels)
        additional_qk_features_s : None or torch.Tensor
            Additional scalar Q/K features with shape (..., items, add_qk_mv_channels, 16)
        scalars : None or torch.Tensor
            Optional input scalars with shape (..., items, s_channels).
        **attn_kwargs
            Optional keyword arguments passed to attention.

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors with shape (..., items, mv_channels, 16).
        output_scalars : torch.Tensor
            Output scalars with shape (..., items, s_channels).
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
            h_s = h_s * self.head_scale.view(*[1] * len(h_s.shape[:-4]), len(self.head_scale), 1, 1)

        h_mv = rearrange(
            h_mv,
            "... n_heads n_items hidden_channels x -> ... n_items (n_heads hidden_channels) x",
        )
        h_s = rearrange(
            h_s,
            "... n_heads n_items hidden_channels -> ... n_items (n_heads hidden_channels)",
        )

        # Transform linearly one more time
        outputs_mv, outputs_s = self.out_linear(h_mv, scalars=h_s)

        # Dropout
        if self.dropout is not None:
            outputs_mv, outputs_s = self.dropout(outputs_mv, outputs_s)

        return outputs_mv, outputs_s
