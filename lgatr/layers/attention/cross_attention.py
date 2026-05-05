"""L-GATr cross-attention."""

import torch
from torch import nn

from ..dropout import GradeDropout
from ..linear import EquiLinear
from .attention import GeometricAttention
from .config import CrossAttentionConfig


class CrossAttention(nn.Module):
    """L-GATr cross-attention.

    Constructs queries, keys, and values, computes geometric attention, and projects linearly to
    outputs.

    Parameters
    ----------
    config
        Attention configuration.
    """

    def __init__(
        self,
        config: CrossAttentionConfig,
    ) -> None:
        super().__init__()

        if (
            config.additional_q_mv_channels > 0
            or config.additional_q_s_channels > 0
            or config.additional_k_mv_channels > 0
            or config.additional_k_s_channels > 0
        ):
            raise NotImplementedError("Cross attention is not implemented with additional channels")

        # Store settings
        self.config = config

        self.q_linear = EquiLinear(
            in_mv_channels=config.q_mv_channels,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.q_s_channels,
            out_s_channels=config.hidden_s_channels * config.num_heads,
        )
        self.kv_linear = EquiLinear(
            in_mv_channels=config.kv_mv_channels,
            out_mv_channels=2
            * config.hidden_mv_channels
            * (1 if config.multi_query else config.num_heads),
            in_s_channels=config.kv_s_channels,
            out_s_channels=2
            * config.hidden_s_channels
            * (1 if config.multi_query else config.num_heads),
        )

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
        multivectors_q: torch.Tensor,
        multivectors_kv: torch.Tensor,
        scalars_q: torch.Tensor | None = None,
        scalars_kv: torch.Tensor | None = None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute cross-attention.

        The result is the following:

        .. code-block::

            # For each head
            queries = linear_channels(multivectors_q)
            keys = linear_channels(multivectors_kv)
            values = linear_channels(multivectors_kv)
            hidden = attention_items(queries, keys, values, biases=biases)
            head_output = linear_channels(hidden)

            # Combine results
            output = concatenate_heads head_output

        Parameters
        ----------
        multivectors_q
            Input multivectors for queries, shape ``(..., items_q, q_mv_channels, 16)``.
        multivectors_kv
            Input multivectors for keys and values, shape ``(..., items_kv, kv_mv_channels, 16)``.
        scalars_q
            Optional input scalars for queries, shape ``(..., items_q, q_s_channels)``.
        scalars_kv
            Optional input scalars for keys and values, shape ``(..., items_kv, kv_s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention.

        Returns
        -------
        outputs_mv
            Output multivectors of shape ``(..., items_q, out_mv_channels, 16)``.
        outputs_s
            Output scalars of shape ``(..., items_q, out_s_channels)``, or None.
        """
        q_mv, q_s = self.q_linear(
            multivectors_q, scalars_q
        )  # (..., num_items, hidden_channels, 16)
        kv_mv, kv_s = self.kv_linear(
            multivectors_kv, scalars_kv
        )  # (..., num_items, 2*hidden_channels, 16)
        k_mv, v_mv = torch.tensor_split(kv_mv, 2, dim=-2)
        if kv_s is None:
            k_s, v_s = None, None
        else:
            k_s, v_s = torch.tensor_split(kv_s, 2, dim=-1)

        # Rearrange to (..., heads, items, channels, 16) shape
        q_mv = q_mv.unflatten(-2, (self.config.hidden_mv_channels, self.config.num_heads)).movedim(
            -2, -4
        )
        if self.config.multi_query:
            k_mv = k_mv.unsqueeze(-4)
            v_mv = v_mv.unsqueeze(-4)
        else:
            k_mv = k_mv.unflatten(
                -2, (self.config.hidden_mv_channels, self.config.num_heads)
            ).movedim(-2, -4)
            v_mv = v_mv.unflatten(
                -2, (self.config.hidden_mv_channels, self.config.num_heads)
            ).movedim(-2, -4)

        # Same for scalars
        if q_s is not None:
            q_s = q_s.unflatten(-1, (self.config.hidden_s_channels, self.config.num_heads)).movedim(
                -1, -3
            )
            if self.config.multi_query:
                k_s = k_s.unsqueeze(-3)
                v_s = v_s.unsqueeze(-3)
            else:
                k_s = k_s.unflatten(
                    -1, (self.config.hidden_s_channels, self.config.num_heads)
                ).movedim(-1, -3)
                v_s = v_s.unflatten(
                    -1, (self.config.hidden_s_channels, self.config.num_heads)
                ).movedim(-1, -3)
        else:
            q_s, k_s, v_s = None, None, None

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
            # The (num_heads, 1, 1, 1) factor right-aligns with the last four dims of h_mv,
            # broadcasting across the leading batch dims (analogous for h_s and three dims).
            h_mv = h_mv * self.head_scale[:, None, None, None]
            if h_s is not None:
                h_s = h_s * self.head_scale[:, None, None]

        h_mv = h_mv.transpose(-4, -3).flatten(-3, -2)
        if h_s is not None:
            h_s = h_s.transpose(-3, -2).flatten(-2, -1)

        # Transform linearly one more time
        outputs_mv, outputs_s = self.out_linear(h_mv, scalars=h_s)

        # Dropout
        if self.dropout is not None:
            outputs_mv, outputs_s = self.dropout(outputs_mv, outputs_s)

        return outputs_mv, outputs_s
