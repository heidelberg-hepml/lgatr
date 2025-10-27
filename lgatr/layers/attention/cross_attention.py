"""L-GATr cross-attention."""

import torch
from einops import rearrange
from torch import nn

from ..dropout import GradeDropout
from ..linear import EquiLinear
from .attention import GeometricAttention
from .config import SelfAttentionConfig


class CrossAttention(nn.Module):
    """L-GATr cross-attention.

    Constructs queries, keys, and values, computes attention, and projects linearly to outputs.

    Parameters
    ----------
    config : SelfAttentionConfig
        Attention configuration.
    """

    def __init__(
        self,
        config: SelfAttentionConfig,
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
            in_mv_channels=config.in_q_mv_channels,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_q_s_channels,
            out_s_channels=config.hidden_s_channels * config.num_heads,
        )
        self.kv_linear = EquiLinear(
            in_mv_channels=config.in_kv_mv_channels,
            out_mv_channels=2
            * config.hidden_mv_channels
            * (1 if config.multi_query else config.num_heads),
            in_s_channels=config.in_kv_s_channels,
            out_s_channels=2
            * config.hidden_s_channels
            * (1 if config.multi_query else config.num_heads),
        )

        # Output projection
        self.out_linear = EquiLinear(
            in_mv_channels=config.hidden_mv_channels * config.num_heads,
            out_mv_channels=config.out_mv_channels,
            in_s_channels=(
                None
                if config.in_kv_s_channels is None
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
        multivectors_kv: torch.Tensor,
        multivectors_q: torch.Tensor,
        scalars_kv: torch.Tensor | None = None,
        scalars_q: torch.Tensor | None = None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cross attention.

        The result is the following:

        .. code-block::

            # For each head
            queries = linear_channels(inputs_q)
            keys = linear_channels(inputs_kv)
            values = linear_channels(inputs_kv)
            hidden = attention_items(queries, keys, values, biases=biases)
            head_output = linear_channels(hidden)

            # Combine results
            output = concatenate_heads head_output

        Parameters
        ----------
        multivectors_kv : torch.Tensor
            Input multivectors for key and value with shape (..., items_kv, mv_channels, 16).
        multivectors_q : torch.Tensor
            Input multivectors for query with shape (..., items_q, mv_channels, 16).
        scalars_kv : None or torch.Tensor
            Optional input scalars for key and value with shape (..., items_kv, s_channels)
        scalars_q : None or torch.Tensor
            Optional input scalars for query with shape (..., items_q, s_channels)
        **attn_kwargs
            Optional keyword arguments passed to attention.

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors with shape (..., items_q, mv_channels, 16).
        output_scalars : torch.Tensor
            Output scalars with shape (..., items_q, s_channels).
        """
        q_mv, q_s = self.q_linear(
            multivectors_q, scalars_q
        )  # (..., num_items, hidden_channels, 16)
        kv_mv, kv_s = self.kv_linear(
            multivectors_kv, scalars_kv
        )  # (..., num_items, 2*hidden_channels, 16)
        k_mv, v_mv = torch.tensor_split(kv_mv, 2, dim=-2)
        k_s, v_s = torch.tensor_split(kv_s, 2, dim=-1)

        # Rearrange to (..., heads, items, channels, 16) shape
        q_mv = rearrange(
            q_mv,
            "... items (hidden_channels num_heads) x -> ... num_heads items hidden_channels x",
            num_heads=self.config.num_heads,
            hidden_channels=self.config.hidden_mv_channels,
        )
        if self.config.multi_query:
            k_mv = rearrange(k_mv, "... items hidden_channels x -> ... 1 items hidden_channels x")
            v_mv = rearrange(v_mv, "... items hidden_channels x -> ... 1 items hidden_channels x")
        else:
            k_mv = rearrange(
                k_mv,
                "... items (hidden_channels num_heads) x -> ... num_heads items hidden_channels x",
                num_heads=self.config.num_heads,
                hidden_channels=self.config.hidden_mv_channels,
            )
            v_mv = rearrange(
                v_mv,
                "... items (hidden_channels num_heads) x -> ... num_heads items hidden_channels x",
                num_heads=self.config.num_heads,
                hidden_channels=self.config.hidden_mv_channels,
            )

        # Same for scalars
        if q_s is not None:
            q_s = rearrange(
                q_s,
                "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
                num_heads=self.config.num_heads,
                hidden_channels=self.config.hidden_s_channels,
            )
            if self.config.multi_query:
                k_s = rearrange(k_s, "... items hidden_channels -> ... 1 items hidden_channels")
                v_s = rearrange(v_s, "... items hidden_channels -> ... 1 items hidden_channels")
            else:
                k_s = rearrange(
                    k_s,
                    "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
                    num_heads=self.config.num_heads,
                    hidden_channels=self.config.hidden_s_channels,
                )
                v_s = rearrange(
                    v_s,
                    "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
                    num_heads=self.config.num_heads,
                    hidden_channels=self.config.hidden_s_channels,
                )
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
