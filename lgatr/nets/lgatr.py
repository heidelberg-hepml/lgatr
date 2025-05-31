"""Equivariant transformer for multivector data."""

from dataclasses import replace
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers.attention.config import SelfAttentionConfig
from ..layers.lgatr_block import LGATrBlock
from ..layers.linear import EquiLinear
from ..layers.mlp.config import MLPConfig


class LGATr(nn.Module):
    """L-GATr network.

    It combines num_blocks L-GATr transformer blocks, each consisting of geometric self-attention
    layers, a geometric MLP, residual connections, and normalization layers. In addition, there
    are initial and final equivariant linear layers.

    Assumes input has shape (..., items, in_mv_channels, 16), output has shape
    (..., items, out_mv_channels, 16), will create hidden representations with shape
    (..., items, hidden_mv_channels, 16). Similar for extra scalar channels.

    Parameters
    ----------
    num_blocks : int
        Number of transformer blocks.
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    attention: Dict
        Data for SelfAttentionConfig
    mlp: Dict
        Data for MLPConfig
    reinsert_mv_channels : None or Tuple[int]
        If not None, specifies multivector channels that will be reinserted in every attention layer.
    reinsert_s_channels : None or Tuple[int]
        If not None, specifies scalar channels that will be reinserted in every attention layer.
    dropout_prob : float or None
        Dropout probability
    checkpoint_blocks : bool
        Whether to use checkpointing for the blocks. If True, will save memory at the cost of speed.
    """

    def __init__(
        self,
        num_blocks: int,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        dropout_prob: Optional[float] = None,
        checkpoint_blocks: bool = False,
    ) -> None:
        super().__init__()
        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = replace(
            SelfAttentionConfig.cast(attention),
            additional_qk_mv_channels=0
            if reinsert_mv_channels is None
            else len(reinsert_mv_channels),
            additional_qk_s_channels=0
            if reinsert_s_channels is None
            else len(reinsert_s_channels),
        )
        mlp = MLPConfig.cast(mlp)
        self.blocks = nn.ModuleList(
            [
                LGATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    attention=attention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels
        self._checkpoint_blocks = checkpoint_blocks

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        **attn_kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor
            Input multivectors with shape (..., items, in_mv_channels, 16).
        scalars : None or torch.Tensor
            Optional input scalars with shape (..., items, in_s_channels).
        **attn_kwargs
            Optional keyword arguments passed to attention.

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors with shape (..., items, out_mv_channels, 16).
        outputs_s : None or torch.Tensor
            Output scalars with shape (..., items, out_s_channels). None if out_s_channels=None.
        """

        # Channels that will be re-inserted in any query / key computation
        (
            additional_qk_features_mv,
            additional_qk_features_s,
        ) = self._construct_reinserted_channels(multivectors, scalars)

        # Pass through the blocks
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    **attn_kwargs,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    **attn_kwargs,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s

    def _construct_reinserted_channels(self, multivectors, scalars):
        """Constructs input features that will be reinserted in every attention layer."""

        if self._reinsert_mv_channels is None:
            additional_qk_features_mv = None
        else:
            additional_qk_features_mv = multivectors[..., self._reinsert_mv_channels, :]

        if self._reinsert_s_channels is None:
            additional_qk_features_s = None
        else:
            assert scalars is not None
            additional_qk_features_s = scalars[..., self._reinsert_s_channels]

        return additional_qk_features_mv, additional_qk_features_s
