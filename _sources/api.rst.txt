API Reference
=============

L-GATr Networks
---------------

We provide two main L-GATr networks, :class:`~lgatr.nets.lgatr.LGATr` as a stack of transformer encoders,
and :class:`~lgatr.nets.conditional_lgatr.ConditionalLGATr` as a stack of transformer decoders.
For tasks where conditional inputs are required, you can process the condition with a :class:`~lgatr.nets.lgatr.LGATr`
and then include this processed condition using a :class:`~lgatr.nets.conditional_lgatr.ConditionalLGATr`.
In addition :class:`~lgatr.nets.lgatr_slim.LGATrSlim` and :class:`~lgatr.nets.conditional_lgatr_slim.ConditionalLGATrSlim`
provide more efficient versions of the respective networks using only scalar and vector representations.

.. autosummary::
   :toctree: generated/
   :recursive:

   lgatr.nets.lgatr.LGATr
   lgatr.nets.conditional_lgatr.ConditionalLGATr
   lgatr.nets.lgatr_slim.LGATrSlim
   lgatr.nets.conditional_lgatr_slim.ConditionalLGATrSlim

L-GATr Layers
-------------

The :class:`~lgatr.nets.lgatr.LGATr` and :class:`~lgatr.nets.conditional_lgatr.ConditionalLGATr` networks
have a structure similar to standard transformers. We construct them using variants of the standard
transformer layers adapted to the geometric algebra framework.

.. autosummary::
   :toctree: generated/
   :recursive:

   lgatr.layers.lgatr_block.LGATrBlock
   lgatr.layers.conditional_lgatr_block.ConditionalLGATrBlock
   lgatr.layers.linear.EquiLinear
   lgatr.layers.attention.self_attention.SelfAttention
   lgatr.layers.attention.cross_attention.CrossAttention
   lgatr.layers.mlp.mlp.GeoMLP
   lgatr.layers.mlp.geometric_bilinears.GeometricBilinear
   lgatr.layers.mlp.nonlinearities.ScalarGatedNonlinearity
   lgatr.layers.layer_norm.EquiLayerNorm
   lgatr.layers.dropout.GradeDropout

L-GATr Primitives
-----------------

The L-GATr primitives implement the core equivariant operations and are called by the L-GATr layers.

.. autosummary::
   :toctree: generated/
   :recursive:

   lgatr.primitives.attention
   lgatr.primitives.bilinear
   lgatr.primitives.dropout
   lgatr.primitives.invariants
   lgatr.primitives.linear
   lgatr.primitives.nonlinearities
   lgatr.primitives.normalization


L-GATr Configuration Classes
----------------------------

L-GATr uses ``dataclass`` objects to organize less relevant hyperparameters like number of heads or the MLP nonlinearity.
The :class:`~lgatr.layers.mlp.config.MLPConfig`, :class:`~lgatr.layers.attention.config.SelfAttentionConfig` and :class:`~lgatr.layers.attention.config.CrossAttentionConfig` are arguments for the :class:`~lgatr.nets.lgatr.LGATr`/:class:`~lgatr.nets.conditional_lgatr.ConditionalLGATr` modules,
whereas the :class:`~lgatr.primitives.config.LGATrConfig` is a global object that is accessed within the L-GATr primitives.

.. autosummary::
   :toctree: generated/
   :recursive:

   lgatr.primitives.config.LGATrConfig
   lgatr.layers.attention.config.SelfAttentionConfig
   lgatr.layers.attention.config.CrossAttentionConfig
   lgatr.layers.mlp.config.MLPConfig

Interface to the Geometric Algebra
----------------------------------

Before we feed data into L-GATr networks and after we extract results, we have to convert between common scalar/vector objects and multivectors.
This is very simple, we still introduce convenience methods for this step.
We also include functionality to construct `spurions`, or reference multivectors,
which can be added as extra items or channels to break equivariance at the input level.

.. autosummary::
   :toctree: generated/
   :recursive:

   lgatr.interface.scalar
   lgatr.interface.vector
   lgatr.interface.pseudoscalar
   lgatr.interface.axialvector
   lgatr.interface.spurions

L-GATr-slim Layers
------------------

In addition to the full L-GATr network, we provide a slimmed-down version that uses only scalar and vector representations instead of full multivectors.
This approach allows a more efficient implementation while achieving similar performance on all high-energy physics tasks we have tested so far.

.. autosummary::
   :toctree: generated/
   :recursive:

   lgatr.nets.lgatr_slim.LGATrSlimBlock
   lgatr.nets.conditional_lgatr_slim.ConditionalLGATrSlimBlock
   lgatr.nets.lgatr_slim.SelfAttention
   lgatr.nets.conditional_lgatr_slim.CrossAttention
   lgatr.nets.lgatr_slim.MLP
   lgatr.nets.lgatr_slim.GatedLinearUnit
   lgatr.nets.lgatr_slim.Linear
   lgatr.nets.lgatr_slim.RMSNorm
   lgatr.nets.lgatr_slim.Dropout
