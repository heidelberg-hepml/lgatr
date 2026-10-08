From lgatr v1 to v2
===================

The ``lgatr`` v2 release renames parts of the interface, mostly to make names consistent
between the :class:`~lgatr.nets.lgatr.LGATr` and :class:`~lgatr.nets.slim.LGATrSlim`
architectures, and changes a handful of defaults. A v2 network constructed with the same
arguments as its v1 counterpart is therefore not numerically identical.

API changes
-----------

- :class:`~lgatr.nets.conditional_lgatr.ConditionalLGATr` and :class:`~lgatr.nets.conditional_slim.ConditionalLGATrSlim` ``__init__``  arguments:

  - ``condition_mv_channels`` → ``mv_channels_cond``
  - ``condition_v_channels``  → ``v_channels_cond``
  - ``condition_s_channels``  → ``s_channels_cond``

  and ``forward`` arguments:

  - ``multivectors_condition`` → ``multivectors_cond``
  - ``vectors_condition``      → ``vectors_cond``
  - ``scalars_condition``      → ``scalars_cond``

- :class:`~lgatr.layers.mlp.config.MLPConfig`:

  - ``activation`` → ``nonlinearity``
  - ``increase_hidden_channels`` → ``mlp_ratio``
  - ``num_hidden_layers`` → ``num_layers_mlp`` (``num_layers_mlp`` counts all layers instead of only the hidden ones, so explicit values have to be increased by one if the parameter is specified)

- :class:`~lgatr.layers.attention.config.SelfAttentionConfig` and
  :class:`~lgatr.layers.attention.config.CrossAttentionConfig`:

  - ``increase_hidden_channels`` → ``attn_ratio``.

- The global ``gatr_config`` object is replaced by
  :class:`~lgatr.primitives.config.PrimitivesConfig`, which is passed to each network as
  ``primitives``. This way several networks with different settings can coexist. The flags are renamed:

  - ``use_fully_connected_subgroup`` → ``subgroup``
  - ``use_bivector`` → ``bivector``
  - ``use_geometric_product`` → ``geometric_product``

- The ``compile_mode`` and ``compile_dynamic`` arguments are replaced by a single
  ``compile_kwargs`` dict that is passed on to :func:`torch.compile`.

Default changes
---------------

- ``norm_elementwise_affine=True`` in all normalization operations (:class:`~lgatr.layers.layer_norm.EquiLayerNorm`,
  :class:`~lgatr.layers.slim_layers.SlimRMSNorm`), allowing for an extra
  learned scalar multiplier. We find that this uniformly improves performance, and it is standard
  in LayerNorm and RMSNorm. Set ``norm_elementwise_affine=False`` to recover v1 behavior.

- ``sparse_gp=True`` in :class:`~lgatr.primitives.config.PrimitivesConfig`. With the new implementation
  this is faster on GPU and CPU. We keep ``sparse_linear=False`` as the default which is faster on GPU,
  but slower on CPU and costs significantly more FLOPs, see :doc:`efficiency`. This only
  affects efficiency, not the network output.

- ``nonlinearity_v="sigmoid"`` in :class:`~lgatr.layers.slim_layers.SlimGLU`. We find that this
  improves stability in extreme cases, because the vector branch can not blow up. Set
  ``nonlinearity_v=None`` to fall back to ``nonlinearity`` and recover the v1 behavior.

- ``compile=True`` no longer implies ``dynamic=True``. Set ``compile_kwargs={"dynamic": True}``
  to recover the old behavior.

- The vector gate in :class:`~lgatr.layers.slim_layers.SlimGLU` is scaled by ``1/sqrt(4)`` to
  correct for variance increase from the Minkowski product, similar to the ``1/sqrt(d_k)``
  factor in attention.

- The scalar bias in the qkv linear layers is turned off in all networks, because they cost
  parameters and extra kernel launches while having little effect.

- The bias of the scalar linear layer in :class:`~lgatr.layers.slim_layers.SlimLinear` is
  initialized to zero to improve numerical stability in extreme cases.
