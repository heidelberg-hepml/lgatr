From lgatr v1 to v2
===================

The ``lgatr`` v2 release renames parts of the interface, mostly to make names consistent
between the :class:`~lgatr.nets.lgatr.LGATr` and :class:`~lgatr.nets.slim.LGATrSlim`
architectures. The concrete changes are:

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
  ``compile_kwargs`` dict that is passed on to :func:`torch.compile`. Note that ``dynamic``
  is not enabled by default anymore, use ``compile_kwargs={"dynamic": True}`` to recover
  the v1 behavior.
