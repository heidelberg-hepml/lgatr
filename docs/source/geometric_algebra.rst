Spacetime Geometric Algebra
===========================

Data representation
-------------------

Lorentz-equivariant architectures hard-code the transformation behavior
(or representations) of network inputs and outputs. Many features, such as
particle type information (PID), are inherently scalar, and processing them
with an equivariant architecture does not give any benefit compared to
a standard architecture like a MLP, graph network, or transformer.
The most common example of non-trivial representations in high-energy physics
are the four-momenta of particles, which are Lorentz vectors.
Before using Lorentz-equivariant architectures, it is essential to identify
the transformation behaviour (or representation) of the network inputs and
outputs under the Lorentz group.

Latent space representations
----------------------------

Besides the representation for input and output features, we also have
to pick a representation for the neural network latent space. Note that this choice
also depends on the approach used to build Lorentz-equivariant architectures, e.g.
PELICAN uses Lorentz-invariant latent features but then requires more complicated
operations to implement permutation-equivariance, and in LLoCa the latent space
representation determines how features are transformed between frames in tensorial
message-passing although all learnable operations use invariant features.
The :class:`~lgatr.nets.lgatr.LGATr` and :class:`~lgatr.nets.slim.LGATrSlim`
architectures use the arguably most straight-forward approach of maintaining
explicit representations throughout the architecture, and modify all standard
neural network operations such that they work on these representations.

For a given task, the latent space representation should be chosen such that it
matches the physics of the problem. For instance, if the network should learn
an amplitude that involves a Levi Civita tensor then it the latent representations
should allow this operation. As we discuss below, :class:`~lgatr.nets.lgatr.LGATr`
can express this operation, whereas :class:`~lgatr.nets.slim.LGATrSlim` cannot.
However, the more expressive operations in :class:`~lgatr.nets.lgatr.LGATr` are more
expensive to implement for a similar amount of learnable parameters, see :doc:`efficiency`.

Scalar and Vector representations
---------------------------------

:class:`~lgatr.nets.slim.LGATrSlim` implements a minimal approach using only scalar
and vector representations. The LorentzNet architecture implements the same idea for
graph networks. The allowed operations on vectors are the Minkowski inner product
and the scalar scalar product of a scalar and a vector.
:class:`~lgatr.nets.slim.LGATrSlim` implements these operations directly as part
of the :class:`~lgatr.nets.slim_layers.SlimGLU`, and also in the attention,
normalization, and nonlinearity operations. See https://arxiv.org/abs/2512.17011
for more information.

Spacetime Geometric Algebra representations
-------------------------------------------

:class:`~lgatr.nets.lgatr.LGATr` uses spacetime geometric algebra representations,
which naturally extends the minimal approach above that uses only scalar and vector
representations. Formally, the spacetime geometric algebra spans the space required
to express all results of the tensor product of two vector representations.
Using Minkowski index notation, a multivector :math:`x` of the spacetime
geometric algebra can be written as

.. math::
    x = x_0 + x_\mu^V \gamma^\mu + x_{\mu\nu}^B \sigma^{\mu\nu} + x_\mu^A \gamma^\mu\gamma^5 + x^P \gamma^5.

The 5 terms in this equation represent the 5 `grades` scalar, vector, bivector, axialvector,
and pseudoscalar, which form subrepresentations of the Lorentz group.
:class:`~lgatr.nets.lgatr.LGATr` internally performs operations on the 16-dimensional object
:math:`(x^S, x_\mu^V, x_{\mu\nu}^B, x_\mu^A, x^P)`, which fully characterizes a multivector.
Additionally, :class:`~lgatr.nets.lgatr.LGATr` relaxes this quite restrictive setup of working
solely with multivectors by allowing additional scalar representations, which mix with the scalar
part of the multivector representations.

:class:`~lgatr.nets.lgatr.LGATr` also covers parity-odd representations of the Lorentz group,
controlled by the flag ``subgroup`` in :class:`~lgatr.primitives.config.PrimitivesConfig`.
The default ``subgroup=True`` allows parity-even and parity-odd representations to mix,
effectively replacing parity-odd representations by parity-even representations.
Changing to ``subgroup=False`` recovers the parity-odd pseudoscalar and axialvector
representations as separate representations that mix non-trivially.
For more information, have a look at https://arxiv.org/abs/2411.00446 and https://arxiv.org/abs/2405.14806.

Embedding data in and extracting data from multivectors
-------------------------------------------------------

The ``lgatr`` package provides tools for embedding and extracting
:mod:`~lgatr.interface.scalar`, :mod:`~lgatr.interface.vector`, :mod:`~lgatr.interface.bivector`, :mod:`~lgatr.interface.axialvector` and
:mod:`~lgatr.interface.pseudoscalar` objects into spacetime geometric algebra representations.
For each representation, we have an embedding and an extraction function.
The embedding function takes the scalar/vector/bivector/axialvector/pseudoscalar
and embeds it into a multivector, while zero-padding the other multivector entries.
The extraction function returns the specified part of the multivector, ignoring
non-zero entries in all other representations. For example:

.. code-block:: python

    import torch
    from lgatr.interface import embed_vector, extract_vector, extract_scalar

    vector = torch.randn(1, 4)
    multivector = embed_vector(vector)
    print(multivector.shape)  # torch.Size([1, 16])

    vector_check = extract_vector(multivector) # same as vector
    scalar = extract_scalar(multivector) # torch.Size([1, 1])
    print(scalar) # tensor([[0.]])

We follow the multivector embedding convention of
`clifford package <https://clifford.readthedocs.io/en/latest/>`_.
Note that only :class:`~lgatr.nets.lgatr.LGATr`-type networks use multivector embeddings.
The :class:`~lgatr.nets.slim.LGATrSlim`-type networks directly take scalar and vector inputs.

Tensor conventions
------------------
.. _tensor-conventions:

Tensors are structured in a consistent way across the package:

- **Scalars** carry no trailing 16. Typical layouts:

  - ``(..., 1)`` — a single scalar embedded for use as a multivector argument.
  - ``(..., channels)`` — a stack of scalar channels.
  - ``(..., items, channels)`` — a sequence of scalar channels (transformer input).

- **Multivectors** (in :class:`~lgatr.nets.lgatr.LGATr`-type networks) carry a fixed last dimension of 16. Typical layouts:

  - ``(..., 16)`` — a single multivector (no channel dimension).
  - ``(..., channels, 16)`` — a stack of multivector channels.
  - ``(..., items, channels, 16)`` — a sequence of multivector channels (transformer input).

- **Lorentz vectors** (in :class:`~lgatr.nets.slim.LGATrSlim`-type networks) carry a fixed last dimension
  of 4: ``(..., channels, 4)`` or ``(..., items, channels, 4)``.

- Leading batch dimensions ``...`` are arbitrary and broadcast as in standard PyTorch ops.
