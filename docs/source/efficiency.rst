Efficient implementation
========================

This page gives advice on how to make :class:`~lgatr.nets.slim.LGATrSlim` and
:class:`~lgatr.nets.lgatr.LGATr` run faster and use less memory if required.
Both tricks are turned off by default, although we recommend to always set
``compile=True``.

torch.compile
--------------------------

Equivariant architectures introduce new operations that are typically not as optimized
as the standard neural network operations. Examples are the linear layers in
:class:`~lgatr.nets.slim.LGATrSlim` / :class:`~lgatr.nets.lgatr.LGATr` that use
joint weights for all vector/multivector components, and the geometric product in
:class:`~lgatr.nets.lgatr.LGATr`. These operations are then typically the reason why
equivariant networks are slower at equal parameter count.

The general problem is sparsity: For linear or bilinear operations with given
coefficients, i.e. a matrix for linear and a 3-tensor for bilinear, non-equivariant
networks allow arbitrary entries for the coefficients, whereas equivariant networks
constrain the list such that coefficients agree or certain coefficients are zero.
When expressing the :class:`~lgatr.nets.lgatr.LGATr` linear and tensor product
(bilinear) operations as coefficient lists, then the list has 87.5% zeros for
the linear layers, and 93.75% for the bilinear layers. One can either implement them
as `dense` operations that use the efficient GEMM matrix multiplication kernels but with
many zero-coefficients, or as `sparse` operations that do not spend compute on
zero-multiplications but use less efficient kernels. The linear and bilinear
operations in :class:`~lgatr.nets.lgatr.LGATr` support both options through
the ``sparse_linear`` and ``sparse_gp`` keys in
:class:`~lgatr.primitives.config.PrimitivesConfig`. The default is ``sparse_gp=True``,
``sparse_linear=False``, which runs fastest on GPU but has significantly higher FLOPs
usage compared to ``sparse_linear=True``. On CPU the fully sparse implementation is fastest.

The optimal solution for the problem of inefficient kernels is to write optimized
triton or CUDA kernels for these operations, or even to create efficient
implementations at the hardware level. Torchs native ``torch.compile`` tool serves
as a cheap variant that dynamically combines operations and selects optimized kernels.
Both :class:`~lgatr.nets.slim.LGATrSlim`-type / :class:`~lgatr.nets.lgatr.LGATr`-type
networks support a ``compile=True`` option which internally applies ``torch.compile``
on ``self.forward``, and also supports a dict of ``compile_kwargs`` that is passed
on without modifications. For instance,

.. code-block:: python

    from lgatr import LGATr

    net = LGATr(
        num_blocks=2,
        in_mv_channels=1,
        out_mv_channels=1,
        hidden_mv_channels=4,
        in_s_channels=5,
        out_s_channels=0,
        hidden_s_channels=32,
        attention=dict(num_heads=4),
        mlp=dict(),
        compile=True,
        compile_kwargs={
            "dynamic": True,
            "fullgraph": True,
            "mode": "default",
        },
    )

We find that ``torch.compile`` significantly reduces time and memory consumption,
and recommend to always turn it on (on GPU and CPU). For varying
shapes, we recommend setting ``compile_kwargs={"dynamic": True}``. If used correctly, the
only cost to pay for ``compile=True`` is a ~1min compilation overhead on the first
network call.

Old torch versions limit what can be compiled. On ``torch<2.3`` the inductor backend
cannot generate code for the attention scale under dynamic shapes, so ``compile=True``
requires ``compile_kwargs={"dynamic": False}`` there. On ``torch<2.2`` compiling
additionally requires ``setuptools<82``, because torch imports ``pkg_resources``, which
setuptools removed in version 82.

Automatic mixed precision
--------------------------------------------

Evaluating networks with float16 or bfloat16 precision can significantly decrease time
and memory usage, because kernels on reduced precision are faster, and weights as
well as activation take less space on disk. Automatic mixed precision (amp) allows to perform the forward pass at float16/bfloat16
precision, and does a more careful treatment of objects in the backward pass compared
to naive float16/bfloat16.

The :class:`~lgatr.nets.slim.LGATrSlim` / :class:`~lgatr.nets.lgatr.LGATr`
architectures both support automatic mixed precision. There are two modes:
``naive_amp=True`` directly applies amp without any modifications, whereas
``naive_amp=False`` performs only operations on scalars in float16/bfloat16,
and uses full float32 precision for operations on vectors. The
``naive_amp=False`` path uses a custom
:class:`~lgatr.utils.autocast.minimum_autocast_precision` decorator that can
be applied on any function to upcast to float32 precision locally.

However, **currently we do not recommend to use amp** with the
:class:`~lgatr.nets.slim.LGATrSlim` or :class:`~lgatr.nets.lgatr.LGATr`
architectures. For tests on jet tagging, we found that networks trained with
amp achieve significantly lower performance in some cases, to the point that
the speed and memory gains from amp do not justify the performance drop.
We are actively working on understanding this better.
For the slim networks, `Light-cone coordinates`_ below removes the numerical
cause behind much of this drop.

Light-cone coordinates
--------------------------------------------

Minkowski products of nearly collinear, nearly massless vectors, as in a jet, are
small differences of large numbers:

.. math::
    p_i \cdot p_j = E_i E_j (1 - \cos\theta_{ij}) \approx E_i E_j \theta_{ij}^2 / 2 .

Rounding the Cartesian components to float16 or bfloat16 therefore destroys the result.
This is why ``naive_amp=False`` keeps attention, the vector linear layers and the metric
contractions in float32, and those float32 islands are what eats most of the speedup that
amp should give.

:func:`~lgatr.interface.lightcone.get_lightcone_frame` removes the cancellation by a change
of basis. Around a reference direction ``n``, it constructs the orthogonal map ``T`` to
light-cone coordinates, in which a Lorentz vector ``v = (t, r)`` is stored as
``x+ = (t + r.n)/sqrt(2)``, ``x- = (t - r.n)/sqrt(2)``, and the two transverse components.
Since ``T eta T^T`` is again a metric of that form, a network fed ``T v`` for every vector
computes the same function as one fed ``v``. What changes is the conditioning: the small
combination ``t - r.n`` is formed once, in float64, and then stored as a number of its own.

The :class:`~lgatr.nets.slim.LGATrSlim` and
:class:`~lgatr.nets.conditional_slim.ConditionalLGATrSlim` networks accept such inputs with
``lightcone=True``. Attention and the vector GEMMs then follow the autocast dtype instead of
being pinned to float32.

.. code-block:: python

    import torch

    from lgatr import LGATrSlim, from_lightcone, get_lightcone_frame, to_lightcone

    # reference: the summed four-momentum of the items the network sees, e.g. the jet momentum
    frame = get_lightcone_frame(jet_momentum)[:, None, None]

    net = LGATrSlim(..., lightcone=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # every vector, including any reference vectors added as extra items, uses the same frame
        outputs_v, outputs_s = net(to_lightcone(vectors, frame), scalars)
    outputs_v = from_lightcone(outputs_v, frame)

The frame assumes that the reference has a non-vanishing spatial direction that is not exactly
along the z axis, which holds for a jet; no fallback is applied otherwise. In a jet-tagging
setup (8 blocks, batch size 2048, ``compile=True``, RTX 5090) this reduced the step time from
143 ms with float32-pinned attention to 61 ms, at the accuracy of float32 training.
