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
(bilinear) operations as coefficient lists, then the list has over 90% zeros for
the linear layers, and over 99% for the bilinear layers. One can either implement them
as `dense` operations that use the efficient GEMM matrix multiplication kernels but with
many zero-coefficients, or as `sparse` operations that do do not spend compute on
zero-multiplications but use less efficient kernels. The linear and bilinear
operations in :class:`~lgatr.nets.lgatr.LGATr` support both options through
the ``sparse_linear`` and ``sparse_gp``` keys in
:class:`~lgatr.primitives.config.PrimitivesConfig`. The default is ``sparse_gp=True``,
``sparse_linear=False``, which runs fastest on GPU but has significnatly higher FLOPs
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

    net = LGATr(
        num_blocks=2,
        in_mv_channels=1,
        out_mv_channels=1,
        hidden_mv_channels=4,
        in_s_channels=5,
        out_s_channels=0,
        hidden_s_channels=32,
        compile=True,
        compile_kwargs={
            "dynamic": True,
            "fullgraph": True,
            "mode": "default",
        },
    )

We find that ``torch.compile`` significantly reduces time and memory consumption,
and recommend to always use turn it on (on GPU and CPU). For varying
shapes, we recommend setting ``compile_kwargs={"dynamic"}``. If used correctly, the
only cost to pay for ``compile=True`` is a ~1min compilation overhead on the first
network call.

Old torch versions limit what can be compiled. On ``torch<2.3`` the inductor backend
cannot generate code for the attention scale under dynamic shapes, so ``compile=True``
requires ``compile_kwargs={"dynamic": False}`` there. On ``torch<2.2`` compiling
additionally requires ``setuptools<82``, because torch imports ``pkg_resources``, which
setuptools removed in version 82.

Automic mixed precision
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
be applied on any function to upcasts to float32 precision locally.

However, **currently we do not recommend to use amp** with the
:class:`~lgatr.nets.slim.LGATrSlim` or :class:`~lgatr.nets.lgatr.LGATr`
architectures. For tests on jet tagging, we found that networks trained with
amp achieve significantly lower performance in some cases, to the point that
the speed and memory gains from amp do not justify the performance drop.
We are working on actively working on understanding this better.
