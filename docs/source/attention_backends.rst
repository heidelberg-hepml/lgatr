Attention Backends
==================

L-GATr prepares queries, keys and values such that they can be processed
with any attention backend. We implement several attention backends or kernels
as described below, and including more backends is straight-forward.

.. code-block:: python

    pip install lgatr  # only default attention
    pip install lgatr[varlen-attention]  # add varlen attention
    pip install lgatr[xformers-attention]  # add xformers attention
    pip install lgatr[flex-attention]  # add flex_attention
    pip install lgatr[flash-attention]  # add flash attention
    pip install lgatr[varlen-attention, xformers-attention,flex-attention,flash-attention]  # add all

You might have to run ``python -m pip install --upgrade pip setuptools wheel``
to update your build environment, extra imports require the most recent versions.

Why care about Attention Kernels?
---------------------------------

As sequence length grows, attention becomes the bottleneck in transformers—both in
memory consumption and computation time. This is because attention is the only
transformer operation whose cost grows quadratically with the sequence length.
To address this, researchers have devoted significant effort to designing more
efficient attention backends that reduce this quadratic blowup. The best-known
example is FlashAttention [1]_, which never writes out the full attention matrix
to memory but instead computes attention in smaller chunks. Today, FlashAttention
(via implementations like PyTorch’s built-in ``torch.nn.functional.scaled_dot_product_attention``,
xFormers, and others) is the standard in most transformer libraries.

However, these highly optimized CUDA kernels impose constraints on the attention
inputs and structure of the attention mask.

- PyTorch’s default attention path assumes dense tensors for
queries, keys, values, and the attention mask. In particle physics applications, each
event often contains a different number of particles—so using dense tensors would require
**padding all events to the maximum length**. An alternative is to work with “sparse” representations
(e.g., concatenating all particles in a long list and tracking event boundaries), then apply
a block-diagonal mask so that only particles within the same event attend to one another.
By avoiding operations on padded particles, this approach can dramatically reduce both memory
usage and computation time.
- Beyond sparse masks, many **advanced positional-encoding schemes**
(such as relative positional embeddings, ALiBi, sliding-window attention, PrefixLM,
tanh-soft-capping, and so on) also require custom attention kernels that deviate from
the dense, full-matrix assumption.
- Optimized attention backends are typically optimized for the **most recent NVIDIA GPUs**, and do not or only partially support older hardware.
- Optimized operations typically only support certain data types. ``float16`` and ``bfloat16`` are most widely supported, and ``float32`` and ``float8`` upcoming.
- Most backends currently only support certain head dimensions, e.g. only powers of 2.

PyTorch's native Attention
--------------------------

PyTorch's native
`scaled_dot_product_attention <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>`_
is easy to use, but it requires dense tensors for queries, keys and values.
It is the only backend that you get if you install via

.. code-block:: python

    pip install lgatr

xformers Attention
------------------

Xformers is a library for `efficient attention implementations <https://facebookresearch.github.io/xformers/components/ops.html#module-xformers.ops>`_ maintained
by facebook, including `support for block-diagonal attention masks <https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.fmha.attn_bias.BlockDiagonalMask>`_.
Unfortunately, xformers does not support MacOS anymore [2]_.
Under the hood, xformers supports `multiple attention backends <https://github.com/facebookresearch/xformers/tree/main/xformers/ops/fmha>`_,
including the varlen FlashAttention mentioned below, and selects the best one for your hardware automatically.
For this reason, the xformers attention backend is not included in the default installation of ``lgatr``.
To use it, you need to install ``lgatr`` with the ``xformers-attention`` extra:

.. code-block:: python

    pip install lgatr[xformers-attention]

PyTorch's flex_attention
------------------------

To mitigate the increasing need for custom attention kernels, PyTorch developers have
designed `flex_attention <https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html>`_,
a tool that aims to generalize all variations of attention kernels while remaining efficient.
The idea is to have two functions ``score_mod`` and ``block_mask`` as arguments
that allow the user to create most attention variants, see `this blog <https://pytorch.org/blog/flexattention/>`_.
flex_attention is considered stable for ``torch>=2.7``, we therefore do not include
it in the default installation of ``lgatr`` yet. You install ``lgatr`` with ``flex-attention`` as follows:

.. code-block:: python

    pip install lgatr[flex-attention]

Original FlashAttention
-----------------------

The `original FlashAttention implementation <https://github.com/Dao-AILab/flash-attention>`_ by Tri Dao is
still actively maintained and widely used within the community. Its support for variable-length sequences
is not yet part of native PyTorch, so we provide an extra for it in ``lgatr``.
Note that xformers might default to using FlashAttention under the hood if it detects that your attention inputs and hardware support it.
See its `documentation <https://deepwiki.com/Dao-AILab/flash-attention/1.1-installation-and-setup>`_ for installation instructions for this package,
the process is a bit more involved than for other backends.
You can install ``lgatr`` with the ``flash-attention`` extra as follows:

.. code-block:: python

    pip install lgatr[flash-attention]

PyTorch's native varlen attention
---------------------------------

PyTorch 2.10 natively includes a varlen attention kernel that is closely inspired by the original flash attention implementation.

More attention backends
-----------------------

L-GATr is designed to be flexible when it comes to attention backends.
If you want to use L-GATr with another attention backend, just open an
Issue on GitHub, or directly implement it in your own Pull Request!

.. [1] Tri Dao, Daniel Y. Fu, Stefano Ermon & Atri Rudra,
   *FlashAttention: Fast and Memory‐Efficient Exact Attention with IO‐Awareness*,
   NeurIPS 2022.
   `arXiv:2205.14135 <https://arxiv.org/abs/2205.14135>`_

.. [2] https://github.com/facebookresearch/xformers/issues/775
