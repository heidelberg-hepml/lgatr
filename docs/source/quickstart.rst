Quickstart
==========

This page sets you up for building and running L-GATr models.

Installation
------------

Before using the package, install it via `pip`:

.. code-block:: bash

   pip install lgatr

Alternatively, if you're developing locally:

.. code-block:: bash

   git clone https://github.com/heidelberg-hepml/lgatr.git
   cd lgatr
   pip install -e .

Building L-GATr
---------------

You can construct a simple :class:`~lgatr.nets.lgatr.LGATr` model as follows:

.. code-block:: python

   from lgatr import (
      LGATr, 
      SelfAttentionConfig, 
      MLPConfig,
   )

   attention = SelfAttentionConfig(num_heads=2)
   mlp = MLPConfig()
   net = LGATr(
      in_mv_channels=1,
      out_mv_channels=1,
      hidden_mv_channels=8,
      in_s_channels=0,
      out_s_channels=0,
      hidden_s_channels=16,
      attention=attention,
      mlp=mlp,
      num_blocks=2,
   )

The :class:`~lgatr.layers.attention.config.SelfAttentionConfig` 
and :class:`~lgatr.layers.mlp.config.MLPConfig` classes allow you to modify
the default :class:`~lgatr.nets.lgatr.LGATr` hyperparameters.

Using L-GATr
------------

Let's generate some toy data, you can think about it as a batch of 128 LHC events, each
containing 20 particles with mass 1, represented by their four-momenta :math:`p=(E, p_x, p_y, p_z)`.

.. code-block:: python

   p3 = torch.randn(128, 20, 1, 3)
   mass = 1
   E = (mass**2 + (p3**2).sum(dim=-1, keepdim=True))**0.5
   p = torch.cat((E, p3), dim=-1)
   print(p.shape) # torch.Size([128, 20, 1, 4])

To use L-GATr, we have to embed these four-momenta into multivectors. We can use functions from
:mod:`lgatr.interface` to do this.

.. code-block:: python

   from lgatr.interface import embed_vector, extract_scalar
   multivector = embed_vector(p)
   print(multivector.shape) # torch.Size([128, 20, 1, 16])

Now we can use the model:

.. code-block:: python

   output_mv, output_s = net(multivectors=multivector, scalars=None)
   out = extract_scalar(output_mv)
   print(out.shape) # torch.Size([128, 20, 1, 1])

We only used the multivector input and output channels of :class:`~lgatr.nets.lgatr.LGATr`
for this test, but you can also use scalar inputs and outputs. 

Next steps
----------

- Have a look at the :doc:`api`
- Demo notebooks (TODO)
- Custom attention kernels (TODO)
- How to use symmetry-breaking inputs (TODO)
