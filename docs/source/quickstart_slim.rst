Quickstart L-GATr-slim
======================

This page sets you up for building and running L-GATr-slim models.
You can execute this code with the `LGATrSlim <https://github.com/heidelberg-hepml/lgatr/blob/main/examples/demo_lgatr_slim.ipynb>`_ notebook.

Installation
------------

Before using the package, install it via pip:

.. code-block:: bash

   pip install lgatr

Alternatively, if you're developing locally:

.. code-block:: bash

   git clone https://github.com/heidelberg-hepml/lgatr.git
   cd lgatr
   pip install -e ".[dev]"
   pre-commit install

Building L-GATr-slim
--------------------

You can construct a simple :class:`~lgatr.nets.lgatr_slim.LGATrSlim` model as follows:

.. code-block:: python

   from lgatr import LGATrSlim

   lgatr = LGATrSlim(
      in_v_channels=1,
      out_v_channels=1,
      hidden_v_channels=8,
      in_s_channels=0,
      out_s_channels=0,
      hidden_s_channels=16,
      num_blocks=2,
      num_heads=1,
   )


Using L-GATr-slim
-----------------

Let's generate some toy data, you can think about it as a batch
of 128 LHC events, each containing 20 particles with mass 1,
represented by their four-momenta :math:`p=(E, p_x, p_y, p_z)` and pid.

.. code-block:: python

   import torch
   p3 = torch.randn(128, 20, 1, 3)
   mass = 1
   E = (mass**2 + (p3**2).sum(dim=-1, keepdim=True))**0.5
   p = torch.cat((E, p3), dim=-1)
   pid = torch.randint(high=3, size=p3.shape[:-1]).float()
   print(p.shape) # torch.Size([128, 20, 1, 4])
   print(pid.shape) # torch.Size([128, 20, 1])

Now we can use the model:

.. code-block:: python

   vectors = p
   scalars = pid
   output_v, output_s = lgatr(vectors=vectors, scalars=scalars)
   print(output_v.shape) # torch.Size([128, 20, 1, 4])
   print(output_s.shape) # torch.Size([128, 20, 1, 1])

Next steps
----------

- Have a look at the :doc:`api`
- Try the `LGATrSlim <https://github.com/heidelberg-hepml/lgatr/blob/main/examples/demo_lgatr_slim.ipynb>`_ notebook and test the `torch.compile` option.
- Custom :doc:`attention_backends`
- How to implement :doc:`symmetry_breaking`
