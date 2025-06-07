Spacetime Geometric Algebra
===========================

Scalar and Vector Data
----------------------

The motivation for using Lorentz-equivariant architectures is to hard-wire
non-trivial symmetry properties of input or output data into the network architecture.
The most common example in high-energy physics are the four-momenta of particles,
which are Lorentz vectors. Usually there is still also scalar data available,
e.g. particle type and other classification scores. 
Before using Lorentz-equivariant
architectures, it is essential to identify the transformation behaviour 
of the network inputs and outputs.

Spacetime Geometric Algebra
---------------------------

When building Lorentz-equivariant architectures, the naive approach is to 
handle scalar and vector representations seperately. L-GATr takes
a different approach and uses (spacetime) geometric algebra representations
to unify these representations.
These representations combines scalar and vector representations with 
higher-order representations into so-called multivector representations. 
Using Minkowski index notation, a multivector :math:`x` of the spacetime 
geometric algebra can be written as

.. math::
    x = x_0 + x_\mu^V \gamma^\mu + x_{\mu\nu}^B \sigma^{\mu\nu} + x_\mu^A \gamma^\mu\gamma^5 + x^P \gamma^5.

The 5 terms in this equation represent the 5 `grades` scalar, vector, antisymmetric tensor, axialvector, 
and pseudoscalar, which form subrepresentations of the Lorentz group.
L-GATr internally performs operations on the 16-dimensional object
:math:`(x^S, x_\mu^V, x_{\mu\nu}^B, x_\mu^A, x^P)`, which fully characterizes a multivector.
For more information, have a look at https://arxiv.org/abs/2411.00446 and https://arxiv.org/abs/2405.14806.

The L-GATr layers mix information of the different grades without violating Lorentz equivariance.
Additionally, L-GATr constrains this quite restrictive setup of working solely only multivectors by allowing
additional scalar representations which mix with the scalar part of multivector representations.

Embedding data in and extracting data from multivectors
-------------------------------------------------------

The `lgatr` package provides tools for embedding and extracting spactime objects 
into spacetime geometric algebra representations in :mod:`~lgatr.interface.scalar`,
:mod:`~lgatr.interface.vector`, :mod:`~lgatr.interface.axialvector` and  
:mod:`~lgatr.interface.pseudoscalar`. For each representation, we have an embedding 
and an extraction function. The embedding function takes the scalar/vector/axialvector/pseudoscalar
and embeds it into a multivector, while zero-padding the other multivector entries.
The extraction function returns the specified part of the multivector, ignoring
non-zero entries in all other representations. For example:

.. code-block:: python
    
    import torch
    from lgatr.interface import embed_vector, extract_vector, extract_scalar

    vector = torch.randn(1, 4)
    multivector = embed_vector(vector)
    print(multivector.shape)  # torch.Size([1, 16])

    vector2 = extract_vector(multivector) # same as vector
    scalar = extract_scalar(scalar) # torch.Size([1, 1])
    print(scalar) # tensor([[0.]])

L-GATr follows the convention of the 
`clifford package <https://clifford.readthedocs.io/en/latest/>`_
for how multivectors are represented.
