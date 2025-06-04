Lorentz Symmetry Breaking
=========================

Why Lorentz Symmetry Breaking?
------------------------------

In many LHC contexts, Lorentz symmetry is only partially preserved. If this partial symmetry
breaking is not accounted for when applying a Lorentz-equivariant architecture, performance
can degrade significantly. This is because these architectures inherently treat inputs related by
global Lorentz transformations as equivalent. However, in the presence of symmetry breaking,
these inputs may carry different physical information. A fully Lorentz-equivariant architecture
is, by construction, blind to these differences, which can limit its effectiveness.

In LHC simulations, full Lorentz symmetry is only present in the first steps of the
simulation workflow, where no detector or reconstruction effects are applied. 
Once detector simulation and reconstruction (e.g. jet algorithms) are applied, 
Lorentz symmetry is typically broken in two ways

- Several effects break the full Lorentz symmetry :math:`SO(1,3)` down to the subgroup
  of rotations :math:`SO(3)`. Examples include the detector symmetry, which is not 
  invariant under boosts, and jet reconstruction algorithms which typically use
  the transverse momentum :math:`p_T`, which is only invariant under :math:`SO(2)`
  rotations around the beam axis.
- The detector geometry singles out the proton beam direction as a preferred spatial axis,
  breaking the :math:`SO(3)` rotation symmetry down to :math:`SO(2)` rotations around the beam axis. 

For these reasons, most problems in LHC physics only have a residual :math:`SO(2)` symmetry.

Lorentz Symmetry Breaking at the Input Level
--------------------------------------------

If most situations only require :math:`SO(2)` invariance, why should we care about
full Lorentz equivariance :math:`SO(1,3)` in the first place? Empirically, :math:`SO(1,3)`
equivariant models with symmetry breaking outperform :math:`SO(2)`-equivariant models.
For instance, the ParT and ParticleNet jet taggers are formally :math:`SO(2)`-equivariant,
because all their input features are :math:`SO(2)`-invariant -- but LorentzNet, PELICAN
and L-GATr still outperform them. 

This is because LorentzNet, PELICAN and L-GATr break Lorentz symmetry in a tuneable way: 
symmetry-breaking inputs are provided, and the network can decide to use them or not,
depending on the phase space region. In other words, symmetry breaking happens at the input
level, while the architecture itself remains fully Lorentz-equivariant. 

We discuss two kinds of symmetry-breaking inputs below. 
They can be used individually or together and also come with internal design choices,
the exact choice of symmetry breaking is a hyperparameter that has to be tuned for each problem.
For more details and studies on the impact of Lorentz symmetry breaking, 
see https://arxiv.org/abs/2411.00446.

Reference Vectors
~~~~~~~~~~~~~~~~~

Reference vectors (``spurions`` in our code) break Lorentz symmetry
by providing reference directions that should remain invariant under the unbroken 
symmetry group (typically :math:`SO(2)`). They should be regarded as part of the network
architecture rather than as part of the input data, because they are always appended to
the input data in the same way. For example, when applying data augmentation, one should
augment only the input particle data, and not the reference vectors. 
Reference vectors can be included as extra tokens or as extra channels.

The ``lgatr`` package provides functions to create the most common reference multivectors.
Note that the beam direction can be either encoded as a vector along the z-axis :math:`(0,0,0,1)`,
or as the plane orthogonal to the z-axis, which can be represented as a 
bivector :math:`(0,0,1,0,0,0,)` in the geometric algebra. Common choices for reference vectors
that break :math:`SO(1,3) \to SO(2)` are

.. code-block:: python

    from lgatr import get_spurions

    spurions = get_spurions(beam_spurion="xyplane", add_time_spurion=True)
    print(spurions.shape)  # (2, 16)

    spurions = get_spurions(beam_spurion="spacelike", add_time_spurion=True, beam_mirror=False)
    print(spurions.shape)  # (2, 16)

    spurions = get_spurions(beam_spurion="timelike", add_time_spurion=False, beam_mirror=True)
    print(spurions.shape)  # (2, 16)

Non-Invariant Scalars
~~~~~~~~~~~~~~~~~~~~~

Another approach is to include invariants under the unbroken subgroup (typically :math:`SO(2)`)
as `scalar` input features under the full Lorentz group. 
For example, one can embed :math:`E` (invariant under :math:`SO(3)`), :math:`p_T` 
(only invariant under :math:`SO(2)`) or :math:`\Delta R` (invariant under :math:`SO(2)`) 
as a scalar input feature in L-GATr. The Lorentz-equivariant architecture treats
these inputs as scalars by construction, leading to Lorentz symmetry breaking.

Implementing this approach is straightforward, just construct the unbroken-subgroup-invariant 
feature and pass it as a scalar input to the model.
