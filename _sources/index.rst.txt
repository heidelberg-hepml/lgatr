L-GATr documentation
====================

The Lorentz-equivariant Geometric Algebra Transformer (L-GATr)
is a multi-purpose neural architecture for high-energy physics.
It uses spacetime geometric algebra representations throughout, allowing
to easily construct Lorentz-equivariant layers.
These layers are combined into a transformer architecture.
The L-GATr-slim variant further improves efficiency by using only
scalar and vector representations.

.. image:: /_static/gatr.png
   :align: center
   :width: 100%

This documentation describes the ``lgatr`` package,
available under https://github.com/heidelberg-hepml/lgatr.

* :doc:`quickstart`
* :doc:`quickstart_slim`
* :doc:`geometric_algebra`
* :doc:`attention_backends`
* :doc:`symmetry_breaking`
* :doc:`api`

Citation
--------

If you find this package useful, please cite our papers:

.. code-block:: bib

   @article{Brehmer:2024yqw,
      author = "Brehmer, Johann and Bres{\'o}, V{\'\i}ctor and de Haan, Pim and Plehn, Tilman and Qu, Huilin and Spinner, Jonas and Thaler, Jesse",
      title = "{A Lorentz-equivariant transformer for all of the LHC}",
      eprint = "2411.00446",
      archivePrefix = "arXiv",
      primaryClass = "hep-ph",
      reportNumber = "MIT-CTP/5802",
      doi = "10.21468/SciPostPhys.19.4.108",
      journal = "SciPost Phys.",
      volume = "19",
      number = "4",
      pages = "108",
      year = "2025"
   }
   @article{Petitjean:2025zjf,
      author = {Petitjean, Antoine and Plehn, Tilman and Spinner, Jonas and K{\"o}the, Ullrich},
      title = "{Economical Jet Taggers -- Equivariant, Slim, and Quantized}",
      eprint = "2512.17011",
      archivePrefix = "arXiv",
      primaryClass = "hep-ph",
      reportNumber = "IPPP/25/93",
      month = "12",
      year = "2025"
   }
   @inproceedings{spinner2025lorentz,
      title={Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics},
      author={Spinner, Jonas and Bres{\'o}, Victor and De Haan, Pim and Plehn, Tilman and Thaler, Jesse and Brehmer, Johann},
      booktitle={Advances in Neural Information Processing Systems},
      year={2024},
      volume={37},
      eprint = {2405.14806},
      url = {https://arxiv.org/abs/2405.14806}
   }
   @inproceedings{brehmer2023geometric,
      title = {Geometric Algebra Transformer},
      author = {Brehmer, Johann and de Haan, Pim and Behrends, S{\"o}nke and Cohen, Taco},
      booktitle = {Advances in Neural Information Processing Systems},
      year = {2023},
      volume = {36},
      eprint = {2305.18415},
      url = {https://arxiv.org/abs/2305.18415},
   }

.. toctree::
   :maxdepth: 1
   :caption: Usage
   :hidden:
   :titlesonly:

   quickstart
   quickstart_slim
   geometric_algebra
   attention_backends
   symmetry_breaking

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api
