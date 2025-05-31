L-GATr documentation
====================

The Lorentz-equivariant Geometric Algebra Transformer (L-GATr)
is a multi-purpose neural architecture for high-energy physics. 
It uses spacetime geometric algebra representations throughout, allowing 
to easily construct Lorentz-equivariant layers. 
These layers are combined into a transformer architecture.

.. image:: /_static/gatr.png
   :align: center
   :width: 100%

This documentation describes the accompanying ``lgatr`` package, 
available under https://github.com/heidelberg-hepml/lgatr.

* :doc:`quickstart`
* :doc:`architecture`
* :doc:`attentionbackends`
* :doc:`symmetrybreaking`
* :doc:`api`

Citation
--------

If you find this package useful, please cite our papers:

.. code-block:: bib

   @article{Brehmer:2024yqw,
      author = "Brehmer, Johann and Bres\'o, V\'\i{}ctor and de Haan, Pim and Plehn, Tilman and Qu, Huilin and Spinner, Jonas and Thaler, Jesse",
      title = "{A Lorentz-Equivariant Transformer for All of the LHC}",
      eprint = "2411.00446",
      archivePrefix = "arXiv",
      primaryClass = "hep-ph",
      reportNumber = "MIT-CTP/5802",
      month = "11",
      year = "2024"
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
   architecture
   attentionbackends
   symmetrybreaking

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api

