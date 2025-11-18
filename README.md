<div align="center">

## Lorentz-Equivariant Geometric Algebra Transformer

[![Tests](https://github.com/heidelberg-hepml/lgatr/actions/workflows/tests.yaml/badge.svg)](https://github.com/heidelberg-hepml/lgatr/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/heidelberg-hepml/lgatr/branch/main/graph/badge.svg)](https://codecov.io/gh/heidelberg-hepml/lgatr)
[![PyPI version](https://img.shields.io/pypi/v/lgatr.svg)](https://pypi.org/project/lgatr)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/lgatr.svg)](https://anaconda.org/conda-forge/lgatr)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

[![LGATr-CS](http://img.shields.io/badge/paper-arxiv.2405.14806-B31B1B.svg)](https://arxiv.org/abs/2405.14806)
[![LGATr-HEP](http://img.shields.io/badge/paper-arxiv.2411.00446-B31B1B.svg)](https://arxiv.org/abs/2411.00446)

</div>

This repository contains a standalone implementation of the **Lorentz-Equivariant Geometric Algebra Transformer (L-GATr)** by [Jonas Spinner](mailto:j.spinner@thphys.uni-heidelberg.de), [Víctor Bresó](mailto:breso@thphys.uni-heidelberg.de), Pim de Haan, Tilman Plehn, Huilin Qu, Jesse Thaler, and Johann Brehmer. L-GATr uses spacetime geometric algebra representations to construct Lorentz-equivariant layers and combines them into a transformer architecture.
You can read more about L-GATr in the following two papers and in the [L-GATr documentation](https://heidelberg-hepml.github.io/lgatr/):
- [Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics](https://arxiv.org/abs/2405.14806) (ML audience)
- [A Lorentz-Equivariant Transformer for All of the LHC](https://arxiv.org/abs/2411.00446) (HEP audience)

![](img/gatr.png)

## Installation

You can either install the latest release using pip
```
pip install lgatr
```
or clone the repository and install the package in dev mode
```
git clone https://github.com/heidelberg-hepml/lgatr.git
cd lgatr
pip install -e .
pre-commit install
```

## How to use L-GATr

Please have a look at the [L-GATr documentation](https://heidelberg-hepml.github.io/lgatr/) and our example notebooks for [LGATr](examples/demo_lgatr.ipynb) and [ConditionalLGATr](examples/demo_conditional_lgatr).

Overview of features in L-GATr:

- L-GATr encoder and decoder as `LGATr` and `ConditionalLGATr`
- Additional attention backends, installation via `pip install lgatr[xformers-attention]` and `pip install lgatr[flex-attention]`. You might have to run `python -m pip install --upgrade pip setuptools wheel
` because extra imports require modern versions of `pip, setuptools, wheel`.
- Support for torch's automatic mixed precision; critical operations are performed in `float32`
- Interface to the geometric algebra: Embedding and extracting multivectors; spurions for symmetry breaking at the input level
- Many hyperparameters to play with, organized via the `SelfAttentionConfig`, `CrossAttentionConfig`, `MLPConfig` and `LGATRConfig` objects

## Examples

- https://github.com/spinjo/weaver-core/blob/lgatr/weaver/nn/model/LGATr.py: L-GATr in the CMS boosted object tagging library `weaver`. Includes examples for how to use L-GATr without the `xformers` package.
- https://github.com/heidelberg-hepml/lloca-experiments: Code for the LLoCa project, including L-GATr as a baseline. The main results from https://github.com/heidelberg-hepml/lorentz-gatr can be reproduced here using the `lgatr` package.
- https://github.com/heidelberg-hepml/high-dim-unfolding: Generative jet substructure unfolding with L-GATr, uses the `ConditionalLGATr`.
- https://github.com/heidelberg-hepml/lorentz-gatr: Original `LGATr` implementation used for the papers. This repo doesn't import the `lgatr` package, but has its own (outdated) `lgatr/` folder.

Let us know if you use `lgatr`, so we can add your repo to the list!

## Contributing

Contributions are welcome! To get started:

1. Fork this repository and create a new branch for your feature or fix.
2. Make your changes, following the existing code style (and using `pre-commit`).
3. Add or update tests where appropriate.
4. Open a pull request with a clear description of your changes.

If you’re not sure where to begin, feel free to open an issue to discuss your idea first.

## Citation

If you find this code useful in your research, please cite our papers

```bibtex
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
```
