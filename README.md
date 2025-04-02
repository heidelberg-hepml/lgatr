<div align="center">

## Lorentz-Equivariant Geometric Algebra Transformer

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

</div>

This repository contains a minimal implementation of the **Lorentz-Equivariant Geometric Algebra Transformer (L-GATr)** by [Jonas Spinner](mailto:j.spinner@thphys.uni-heidelberg.de), [Víctor Bresó](mailto:breso@thphys.uni-heidelberg.de), Pim de Haan, Tilman Plehn, Huilin Qu, Jesse Thaler, and Johann Brehmer. L-GATr uses spacetime geometric algebra representations to construct Lorentz-equivariant layers and combines them into a transformer architecture.
You can read more about L-GATr in the following two papers:
- [Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics](https://arxiv.org/abs/2405.14806) (ML audience)
- [A Lorentz-Equivariant Transformer for All of the LHC](https://arxiv.org/abs/2411.00446) (HEP audience)

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
```
If you want a specific `branch` (e.g. the `xformers` or `flex_attention` branch), you can do `pip install https://github.com/heidelberg-hepml/lgatr.git@basics` or have a line `lgatr @ https://github.com/heidelberg-hepml/lgatr.git@basics` in your `requirements.txt`.

## How to use L-GATr

1. Instantiate the `LGATr` class. Hyperparameters related to attention and mlp blocks are organized in dataclasses, see `lgatr/layers/attention/config.py` and `lgatr/layers/mlp/config.py`. They can be initialized using dicts or these dataclass classes.
2. Embed the network inputs into the geometric algebra using functions from `lgatr/interface/`. You might want to use `spurions.py` to break Lorentz equivariance at the input level, see [Section 2.3 of the HEP paper](https://arxiv.org/abs/2411.00446) for a discussion on symmetry breaking and when it is needed.
3. Now you're ready to push your data through the L-GATr network!

More features:

- Global `LGATr` design choices are controlled by the `gatr_config` object from `lgatr/primitives/config.py`.
- L-GATr supports mixed precision. The critical operations are performed in `float32`.
- The default branch only has the default torch attention backend. There are seperate branches for the `xformers` and `flex_attention` backends. We do not include them in the main branch yet because of their additional requirements.

## Future

We are planning to extend this package in the future. If you would use them or you have more ideas, please use open an issue or a pull request.

- L-GATr transformer decoder using cross-attention.
- Add `docs`

## Examples

- https://github.com/heidelberg-hepml/lorentz-gatr: Original `LGATr` implementation used for the papers
- https://github.com/spinjo/weaver-core/blob/lgatr/weaver/nn/model/LGATr.py: L-GATr in the CMS boosted object tagging library `weaver`

Let us know if you use `lgatr`, so we can add your repo to the list!

## Citation

If you find this code useful in your research, please cite the following papers

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

