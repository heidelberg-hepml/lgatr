# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Planned July 2026

### Added

- `norm_elementwise_affine=True` option for all networks (changed default behavior)
- `sparse_gp=True` and `sparse_linear=True` options in `PrimitivesConfig` (changed default to `sparse_gp=True` because always faster)
- `nonlinearity_v` option for `LGATrSlim`/`ConditionalLGATrSlim` (changed default to `nonlinearity_v="sigmoid"` because more stable)
- `naive_amp=False` option and public `naive_amp` context manager to bypass `minimum_autocast_precision` and run the forward in the surrounding autocast dtype (e.g. bf16)
- `torch.compile` support for `LGATr`/`ConditionalLGATr`; `warmup_caches` helper for primitives under `mode="reduce-overhead"`
- `activation_memory_budget` option in `torch.compile` to trade backward FLOPs for a lower activation-memory peak (helps for `LGATrSlim`)
- Unit tests for all supported torch versions `torch>=2.4`; generally extended unit tests

### Fixed

- Consistently support `scalars=None`
- Set `requires_grad_(False)` for unused params to avoid `DDP` issues
- Unified docstrings, and format equations for sphinx readability
- Micro speed/memory optimizations for `LGATr`/`LGATrSlim` primitives
- `varlen` and `xformers` attention backends now support head dims that are not a multiple of 8 via zero-padding
- Custom access to `xformers` kernels to allow `torch.compile` `xformers` attention backend without graph breaks

### Changed

- `LGATrSlim`/`ConditionalLGATrSlim` hidden layers store vectors channel-last `(..., 4, channels)`
- Unify `get_nonlinearity()` between `LGATr`/`LGATrSlim`
- Enforce `torch>=2.4` autocast syntax, drop support for low-torch-2 versions
- Unify variable naming across the code; affects public API for conditional networks
- `PrimitivesConfig` is now a model input like `MLPConfig`, no global `gatr_config` anymore
- Renamed `PrimitivesConfig` flags: `use_fully_connected_subgroup`→`subgroup`, `use_bivector`→`bivector`, `use_geometric_product`→`geometric_product`
- `minimum_autocast_precision` return outputs as tuple and downcast to low dtype automatically
- Removed scalar bias from qkv linear layers in all models
- Slim stability refinements: initialize `linear_s` bias to 0, scale GLU inner product by 1/sqrt(4)
- Replaced the separate `compile_mode`/`compile_dynamic`/`compile_fullgraph` arguments with a single `compile_kwargs` dict forwarded verbatim to `torch.compile` in `compile_model` and all nets.
- Different amp strategy: vector/multivector path stays in fp32, only scalar path uses amp
- Unified the `LGATr` config names:  `increase_hidden_channels`->`attn_ratio`,  `increase_hidden_channels`->`mlp_ratio`, `activation`->`nonlinearity`, `num_hidden_layers`->`num_layers_mlp`
- Renamed slim building blocks with a `Slim` prefix (`SlimLinear` etc)

### Removed

- `einops`/`opt_einsum`/`numpy`/`lloca` requirements, now simply `torch>=2.4`

## [1.4.4] - 27.04.2026

### Added

- `compile_mode` and `compile_dynamic` argument for `LGATrSlim`

### Fixed

- Simpler indexing method for the gated nonlinearity to improve memory usage and speed
- Correct attention keywords in `ConditionalLGATrSlim`
- Import `LGATrSlim` and `ConditionalLGATrSlim` from `lgatr.nets`
- Improve `LGATrSlim`/`ConditionalLGATrSlim` amp handling

## [1.4.3] - 27.01.2026

### Added

- PyTorch 2.10's `varlen_attn` backend

### Changed

- Refactor `torch.compile` handling in attention
- Add `dtype` keyword argument to xformers attention to allow downcasting to float16/bfloat16 and enforcing flash-attention backends
- Improve `attention_backends` unit tests

### Fixed

- Bug in `minimum_autocast_precision`

## [1.4.2] - 08.01.2026

### Added

- FlashAttention varlen attention backend https://github.com/Dao-AILab/flash-attention
- References to L-GATr-slim paper

### Changed

- Collect optional requirements in `[dev]` extra
- Improvements in unit tests

## [1.4.1] - 22.12.2025

### Added

- `ConditionalLGATrSlim`

### Changed

- Fixed small typos etc in `lgatr` and `lgatr_slim` code
- Explain `lgatr_config` class in the demo notebook
- Update references in README

## [1.4.0] - 11.12.2025

### Added

- `LGATrSlim` network plus unit tests, example notebooks, docs
- Hidden variables `_out_mv_channels` etc in `EquiLinear` for convenient access

### Changed

- Ruff settings in pyproject.toml

## [1.3.3] - 18.11.2025

### Added

- Unit tests for `use_fully_connected_subgroup=False`

### Changed

- Improve autocast support (avoid nans; support old torch versions)
`- Drop `black` as formatter and fully move to `ruff`

### Fixed

- Correct install commands with extras, e.g. `pip install lgatr[xformers_attention]` -> `pip install lgatr[xformers-attention]` (pypi doesn't support `_` in package names)
- Subtle bug in `compute_pin_equi_linear_basis` triggered `when modifying `use_fully_connected_subgroup`

### Removed

- `requirements.txt` (already part of `pyproject.toml`)

## [1.3.2] - 29.10.2025

### Added

- `CHANGELOG.md`
- `.pre-commit-config.yaml` with `black`, `ruff` and `pre-commit`

### Changes

- Refactor everything based on ruff and black (using line-width 100 instead of 88)

## [1.3.1] - 16.10.2025

### Added

- Dynamic versioning based on git tags (update workflows and README)
- `activation=silu`

### Changes

- Defaults for `increase_hidden_channels` in (attention, MLP) changed from (2, 2) to (1, 4) because this is the usual convention
- Mention more repos that use `lgatr` in README

### Removed

- Option `mix_pseudoscalar_into_scalar` (now equal to `use_fully_connected_subgroup`)

## [1.3.0] - 07.06.2025

### Added

- Introduction to geometric algebra in `docs/`

### Changed

- Corrections and small additions in `docs/`
- Small refinements in code and tests

## [1.2.0] - 01.06.2025

### Added

- `docs/`
- `examples/demo_*.ipynb`
- Build-extras `xformers_attention`/`flex_attention` in `lgatr/primitives/attention_backends/`
- Codecov coverage tracking

### Changed

- Unify docstrings
- Refactor README
- Rename `GATrConfig` to `LGATrConfig`

### Fixed

- Bug in `pyproject.toml` that caused incorrect builds

## [1.0.3] - 27.05.2025

### Added

- `ConditionalLGATr`, `ConditionalLGATrBlock`, `CrossAttention`, `CrossAttentionConfig`
- Interface for axialvectors and pseudoscalars

## [1.0.2] - 02.04.2025

_Increment version._

## [1.0.1] - 02.04.2025

_Update README._

## [1.0.0] - 18.03.2025

_First release._
