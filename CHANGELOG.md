# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.3] - 18.11.2025

### Changed

- Improve autocast support (avoid nans; support old torch versions)
`- Drop `black` as formatter and fully move to `ruff`

### Fixed

- Correct install commands with extras, e.g. `pip install lgatr[xformers_attention]` -> `pip install lgatr[xformers-attention]` (pypi doesn't support `_` in package names)

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
