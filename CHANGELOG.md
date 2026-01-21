# Changelog

All notable changes to this project will be documented in this file.

## Unreleased
- Nothing yet.

## [v0.1.2] - 2026-01-21
### Highlights
- Restructured experiment data/layout: moved curated/synthetic datasets under `experiments/datasets/` (including D2/D3/D4), updated experiment READMEs/scripts to match, and cleaned up legacy result artifacts.
- Added `experiments/run_all_experiments.py` to generate paper outputs across experiments and consolidated documentation for running experiments.
- Relocated `ParameterJSON` assets under `apps/assets/` and updated Streamlit UI schema lookup accordingly.
- Refined ignore rules and repo hygiene (streamlined `.gitignore`, removed stale `cortipy_runs` placeholder).
- Updated BIDS dataset path reference in `cortipy/shared/bids.py` and added CI badge to `README.md`.

## [v0.1.1] - 2026-01-15
### Highlights
- Added comprehensive experiment documentation with linked readmes and example outputs.
- Fixed lint issues across core modules, experiments, scripts, and tests; aligned plotting exports.
- Improved Streamlit/evaluator plotting helpers and resolved missing references in tests.
- Updated CI dependency constraints for `pybv` across Python 3.9–3.12.
- Marked test packages for reliable imports during pytest collection.

## [v0.1.0] - 2025-12-16
### Highlights
- Added Apache-2.0 license and citation metadata.
- Added governance docs: Code of Conduct, Contributing guide, Security policy.
- Aligned dependencies across `pyproject.toml`, `requirements.txt`, and `environment.yml`.
- Added issue/PR templates plus support/roadmap notes in the README.
- Added clinical-use disclaimer and privacy notices in the README and Streamlit UI.
- Added version bump script for semantic versioning and cleaned repository artifacts (`cortipy_runs` placeholder).
- Added install matrix and hardware compatibility tables.

### Known issues
- Format-specific workflows still rely on optional dependencies (e.g., `pyedflib`, `pyarrow`/`fastparquet`, `h5py`, `zarr`); install them when using those formats.
