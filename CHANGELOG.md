# Changelog

All notable changes to this project will be documented in this file.

## Unreleased
- Nothing yet.

## [0.1.0] - 2025-12-16
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
