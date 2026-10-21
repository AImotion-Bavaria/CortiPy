# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## [v0.2.3] - 2026-10-21
### Highlights
- VEP live plot, module and evaluator follow ISCEV: no baseline correction; epochs are cut at the trigger sample via the new `segment_at_triggers`, which keeps the last complete epoch.
- Live VEP average: 0.4 s epochs, 0–300 ms view, and the first 10 ms are skipped (the amplifier start-up ramp halved the detected reversal rate and collapsed Fsp).
- JSON-LD import accepts "Ch1" slot names, keeps GND as ground and drops the trigger from the electrode table; `TriggerChannel` help names the actiCHamp input number (33 = AUX 1).
- actiCHamp: live impedance works after "Connect device"; the producer is restarted when it is already streaming.
- Export keeps electrode names in the saved columns; the 50 Hz notch on export was dropped.
- Removed the Brain Products Amplifier SDK headers, import library, documentation and samples, and the MATLAB/MEX helpers, from `cortipy/devices/actichamp`; only the runtime files and the producer source remain. Added `build.bat` and third-party license notes.
- Rewrote `README.md` and `cortipy/devices/actichamp/README.md`.
- Capped `pyarrow` below 26, which requires NumPy 2.

## [v0.2.2] - 2026-09-23
### Highlights
- Acquisition version for the VEP in-ear study: every SBIDS recording stores the CortiPy version and source revision it was recorded with.

## [v0.2.1] - 2026-09-23
### Highlights
- Streamlit UI rewrite: `apps/streamlit_app.py` split into `cortipy/ui_streamlit/` modules; guided sidebar, setup progress checklist and diagnostics panel; `scripts/run_ui.py` launcher for macOS Apple Silicon.
- SBIDS export: JSON-LD + Parquet, one JSON-LD per dataset folder accumulating all runs, with participant metadata, full montage, VEP Fsp and recording start time; lossless BIDS/SBIDS round trip.
- New/Load dataset workflow: recordings accumulate in one folder; run folders are named after the typed filename and numbered instead of overwritten.
- Electrodes and impedance: electrode configuration merged into the session config; live actiCHamp impedance before and during acquisition; electrode table shows exactly the recorded channels, with quick-fill from the standard 10-20 montage.
- Devices: actiCHamp trigger and live-stream reference fixes; Unicorn Bluetooth port resolved via the device MAC (Windows) and stream recovery after a missed ack.
- Signal fixes: referencing keeps non-EEG channels; corrected single-sided FFT bins; SSVEP multichannel PSD and `stim_freqs` fixes.
- Added the `CortiPy_VEP_Ear_Study` submodule.
- CI: Ruff passes (target py39, submodule excluded, documented ignore list); lint no longer blocks the tests; removed Dependabot.
- Updated dependency bounds; `cortipy_runs/` ignored at any depth.


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
