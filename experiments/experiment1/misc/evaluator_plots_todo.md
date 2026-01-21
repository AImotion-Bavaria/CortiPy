# Evaluator Plot Parity (from Experiment 1)

We need to bring the Experiment 1 plotting logic into the evaluators under `cortipy/evaluation` as `plot_*` helpers. Until then, Experiment 1 keeps ad‑hoc plotting.

## Missing data/metadata
- Oddball/VEP: no saved markers/stim channels in BIN/params; current Experiment 1 uses heuristics. We need real event markers or trial metadata to align traces/topomaps.
- VEP/Oddball: persist trial count/epoch length/fs in params.json (or sidecar) to avoid guesswork.
- Optional: add stim channel for synthetic datasets to drop heuristics entirely.

## Tasks per evaluator
- **ASSR (`cortipy/evaluation/assr.py`)** ✅
  - Full PSD plot + optional dB topomap added.
- **SSVEP (`cortipy/evaluation/ssvep.py`)** ✅
  - PSD helper + optional dB topomap added.
- **ABR (`cortipy/evaluation/bera.py`)** ✅
  - Event-locked trace + optional 7 ms topomap added.
- **VEP (`cortipy/evaluation/vep.py`)** ✅
  - Oz overlay and 100 ms topomap added.
- **Oddball/P300 (`cortipy/evaluation/p300.py`)** ✅
  - Standard/target ERP overlay and 300 ms topomap added (uses markers when available; otherwise mean).

## Integration
- Keep plotting opt-in via `show_plots` to avoid altering existing analysis.
- Share common helpers (PSD smoothing, topomap styling) in a small utility to avoid duplication.
