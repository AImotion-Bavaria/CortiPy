# Evaluator Plot Parity (from Experiment 1)

We need to bring the Experiment 1 plotting logic into the evaluators under `cortipy/evaluation` as `plot_*` helpers. Until then, Experiment 1 keeps ad‑hoc plotting.

## Missing data/metadata
- Oddball/VEP: no saved markers/stim channels in BIN/params; current Experiment 1 uses heuristics. We need real event markers or trial metadata to align traces/topomaps.
- VEP/Oddball: persist trial count/epoch length/fs in params.json (or sidecar) to avoid guesswork.
- Optional: add stim channel for synthetic datasets to drop heuristics entirely.

## Tasks per evaluator
- **ASSR (`cortipy/evaluation/assr.py`)**
  - Add `plot_assr_full_psd` matching Experiment 1 (0–500 Hz dB PSD with mild smoothing, edge-bin drop).
  - Add optional topomap helper if montage/info is available (power in dB, turbo colormap, fixed range −80 to −20, ~8 contours).
- **SSVEP (`cortipy/evaluation/ssvep.py`)**
  - Expose PSD plot (already added `plot_ssvep_power_db`, verify parity with Experiment 1 settings).
  - Add topomap helper like Experiment 1 (power in dB, turbo, −80 to −20, ~8 contours) when info is available.
- **ABR (`cortipy/evaluation/bera.py` or ABR path)**
  - Add event-locked trace plotting (0–15 ms, all trials gray + mean blue, axes −0.3–0.4 µV) and Wave V topomap.
- **VEP (`cortipy/evaluation/vep.py`)**
  - Add 0–500 ms single-trial overlay/mean plot (Oz), matching Experiment 1 baselines and limits.
  - Add 100 ms topomap.
- **Oddball (`cortipy/evaluation/p300.py` or relevant)**
  - Add standard/target ERP plot (0–600 ms, baseline and dip alignment) using real markers when available.
  - Add 300 ms topomap for target/standard (requires events/labels).

## Integration
- Keep plotting opt-in via `show_plots` to avoid altering existing analysis.
- Share common helpers (PSD smoothing, topomap styling) in a small utility to avoid duplication.
