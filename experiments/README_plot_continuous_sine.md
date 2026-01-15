# plot_continuous_sine.py

Back to [experiments overview](README.md).

## Purpose
Quick visual sanity check of the ContinuousSine dataset stored in BIN format. It plots a short raw time series from one channel and the corresponding power spectral density (PSD).

## Inputs and data sources
- Default dataset directory: `experiments/synData/ContinuousSine/`
- Default BIN file: `ContinuousSine_continuous.bin`
- Metadata: `params.json` if present (used implicitly by `CortiDataset.from_bin`)
- Fallbacks if metadata is missing:
  - `--channel-count` (default 64)
  - `--sampling-rate` (default 1000 Hz)

## What the script does
1. Loads the BIN dataset with `CortiDataset.from_bin` and retrieves the MNE `Raw` object.
2. Selects one channel (default channel index 0; clamped to valid range).
3. Plots the first `--seconds` of that channel's time series.
4. Computes a Welch PSD on the same channel using MNE (`raw.compute_psd`).
5. Plots the PSD on a log scale.

## Outputs and plots
- Two on-screen matplotlib figures:
  - Time series in microvolts vs seconds (first N seconds).
  - PSD in microvolts^2/Hz vs frequency.
- No files are written unless you save figures manually from the UI.

## How to run
```bash
python experiments/plot_continuous_sine.py
python experiments/plot_continuous_sine.py --seconds 5 --channel 3
```

## Related experiments
- [save_sine_avg_psd](README_save_sine_avg_psd.md) creates averaged ERP/PSD outputs for the same dataset.
- [plot_eeglab_sine_bins](README_plot_eeglab_sine_bins.md) plots the EEGLAB reference BINs.
- [compare_sine_bins](README_compare_sine_bins.md) compares EEGLAB and CortiPy-derived BIN outputs.
