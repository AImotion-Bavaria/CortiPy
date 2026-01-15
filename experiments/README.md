# Experiments Overview

This folder collects the runnable experiment scripts and small analysis utilities used to validate CortiPy, compare formats, and inspect synthetic data. Each experiment has its own README with detailed steps, inputs, and outputs.

## Quick links
- Background docs:
  - [Dataset resources](Dataset.md)
  - [Synthetic dataset specification](Syn_Datasets.md)
  - [Research questions and experiment outline](RQ_and_Ex.md)
- Synthetic data used by multiple experiments: `experiments/synData/` (ABR, ASSR, Oddball, VEP, SSVEP, ContinuousSine)

## Experiments (by theme)

## Quick Start (recommended order)
1. Generate and inspect sine outputs:
   - [save_sine_avg_psd](README_save_sine_avg_psd.md) (produces `results_sine/` plots)
   - [plot_eeglab_sine_bins](README_plot_eeglab_sine_bins.md) and [compare_sine_bins](README_compare_sine_bins.md)
2. Run Experiment 1 for validation and EEGLAB parity plots:
   - [run_experiment1](format_benchmark/README_run_experiment1.md)
3. Run format benchmarks:
   - [run_experiment2](format_benchmark/README_run_experiment2.md) (size/latency)
   - [run_experiment3](format_benchmark/README_run_experiment3.md) (streaming access)
4. Optional real-dataset load benchmarks:
   - [bids_performance](README_bids_performance.md)
   - [sbids_performance](README_sbids_performance.md)

### Synthetic sine inspection and EEGLAB parity
- [plot_continuous_sine](README_plot_continuous_sine.md)
- [save_sine_avg_psd](README_save_sine_avg_psd.md)
- [plot_sine_bins](README_plot_sine_bins.md)
- [plot_eeglab_sine_bins](README_plot_eeglab_sine_bins.md)
- [compare_sine_bins](README_compare_sine_bins.md)

### Load performance benchmarks
- [bids_performance](README_bids_performance.md)
- [sbids_performance](README_sbids_performance.md)

### Format benchmark suite
- [run_experiment1](format_benchmark/README_run_experiment1.md)
- [run_experiment1_V2](format_benchmark/README_run_experiment1_V2.md)
- [run_experiment2](format_benchmark/README_run_experiment2.md)
- [run_experiment3](format_benchmark/README_run_experiment3.md)

## Shared data conventions
- `experiments/synData/*` contains synthetic datasets with `params.json` and one or more `.bin` files. Several scripts expect specific filenames (for example `ContinuousSine_continuous.bin`).
- EEGLAB reference outputs live alongside the synthetic data (for example `EEGlab_erp_average.bin`, `EEGlab_psd_dB_continuous.bin`).
- Format benchmark outputs are written under `experiments/format_benchmark/results_exp*` and may be large; most scripts can delete artifacts automatically after measurements.

If you add a new experiment script, create a sibling README and link it here.
