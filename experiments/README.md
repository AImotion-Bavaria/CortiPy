# Experiments Overview

This folder collects the runnable experiment scripts and small analysis utilities used to validate CortiPy, compare formats, and inspect synthetic data. Each experiment has its own README with detailed steps, inputs, and outputs.

## Quick links
- Background docs:
  - [Dataset resources](datasets/misc/Dataset.md)
  - [Synthetic dataset specification](datasets/misc/Syn_Datasets.md)
  - [Research questions and experiment outline](misc/RQ_and_Ex.md)
- Synthetic datasets (paper naming D1–D4): `experiments/datasets/` (see `experiments/datasets/README.md`)

## Experiments (by theme)

## Quick Start (recommended order)
1. Generate and inspect sine outputs:
   - [save_sine_avg_psd](experiment1/README_save_sine_avg_psd.md) (produces `experiment1/results_sine/` plots)
   - [plot_eeglab_sine_bins](experiment1/README_plot_eeglab_sine_bins.md) and [compare_sine_bins](experiment1/README_compare_sine_bins.md)
2. Run Experiment 1 for validation and EEGLAB parity plots:
   - [run_experiment1](experiment1/README_run_experiment1.md)
3. Run format benchmarks:
   - [run_experiment2](experiment2/README_run_experiment2.md) (size/latency)
   - [run_experiment3](experiment3/README_run_experiment3.md) (streaming access)
4. Run the paper-output pipeline:
   - `PYTHONPATH=. python experiments/run_all_experiments.py`
   - Add `--plot-all` to keep each experiment’s full plot suite; default only emits paper outputs into `experiments/paper_outputs/`.
5. Optional utilities and benchmarks:
   - [bids_performance](experiment2/misc/README_bids_performance.md)
   - [sbids_performance](experiment2/misc/README_sbids_performance.md)

## Paper outputs
Run `PYTHONPATH=. python experiments/run_all_experiments.py` to generate and collect:
- `extension_rw_throughput_box.pdf`
- `CortiPy_ABR_topomap_6ms.pdf`
- `CortiPy_ASSR_topomap_40Hz.pdf`
- `CortiPy_Oddball_topomap_300ms.pdf`
- `CortiPy_VEP_topomap_100ms.pdf`
- `CortiPy_SSVEP_topomap_10Hz.pdf`
- `D1_streaming_single_channel_latency_table.pdf`
- `SBIDS_load_performance.pdf`

### Synthetic sine inspection and EEGLAB parity
- [plot_continuous_sine](experiment1/README_plot_continuous_sine.md)
- [save_sine_avg_psd](experiment1/README_save_sine_avg_psd.md)
- [plot_sine_bins](experiment1/README_plot_sine_bins.md)
- [plot_eeglab_sine_bins](experiment1/README_plot_eeglab_sine_bins.md)
- [compare_sine_bins](experiment1/README_compare_sine_bins.md)

### Load performance benchmarks
- [bids_performance](experiment2/misc/README_bids_performance.md)
- [sbids_performance](experiment2/misc/README_sbids_performance.md)

### Format benchmark suite
- [run_experiment1](experiment1/README_run_experiment1.md)
- [run_experiment1_V2](experiment1/README_run_experiment1_V2.md)
- [run_experiment2](experiment2/README_run_experiment2.md)
- [run_experiment3](experiment3/README_run_experiment3.md)

## Shared data conventions
- D2 and D4 dataset folders contain `params.json` and one or more `.bin` files. Several scripts expect specific filenames (for example `ContinuousSine_continuous.bin`).
- EEGLAB reference PDFs and numeric exports live alongside the D2/D4 data (for example `EEGlab_erp_average.bin`, `EEGlab_psd_dB_continuous.bin`).
- Format benchmark outputs are written under `experiments/experiment2/results` and `experiments/experiment3/results` and may be large; most scripts can delete artifacts automatically after measurements.

If you add a new experiment script, create a sibling README and link it here.
