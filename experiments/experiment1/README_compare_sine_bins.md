# compare_sine_bins.py

Back to [experiments overview](../README.md).

## Purpose
Compare ERP and PSD BIN outputs from EEGLAB with CortiPy-regenerated outputs. The script prints numeric summaries and can plot side-by-side traces.

## Inputs and data sources
Default EEGLAB files:
- `experiments/datasets/D2_software_curated_signals/ContinuousSine/EEGlab_erp_average.bin`
- `experiments/datasets/D2_software_curated_signals/ContinuousSine/EEGlab_psd_dB_continuous.bin`

Default CortiPy files:
- `experiments/experiment1/results_sine/bin/CortiPy_erp_average.bin`
- `experiments/experiment1/results_sine/bin/CortiPy_psd_dB.bin`

Both file types are expected to be axis+data in a single BIN, typically interleaved (axis0, data0, ...).

## What the script does
1. Loads each BIN as float64 and splits axis/data (interleaved or half).
2. For ERP and PSD separately, computes:
   - Min, max, mean, std for EEGLAB and CortiPy data.
   - Difference statistics (CortiPy minus EEGLAB).
   - Ratio at the maximum absolute EEGLAB point (scale sanity check).
   - Axis differences (max and mean).
3. Optionally plots four panels:
   - EEGLAB ERP
   - CortiPy ERP
   - EEGLAB PSD (dB)
   - CortiPy PSD (dB)

## Outputs and plots
- Console summary statistics for ERP and PSD.
- Optional 2x2 matplotlib plot grid.

## Example outputs (current results)
These plots are representative of the ERP/PSD shapes being compared.

![Sine average ERP](results_sine/sine_average.png)
![Sine PSD](results_sine/sine_psd.png)

## How to run
```bash
python experiments/experiment1/compare_sine_bins.py
python experiments/experiment1/compare_sine_bins.py --split half --no-plot
```

## Related experiments
- [save_sine_avg_psd](README_save_sine_avg_psd.md) produces the CortiPy BINs.
- [plot_eeglab_sine_bins](README_plot_eeglab_sine_bins.md) plots the EEGLAB reference BINs.
- [plot_sine_bins](README_plot_sine_bins.md) plots generic axis+data BIN files.
