# plot_eeglab_sine_bins.py

Back to [experiments overview](../README.md).

## Purpose
Visualize the EEGLAB reference ERP and PSD BIN files for the synthetic ContinuousSine dataset. The script handles multiple axis/data storage layouts.

## Inputs and data sources
Default BINs:
- `experiments/datasets/D2_software_curated_signals/ContinuousSine/EEGlab_erp_average.bin`
- `experiments/datasets/D2_software_curated_signals/ContinuousSine/EEGlab_psd_dB_continuous.bin`

Both files store axis and data in the same file; the script can split them as interleaved, concatenated halves, or Fortran-ordered pairs.

## What the script does
1. Loads the raw BIN files as float64 arrays.
2. Splits axis and data using one of the supported modes:
   - `interleaved`: axis0, data0, axis1, data1, ...
   - `half`: first half axis, second half data
   - `fortran`: 2-column Fortran-ordered reshape
   - `auto`: chooses the split with a monotonic axis and largest span
3. Plots ERP vs time and PSD (dB) vs frequency.

## Outputs and plots
- Two on-screen matplotlib subplots:
  - ERP (time vs amplitude)
  - PSD in dB (frequency vs power)

## Example outputs (current results)
No pre-rendered EEGLAB sine plots are stored in the repo. The plots below show the typical ERP/PSD shapes produced by [save_sine_avg_psd](README_save_sine_avg_psd.md); run this script to see the EEGLAB BINs directly.

![Sine average ERP](results_sine/sine_average.png)
![Sine PSD](results_sine/sine_psd.png)

## How to run
```bash
python experiments/experiment1/plot_eeglab_sine_bins.py
python experiments/experiment1/plot_eeglab_sine_bins.py --split auto
```

## Related experiments
- [save_sine_avg_psd](README_save_sine_avg_psd.md) generates CortiPy ERP/PSD BIN outputs.
- [plot_sine_bins](README_plot_sine_bins.md) plots BIN outputs from other pipelines.
- [compare_sine_bins](README_compare_sine_bins.md) compares EEGLAB vs CortiPy outputs.
