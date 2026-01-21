# run_experiment1.py

Back to [experiments overview](../README.md).

## Purpose
Experiment 1 focuses on streaming and numerical validation. It validates BIN I/O integrity, runs a synthetic pipeline surrogate, and generates CortiPy plots to compare against EEGLAB reference figures for multiple paradigms.

## Inputs and data sources
- Synthetic datasets (paper naming):
  - D2: `experiments/datasets/D2_software_curated_signals/ContinuousSine/`
  - D4: `experiments/datasets/D4_sereega_evoked_potentials/` (ABR, ASSR, Oddball, VEP, SSVEP)
- EEGLAB reference PDFs in the same folders (copied into results for comparison).
- No external datasets required.

## What the script does
1. **Bitwise identity check**
   - Generates the D1 synthetic dataset `rand1`, exports it to BIN, reloads it, and re-exports it.
   - Compares SHA256 hashes of the two BIN exports to confirm byte-for-byte parity.
2. **Pipeline equivalence surrogate**
   - Uses `CortiDataset.synthetic_sine_trigger()` to synthesize a sine + trigger stream.
   - Epochs on triggers, computes ERP and PSD, and saves reference plots.
3. **Manual placeholders**
   - Writes notes for hardware-dependent steps (live acquisition, topography comparison).
4. **Dataset plot generation**
   - Loads each synthetic dataset from `experiments/datasets/` (D2 + D4).
   - Applies a standard montage and uses CortiPy evaluators to produce traces/topomaps/PSDs.
   - Copies EEGLAB PDFs into results and generates side-by-side image comparisons.
5. Writes a JSON summary into `results/`.

## Outputs and plots
Written under `experiments/experiment1/results/`:
- `bitwise_identity/` with hash report.
- `pipeline_equivalence/` plots: `evoked_cortipy.png` and `psd_cortipy.png` (and PDFs).
- `cortipy_plots/` with evaluator-generated images for each dataset.
- `eeglab_refs/` with copied EEGLAB reference PDFs/PNGs.
- Side-by-side comparison images for each dataset and plot type.
- `experiment1_results.json` with structured results and metadata.

## Example outputs (current results)
![Pipeline evoked](results/pipeline_equivalence/evoked_cortipy.png)
![Pipeline PSD](results/pipeline_equivalence/psd_cortipy.png)
![ABR trace comparison](results/comparisons/ABR_trace_comparison.png)
![SSVEP PSD comparison](results/comparisons/SSVEP_psd_comparison.png)

## How to run
```bash
PYTHONPATH=. python experiments/experiment1/run_experiment1.py
```

## Related experiments
- [run_experiment1_V2](README_run_experiment1_V2.md) is a variant with different plotting and styling.
- [compare_sine_bins](../README_compare_sine_bins.md) numerically compares EEGLAB vs CortiPy BINs.
