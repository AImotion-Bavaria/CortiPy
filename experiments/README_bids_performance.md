# bids_performance.py

Back to [experiments overview](README.md).

## Purpose
Measure load performance when reading EEG data from a BIDS dataset via `cortipy.shared.BIDSLoader`. The script captures wall time, CPU time, CPU utilization, and memory usage.

## Inputs and data sources
- A BIDS dataset root directory on disk.
- Optional filters: subject, session, task, run.
- Optional list of allowed file extensions (defaults to common EEG formats).

This script does not generate data; it loads existing BIDS datasets.

## What the script does
1. Resolves recordings using `BIDSLoader` and the provided filters.
2. Measures load time and resource usage while calling `read_bids`:
   - Wall-clock time (`time.perf_counter`).
   - CPU time (`time.process_time`).
   - CPU utilization and RSS peaks sampled in a background thread (requires `psutil`).
3. Supports two modes:
   - Single recording (first match).
   - All matching recordings (aggregated summary + per-recording stats).
4. Optionally writes results to JSON and/or a PDF report with bar charts.

## Outputs and plots
- Console summary and JSON output (printed unless `--output` is used).
- Optional JSON file (`--output`).
- Optional PDF report (`--pdf-output`) with bar charts and summary text.

## How to run
```bash
python experiments/bids_performance.py /path/to/bids_root --subject 01 --task oddball
python experiments/bids_performance.py /path/to/bids_root --all-recordings --output results/bids_perf.json
```

## Related experiments
- [sbids_performance](README_sbids_performance.md) benchmarks SBIDS JSON-LD loading.
- [run_experiment2](format_benchmark/README_run_experiment2.md) measures write/read latency for exported BIDS formats.
