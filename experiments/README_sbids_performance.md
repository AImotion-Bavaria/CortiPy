# sbids_performance.py

Back to [experiments overview](README.md).

## Purpose
Measure load performance when reading EEG data referenced by an SBIDS JSON-LD document via `cortipy.shared.SBIDSLoader`. Captures wall time, CPU time, CPU utilization, and memory usage.

## Inputs and data sources
- An SBIDS JSON-LD file (metadata describing recordings).
- Optional data roots to resolve referenced raw files.
- Optional recording @id filter.

This script does not generate data; it loads existing SBIDS descriptors and referenced raw files.

## What the script does
1. Parses recording IDs from the JSON-LD graph when `--all-recordings` is set.
2. Loads a single recording or all recordings with `SBIDSLoader.read_sbids`.
3. Measures:
   - Wall-clock time and CPU time.
   - CPU utilization and RSS peaks via background sampling (requires `psutil`).
4. Generates aggregate stats when benchmarking all recordings.
5. Optionally writes results to JSON and/or a PDF report with bar charts.

## Outputs and plots
- Console summary and JSON output (printed unless `--output` is used).
- Optional JSON file (`--output`).
- Optional PDF report (`--pdf-output`) with bar charts and summary text.

## How to run
```bash
python experiments/sbids_performance.py /path/to/recordings.jsonld --recording-id urn:recording:001
python experiments/sbids_performance.py /path/to/recordings.jsonld --all-recordings --data-root /data/raw
```

## Related experiments
- [bids_performance](README_bids_performance.md) benchmarks BIDS loading.
- [run_experiment2](format_benchmark/README_run_experiment2.md) measures write/read latency for SBIDS exports.
