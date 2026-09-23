# cortipy package

`cortipy` is the pure-Python implementation of the EEG Analysis Tool that used to live
inside MATLAB. The package bundles the measurement pipeline, module implementations,
device abstractions, evaluators, and minimal UI helpers so it can be published as its
own installable project.

```
cortipy/
├── core/        # MeasurementPipeline, ModuleContext, params helpers
├── devices/     # LSL, ActiCHamp-compatible, UNICORN, offline + dummy adapters
├── modules/     # Alpha, VEP, SSVEP, BERA, ASSR, P300, BCI acquisition loops
├── evaluation/  # Module-specific analysis and plotting code
├── shared/      # FFT, filters, plotting, report helpers, etc.
└── ui/          # Config loaders, save manager, and session runner
```

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

This installs `cortipy` in editable mode together with its runtime dependencies
(`numpy`, `scipy`, `pylsl`, `matplotlib`). Optional UI extras such as Streamlit can be
installed via `pip install streamlit`.

## Running measurements from Python

```python
from cortipy import MeasurementPipeline
from cortipy.core.pipeline import PipelineHooks

params = {
    "Method": "Alpha",
    "Device": "ActiCHamp",
    "Parameters": {
        "fs": 250,
        "RecordingTime": 60,
        "NumberEEGChannels": 8,
        "ReferenceChannel": 1,
        "TriggerChannel": 9,
        "Trigger": "Fixed",
        "TriggerTime": 10,
    },
    "Channels": [{"Position": f"Ch{i+1}"} for i in range(8)],
}

hooks = PipelineHooks(params_provider=lambda prev: params if prev is None else None)
MeasurementPipeline(hooks=hooks).run()
```

### Config files, runners, and UI

- `cortipy.ui.run_from_config` loads JSON/TOML configs (compatible with the legacy MATLAB
  `Params` structure) and executes the pipeline once.
- `cortipy.ui.SessionOptions` + `SaveManager` reproduce MATLAB’s `saveDatamain`
  behaviour, including timestamped run folders containing `params.json` + `data.npz`.
- `apps/streamlit_app.py` is the Streamlit UI entrypoint. It composes focused
  `cortipy/ui_streamlit/` modules for workflow, session forms, device settings, charts,
  live preview, data import, and electrodes.

### UNICORN hardware specifics

- Provide the UNICORN virtual COM port (for example `"COM7"` on Windows or
  `"/dev/tty.Unicorn-DevB"` on macOS) via `Params["Parameters"]["UNICORNPort"]`
  (aliases `UNICORNAddress` / `UnicornPort` are accepted for compatibility).
- Optional extras: `UNICORNDeviceName` (for logging) and `UnicornTimeout`
  (seconds). The adapter mirrors the MATLAB helpers, returning 16-column packets
  `[EEG(8), accel(3), gyro(3), battery, counter]`; modules continue to slice the
  EEG subset just like `data(:, 1:8)` in the original code.
- The serial protocol relies on `pyserial`, which ships with the package
  dependencies—no additional install steps required.

## Testing and validation

The `tests/regression` fixtures mirror deterministic MATLAB exports. Run `pytest` for
Python-only checks or `tests/run_parity_tests.m` inside MATLAB to generate fresh dummy
data and compare the legacy stack vs. the Python port. Continuous integration should
exercise at least the regression scripts before cutting releases.

## SBIDS import/export

- `SBIDSLoader` resolves an SBIDS `sbids_meta_<dataset>.jsonld` document plus its raw
  data into a `BIDSLoadResult` (e.g., `SBIDSLoader("sbids").read_sbids(...)`).
- `SbidsExporter` / `export_dataset` convert a CortiPy run folder containing
  `params.json` + data into SBIDS JSON-LD that can be validated with `sbids/test.py`.

```python
from cortipy.shared import SBIDSLoader, export_sbids_dataset

# Load an existing SBIDS export
result = SBIDSLoader("sbids", data_roots=["cortipy_runs"]).read_sbids(
    meta_path="sbids/sbids_meta_20251113-164440_SSVEP.jsonld"
)

# Export a run directory to SBIDS metadata
export_sbids_dataset(
    dataset_dir="cortipy_runs/20251113-164440_SSVEP",
    dataset_id="20251113-164440_SSVEP",
    dataset_name="20251113-164440_SSVEP",
    output="sbids/sbids_meta_20251113-164440_SSVEP.jsonld",
    indent=2,
)
```

## Performance experiments (BIDS)

`experiments/bids_performance.py` benchmarks loading a BIDS dataset and reports
wall-clock, CPU, and memory usage (psutil recommended). Example:

```bash
python experiments/bids_performance.py /path/to/bids_root --subject 01 --task rest --output /tmp/bids_perf.json
```

- Add `--pdf-output /tmp/report.pdf` for a chart-based PDF report (requires matplotlib).
- The script now prints a structured terminal summary; JSON is still written when `--output` is used.
- Use `--all-recordings` to load/benchmark every recording matching the filters (instead of only the first).

## Performance experiments (SBIDS)

`experiments/sbids_performance.py` benchmarks loading recordings referenced by an SBIDS
JSON-LD via `SBIDSLoader`. Examples:

```bash
# Single recording (first in the doc)
PYTHONPATH=. python experiments/sbids_performance.py /path/to/sbids_meta.jsonld \
  --output /tmp/sbids_perf.json --pdf-output /tmp/sbids_perf.pdf

# All recordings in the SBIDS document
PYTHONPATH=. python experiments/sbids_performance.py /path/to/sbids_meta.jsonld \
  --all-recordings --data-root /path/to/raw_root \
  --output /tmp/sbids_all_perf.json --pdf-output /tmp/sbids_all_perf.pdf
```
