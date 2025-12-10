# cortipy package

`cortipy` is the pure-Python implementation of the EEG Analysis Tool that used to live
inside MATLAB. It bundles the measurement pipeline, acquisition modules, device
adapters, evaluators, and a Streamlit UI so the stack can be installed as a standard
Python package.

```
cortipy/
├── core/        # MeasurementPipeline, ModuleContext, params helpers
├── devices/     # UNICORN, ActiCHamp (Win), LSL, offline + dummy adapters
├── modules/     # Alpha, VEP, SSVEP, BERA, ASSR, P300, BCI acquisition loops
├── evaluation/  # Module-specific analysis and plotting code
├── shared/      # FFT, filters, plotting, report helpers, etc.
└── ui/          # Config loaders, Streamlit UI, session runner, save helpers
```

## Installation

Install in editable mode with optional UI extras:

```bash
python -m venv .venv
source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .            # core toolkit
pip install -e .[ui]        # adds Streamlit dependencies
# Add `[bids]` to enable the BIDS loader/exporter extras (EDF/BDF, Parquet, HDF5, Zarr)
pip install -e .[bids]
pip install -e .[ui,bids]   # UI plus BIDS extras
```

For conda users, an environment spec is provided. It pins the same runtime
dependencies and installs `cortipy` from the local checkout via pip:

```bash
conda env create -f environment.yml
conda activate cortipy
```

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
- `apps/streamlit_app.py` provides a Streamlit interface for configuring params,
  naming electrodes, selecting devices, and launching/monitoring measurements with live
  previews. Saved runs appear in the sidebar so you can browse and reload them.

### Running the Streamlit UI

```bash
pip install -e .[ui]          # once per environment
streamlit run apps/streamlit_app.py
```

Open `http://localhost:8501` (default Streamlit port) and configure a run:
- Pick a method and device, fill in participant metadata, and assign electrodes before starting.
- Set the sidebar “Save directory” (defaults to `./cortipy_runs`); each run gets its own timestamped folder.
- Use “Simulate run” to exercise the UI without hardware, or “Use imported data” to replay `.npz` exports offline.
- The “Preview” tab shows the assembled params and lets you download `params.json` for later scripting.
- Finished runs appear under “Saved sessions” so you can reload params and browse results.
- App logs live in `streamlit_app.log` next to the repo; keep it handy when debugging device connections.

### Device specifics

- **UNICORN**: Provide the virtual COM port (for example `"COM7"` on Windows or
  `"/dev/tty.Unicorn-DevB"` on macOS) via `Params["Parameters"]["UNICORNPort"]`
  (aliases `UNICORNAddress` / `UnicornPort` are accepted). Optional extras:
  `UNICORNDeviceName` for logging and `UnicornTimeout` (seconds). The adapter returns
  16-column packets `[EEG(8), accel(3), gyro(3), battery, counter]`.
- **ActiCHamp**: Windows-only because the vendor SDK uses Win32 shared memory and DLLs.
  Binaries ship in `cortipy/devices/actichamp`; override with
  `Params["ActiChampPath"]` if needed. Supply `fs`, `NumberEEGChannels`, and optional
  `NumberAUXChannels`; triggers follow EEG/AUX data.
- **LSL / Dummy / Offline**: Use `Device` = `"LSL"` for any LabStreamingLayer EEG
  stream, `"Dummy"` (or `"Sim"`) for synthetic data, or `"Offline"` to replay
  `.npz/.mat` files. All are selectable from the Streamlit UI for testing without
  hardware.

### BIDS import/export

- `cortipy.shared.BIDSLoader` can read/write BIDS datasets. Supported inputs include EDF/BDF, BrainVision (`.vhdr/.eeg`), EEGLAB (`.set`), FIF, Parquet, HDF5, and Zarr.
- `cortipy.shared.ExperimentBinLoader` reads/writes the `.bin` + `params.json` pairs used in `experiments/synData`, returning a `BIDSLoadResult` so you can analyse or re-export them.
- Extra dependencies for non-default formats: `pyedflib` (EDF/BDF export), `pyarrow` (Parquet), `h5py` (HDF5), `zarr` (Zarr); BrainVision/EEGLAB exports still rely on `pybv` / `eeglabio`.
- Install them via `pip install -e .[bids]` (or combine with `[ui]`) to enable all BIDS I/O features.

```python
from cortipy.shared import BIDSLoader, ExperimentBinLoader

bin_loader = ExperimentBinLoader("experiments/synData")
result = bin_loader.read_bin("Oddball")  # loads params.json + Oddball_scalpdata.bin

# Convert to BIDS
BIDSLoader("exports/oddball_bids").to_bids(result.raw, subject="01", task="oddball", overwrite=True)

# Convert any Raw/numpy data back into the bin layout
bin_loader.write_bin(result.raw, "exports/oddball_bin", params=result.metadata["params"], overwrite=True)
```

## Testing and validation

The `tests/regression` fixtures mirror deterministic MATLAB exports. Run `pytest` for
Python-only checks or `tests/run_parity_tests.m` inside MATLAB to generate fresh dummy
data and compare the legacy stack vs. the Python port. Continuous integration should
exercise at least the regression scripts before cutting releases.
