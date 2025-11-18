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
- `apps/streamlit_app.py` provides the Streamlit UI that recreates the MATLAB
  configuration dialog (method/device params, proband metadata, electrode naming) and
  gives you start/stop controls plus saved-session browsing.

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

### ActiCHamp hardware specifics

- Requires Windows because the vendor SDK uses Win32 shared memory and DLLs.
- The shared-memory producer and DLLs are bundled in `cortipy/devices/actichamp`;
  set `Params["ActiChampPath"]` to override the location if needed.
- Provide `Params["Parameters"]["fs"]` (sampling rate), `NumberEEGChannels`, and
  optional `NumberAUXChannels`; triggers are appended after the EEG/AUX channels.
- The adapter launches `EEG_SharedMemoryProducer.exe`, writes the target sampling
  rate, waits for the `acquisitionReady` flag, and reads from the ring buffer
  without altering the vendor math or DLLs.

## Testing and validation

The `tests/regression` fixtures mirror deterministic MATLAB exports. Run `pytest` for
Python-only checks or `tests/run_parity_tests.m` inside MATLAB to generate fresh dummy
data and compare the legacy stack vs. the Python port. Continuous integration should
exercise at least the regression scripts before cutting releases.
