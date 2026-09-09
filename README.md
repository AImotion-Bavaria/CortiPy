# cortipy package

[![CI](https://github.com/AwesomeEEGTools/CortiPy/actions/workflows/ci.yml/badge.svg)](https://github.com/AwesomeEEGTools/CortiPy/actions/workflows/ci.yml)

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

## Clinical use and privacy

- CortiPy is for research and educational use only; it is not a medical device and must not be used for diagnosis or patient care.
- Follow institutional approvals and local regulations before any clinical evaluation; validate device interoperability and trigger/latency behavior in your lab.
- Do not store or share identifiable participant data in repositories or issue trackers. Anonymize runs and keep PHI on secured systems; use synthetic/anonymized data for examples.
- Questions or incident reports: joh1391@thi.de, Rahul.Mondal@thi.de, Laurens.Kreilinger@thi.de.

## Installation matrix

| Platform | pip | conda | Notes |
| --- | --- | --- | --- |
| Windows 10/11 | ✅ `pip install -e .[ui,bids,sbids]` | ✅ `conda env create -f environment.yml` | ActiCHamp binaries supported only on Windows; Streamlit UI works. |
| macOS (Apple/Intel) | ✅ | ✅ | UNICORN supported; ActiCHamp not supported (Windows-only SDK). |
| Linux (Ubuntu 22.04+) | ✅ | ✅ | UNICORN supported; ActiCHamp not supported (Windows-only SDK). |

## Hardware compatibility

| Device | Windows | macOS | Linux | Notes |
| --- | --- | --- | --- | --- |
| UNICORN | ✅ | ✅ | ✅ | Uses virtual COM/Bluetooth serial; verify port name (e.g., `COM7`, `/dev/tty.Unicorn-DevB`). |
| ActiCHamp | ✅ | ❌ | ❌ | Windows-only due to vendor SDK binaries shipped in `cortipy/devices/actichamp`. |
| LSL streams | ✅ | ✅ | ✅ | Any LabStreamingLayer EEG source. |
| Dummy/Sim | ✅ | ✅ | ✅ | For UI/tests without hardware. |
| Offline replay | ✅ | ✅ | ✅ | Replays `.npz/.mat`/BIDS data. |

## Support and triage

- Issue/PR templates live in `.github/ISSUE_TEMPLATE` and `.github/pull_request_template.md`; please avoid sharing PHI in tickets.
- Response target: within 5 business days for new issues and vulnerability reports (use the contacts above).
- For clinical/academic collaborations, reach out via Rahul.Mondal@thi.de or Laurens.Kreilinger@thi.de.

## Roadmap (abridged)

- CI/CD hardening: lint/type/coverage gates, release automation to PyPI.
- Streamlit UX: impedance/latency display, clearer device status, structured logging.
- Docs: quickstart, device setup, BIDS/SBIDS workflows, troubleshooting.
- Data governance: anonymization guidance, SBOM/dependency provenance for clinical reviews.
- Packaging: signed wheels/sdists, unified versioning and changelog.

## Format benchmark experiments

- **Experiment 1 – Streaming & numerical validation** (`experiments/experiment1/run_experiment1.py`): `PYTHONPATH=. python ...` with no flags. Checks bitwise identity of BIN exports, runs a synthetic sine pipeline surrogate, and renders CortiPy plots for ABR/ASSR/Oddball/VEP/SSVEP/Sine10Hz to compare against EEGLAB references. Outputs live under `experiments/experiment1/results/` (JSON + PNG/PDFs). Hardware/EEGLAB numeric correlations are documented as manual placeholders.
- **Experiment 2 – File format storage & latency** (`experiments/experiment2/run_experiment2.py`): measures export size and read/write latency across BIDS/SBIDS containers and formats (parquet/edf/zarr/hdf5) using synthetic 1 h datasets (19/64/256 ch). Default: run all datasets, containers, and formats; results saved to `experiments/experiment2/results/experiment2_results.json` with plots. Key flags: `--datasets` labels, `--containers` bids|sbids, `--formats` parquet|edf|zarr|hdf5, `--runs` N, `--keep-artifacts`, `--no-purge`, `--plot-scope all|synthetic|bin`, `--plots-only`, `--results-json PATH`.
- **Experiment 3 – Streaming access benchmark** (`experiments/experiment3/run_experiment3.py`): times streaming-style reads (open handle, single channel, random windows) for the same synthetic datasets exported to BIDS/SBIDS formats. Default: all datasets, containers, formats; results to `experiments/experiment3/results/experiment3_results.json` with plots. Key flags: `--datasets`, `--containers`, `--formats`, `--runs`, `--window-s` (window duration), `--channel-idx` (0-based), `--keep-artifacts`, `--no-purge`, `--plots-only`, `--results-json PATH`.

## Documentation

- Quickstart, API reference, and workflows will live in a docs site (MkDocs/Sphinx). For now, see the README and in-code docstrings.
- Release notes are maintained in `CHANGELOG.md`.

## Installation

Install with Python 3.9-3.12 in editable mode with optional UI extras:

```bash
# Windows example with the Python launcher:
py -3.12 -m venv .venv

# macOS/Linux example:
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
- `apps/streamlit_app.py` is the Streamlit entrypoint. It composes focused helpers in
  `cortipy/ui_streamlit/` for workflow dashboard, session forms, device settings, live
  preview, charts, imports, and electrode handling.

### Running the Streamlit UI

```bash
pip install -e .[ui]          # once per environment
streamlit run apps/streamlit_app.py
```

On **macOS (Apple Silicon)** use the launcher instead — plain `streamlit run` can crash on
first use because numpy 1.26 runs a subprocess when `numpy.testing` is imported inside a
Streamlit render thread (a fork-safety issue that does not occur on Windows/Linux):

```bash
python scripts/run_ui.py            # http://localhost:8501
python scripts/run_ui.py --port 8600
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

### Stimulus and trigger wiring

For event-related recordings, the stimulus computer should provide a trigger signal
that is recorded together with the EEG. With an ActiCHamp, connect the trigger output
to one AUX input and configure that input as `TriggerChannel`. Keep the trigger timing
and sampling rate identical to the EEG recording.

- **Visual VEP and visual oddball**: connect the screen or stimulus presentation
  trigger output to an ActiCHamp AUX input. CortiPy uses the recorded trigger to align
  and average the visual responses.
- **Acoustic oddball**: use the same trigger principle with an acoustic stimulus and
  record its timing on an AUX input.
- **ABR/BERA**: use an acoustic trigger. The current ABR acquisition path requires an
  ActiCHamp; verify the connected AUX input and trigger timing before recording.
- **Alpha, eyes open/closed**: a trigger is optional when the analysis uses an
  internally defined timing sequence. If the eyes-open/eyes-closed changes are marked
  externally, record those events as triggers. The same trigger wiring used for VEP can
  be used.

Set the relevant trigger configuration in `Params["Parameters"]`, for example
`TriggerChannel`, `Trigger`, and `TriggerTime`. The exact AUX channel number depends on
the ActiCHamp wiring and must match the recorded data columns.

### BIDS import/export

- `cortipy.shared.BIDSLoader` can read/write BIDS datasets. Supported inputs include EDF/BDF, BrainVision (`.vhdr/.eeg`), EEGLAB (`.set`), FIF, Parquet, HDF5, and Zarr.
- `cortipy.shared.ExperimentBinLoader` reads/writes the `.bin` + `params.json` pairs used in `experiments/datasets/D2_software_curated_signals` and `experiments/datasets/D4_sereega_evoked_potentials`, returning a `BIDSLoadResult` so you can analyse or re-export them.
- Extra dependencies for non-default formats: `pyedflib` (EDF/BDF export), `pyarrow` or `fastparquet` (Parquet), `h5py` (HDF5), `zarr` (Zarr), `pybv` (BrainVision export), and `eeglabio` (EEGLAB export).
- Install them via `pip install -e .[bids]` (or combine with `[ui]`) to enable all BIDS I/O features.

```python
from cortipy.shared import BIDSLoader, ExperimentBinLoader

bin_loader = ExperimentBinLoader("experiments/datasets/D4_sereega_evoked_potentials")
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
