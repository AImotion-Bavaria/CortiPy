# CortiPy

[![CI](https://github.com/AImotion-Bavaria/CortiPy/actions/workflows/ci.yml/badge.svg)](https://github.com/AImotion-Bavaria/CortiPy/actions/workflows/ci.yml)

CortiPy is a Python toolkit for EEG acquisition and analysis. It is a ground-up Python
port of a MATLAB-based EEG Analysis Tool and bundles the measurement pipeline, device
adapters, acquisition modules, evaluators, import/export helpers and a Streamlit UI in
one installable package.

```
device or imported data  ->  pipeline  ->  modules  ->  evaluation  ->  UI or exports
```

## Contents

- [Clinical use and privacy](#clinical-use-and-privacy)
- [Installation](#installation)
- [Hardware compatibility](#hardware-compatibility)
- [Quick start](#quick-start)
- [Streamlit UI](#streamlit-ui)
- [Device specifics](#device-specifics)
- [Stimulus and trigger wiring](#stimulus-and-trigger-wiring)
- [BIDS import/export](#bids-importexport)
- [Format benchmark experiments](#format-benchmark-experiments)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [Documentation and support](#documentation-and-support)
- [Roadmap](#roadmap)
- [License and third-party components](#license-and-third-party-components)

## Clinical use and privacy

- CortiPy is for research and educational use only. It is not a medical device and must not be used for diagnosis or patient care.
- Follow institutional approvals and local regulations before any clinical evaluation. Validate device interoperability and trigger/latency behavior in your lab.
- Do not store or share identifiable participant data in repositories or issue trackers. Anonymize runs, keep participant data on secured systems, and use synthetic or anonymized data for examples.

## Installation

CortiPy needs Python 3.9 to 3.12.

```bash
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[ui,bids,sbids]"   # full app plus conversion scripts
```

The extras can also be installed separately:

| Extra | Adds |
| --- | --- |
| *(none)* | Core toolkit: pipeline, devices, modules, evaluation. |
| `ui` | Streamlit UI. |
| `bids` | BIDS import/export formats (EDF/BDF, Parquet, HDF5, Zarr, BrainVision, EEGLAB). |
| `sbids` | SBIDS conversion and validation. |
| `stim` | Stimulus presentation with pygame. |
| `dev` | Ruff for linting. |

For conda users, `environment.yml` pins the runtime dependencies and installs `cortipy`
from the local checkout:

```bash
conda env create -f environment.yml
conda activate cortipy
```

| Platform | pip | conda | Notes |
| --- | --- | --- | --- |
| Windows 10/11 | ✅ | ✅ | All devices, including ActiCHamp. |
| macOS (Apple/Intel) | ✅ | ✅ | ActiCHamp is not supported. Start the UI with `python scripts/run_ui.py` on Apple Silicon. |
| Linux (Ubuntu 22.04+) | ✅ | ✅ | ActiCHamp is not supported. |

## Hardware compatibility

| Device | Windows | macOS | Linux | Notes |
| --- | --- | --- | --- | --- |
| UNICORN | ✅ | ✅ | ✅ | Uses virtual COM/Bluetooth serial; verify the port name (e.g. `COM7`, `/dev/tty.Unicorn-DevB`). |
| ActiCHamp | ✅ | ❌ | ❌ | Windows only. Uses runtime components of the Brain Products Amplifier SDK, see [License and third-party components](#license-and-third-party-components). |
| LSL streams | ✅ | ✅ | ✅ | Any LabStreamingLayer EEG source. |
| Dummy/Sim | ✅ | ✅ | ✅ | Synthetic EEG for the UI and tests. No hardware needed. |
| Offline replay | ✅ | ✅ | ✅ | Replays `.npz`/`.mat`/BIDS data. |

## Quick start

This example runs an alpha measurement on the built-in dummy device, so it works
without an amplifier:

```python
from cortipy import MeasurementPipeline
from cortipy.core.pipeline import PipelineHooks

params = {
    "Method": "Alpha",
    "Device": "Dummy",
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

print(params["data"].shape)        # (samples, channels)
print(list(params["Evaluation"]))  # results of the alpha evaluator
```

Replace `"Dummy"` with `"UNICORN"`, `"ActiCHamp"`, `"LSL"` or `"Offline"` and add the
device parameters described in [Device specifics](#device-specifics) to record from
real hardware or replay a file.

The parameter keys are MATLAB-style PascalCase (`Method`, `Device`, `TriggerChannel`)
so that configurations stay compatible with the original MATLAB `Params` structure.
EEG data is stored as `(samples, channels)` in microvolts.

### Config files and runners

- `cortipy.ui.run_from_config` loads a JSON/TOML config and executes the pipeline once.
- `cortipy.ui.SessionOptions` and `SaveManager` write timestamped run folders that
  contain `params.json` and `data.npz`.

## Streamlit UI

```bash
streamlit run apps/streamlit_app.py
```

On **macOS (Apple Silicon)** use the launcher instead. Plain `streamlit run` can crash
on first use because numpy 1.26 runs a subprocess when `numpy.testing` is imported
inside a Streamlit render thread:

```bash
python scripts/run_ui.py            # http://localhost:8501
python scripts/run_ui.py --port 8600
```

Open `http://localhost:8501` and configure a run:

- **Session configuration:** pick a method and device, fill in the participant metadata and assign the electrodes. For an ActiCHamp, the electrode impedances are shown here.
- **Simulate run:** turn this on to exercise the UI with synthetic data. No hardware is opened.
- **Live preview:** shows the incoming signal before the recording starts.
- **Charts:** shows the results of the finished run.
- **Saved sessions:** reload the parameters of earlier runs and browse their results.

Runs are saved to `cortipy_runs` in the folder the app was started from, each in its own
timestamped folder. App logs are written to `streamlit_app.log` in the repository root;
keep it handy when debugging device connections.

## Device specifics

- **UNICORN**: Provide the virtual COM port (for example `"COM7"` on Windows or
  `"/dev/tty.Unicorn-DevB"` on macOS) via `Params["Parameters"]["UNICORNPort"]`
  (aliases `UNICORNAddress` / `UnicornPort` are accepted). Optional extras:
  `UNICORNDeviceName` for logging and `UnicornTimeout` (seconds). The adapter returns
  16-column packets `[EEG(8), accel(3), gyro(3), battery, counter]`.
- **ActiCHamp**: Windows only. CortiPy starts a small producer program from
  `cortipy/devices/actichamp` that talks to the amplifier and passes the data on through
  shared memory. BrainVision Recorder must be closed. Supply `fs`, `NumberEEGChannels`
  and optionally `NumberAUXChannels`; the AUX columns follow the EEG columns. Set
  `Params["ActiChampPath"]` to use a producer in a different folder. See
  [cortipy/devices/actichamp/README.md](cortipy/devices/actichamp/README.md) for details
  and for how to rebuild the producer.
- **LSL / Dummy / Offline**: Use `Device` = `"LSL"` for any LabStreamingLayer EEG
  stream, `"Dummy"` (or `"Sim"`) for synthetic data, or `"Offline"` to replay
  `.npz`/`.mat` files.

## Stimulus and trigger wiring

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

## BIDS import/export

- `cortipy.shared.BIDSLoader` reads and writes BIDS datasets. Supported inputs include EDF/BDF, BrainVision (`.vhdr/.eeg`), EEGLAB (`.set`), FIF, Parquet, HDF5, and Zarr.
- `cortipy.shared.ExperimentBinLoader` reads and writes the `.bin` + `params.json` pairs used in `experiments/datasets/D2_software_curated_signals` and `experiments/datasets/D4_sereega_evoked_potentials`, returning a `BIDSLoadResult` so you can analyse or re-export them.
- The non-default formats need the `bids` extra: `pip install -e ".[bids]"`.
- `scripts/` contains ready-made converters between the bin, BIDS and SBIDS layouts, for example `bin_to_bids.py` and `bids_to_sbids.py`.

```python
from cortipy.shared import BIDSLoader, ExperimentBinLoader

bin_loader = ExperimentBinLoader("experiments/datasets/D4_sereega_evoked_potentials")
result = bin_loader.read_bin("Oddball")  # loads params.json + Oddball_scalpdata.bin

# Convert to BIDS
BIDSLoader("exports/oddball_bids").to_bids(result.raw, subject="01", task="oddball", overwrite=True)

# Convert any Raw/numpy data back into the bin layout
bin_loader.write_bin(result.raw, "exports/oddball_bin", params=result.metadata["params"], overwrite=True)
```

## Format benchmark experiments

`experiments/run_all_experiments.py` runs all three experiments and generates the paper
outputs. See [experiments/README.md](experiments/README.md) for details.

- **Experiment 1 – Streaming & numerical validation** (`experiments/experiment1/run_experiment1.py`): `PYTHONPATH=. python ...` with no flags. Checks bitwise identity of BIN exports, runs a synthetic sine pipeline surrogate, and renders CortiPy plots for ABR/ASSR/Oddball/VEP/SSVEP/Sine10Hz to compare against EEGLAB references. Outputs live under `experiments/experiment1/results/` (JSON + PNG/PDFs). Hardware/EEGLAB numeric correlations are documented as manual placeholders.
- **Experiment 2 – File format storage & latency** (`experiments/experiment2/run_experiment2.py`): measures export size and read/write latency across BIDS/SBIDS containers and formats (parquet/edf/zarr/hdf5) using synthetic 1 h datasets (19/64/256 ch). Default: run all datasets, containers, and formats; results saved to `experiments/experiment2/results/experiment2_results.json` with plots. Key flags: `--datasets` labels, `--containers` bids|sbids, `--formats` parquet|edf|zarr|hdf5, `--runs` N, `--keep-artifacts`, `--no-purge`, `--plot-scope all|synthetic|bin`, `--plots-only`, `--results-json PATH`.
- **Experiment 3 – Streaming access benchmark** (`experiments/experiment3/run_experiment3.py`): times streaming-style reads (open handle, single channel, random windows) for the same synthetic datasets exported to BIDS/SBIDS formats. Default: all datasets, containers, formats; results to `experiments/experiment3/results/experiment3_results.json` with plots. Key flags: `--datasets`, `--containers`, `--formats`, `--runs`, `--window-s` (window duration), `--channel-idx` (0-based), `--keep-artifacts`, `--no-purge`, `--plots-only`, `--results-json PATH`.

## Repository layout

```
cortipy/
├── core/          # MeasurementPipeline, ModuleContext, params helpers
├── devices/       # UNICORN, ActiCHamp (Windows), LSL, offline and dummy adapters
├── modules/       # Alpha, VEP, SSVEP, BERA, ASSR, P300, BCI acquisition loops
├── evaluation/    # Module-specific analysis and plotting code
├── shared/        # Filters, FFT, channels, units, BIDS/SBIDS and report helpers
├── ui/            # Config loaders, session runner and save helpers (no GUI)
└── ui_streamlit/  # The Streamlit application
apps/              # Streamlit entry point (streamlit_app.py)
scripts/           # bin/BIDS/SBIDS converters, run_ui.py
experiments/       # Benchmark suite and datasets
tests/             # Unit, regression, experiment and UI tests
```

`CortiPy_VEP_Ear_Study` is a separate Git repository included as a submodule. Run
`git submodule update --init --recursive` after cloning to fetch it.

## Testing

```bash
pip install pytest
pytest                                  # full suite
pytest tests/path/to/test_file.py       # one file

pip install -e ".[dev]"
ruff check .
```

No EEG hardware is needed; the tests run on the dummy device. The MATLAB parity tests
in `tests/regression/test_evaluators.py` are skipped unless `CORTIPY_RUN_REGRESSION=1`
is set and the MATLAB-generated fixtures are present.

CI runs Ruff, pytest on Python 3.9 to 3.12 and a package build for pull requests and
for pushes to `main`.

## Documentation and support

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) describes the architecture and the domain background.
- Release notes are maintained in [CHANGELOG.md](CHANGELOG.md).
- See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening a pull request.
- Report bugs through the issue templates. Report vulnerabilities as described in [SECURITY.md](SECURITY.md), not in a public issue. Response target: within 5 business days.
- Do not include identifiable participant data in issues or pull requests.
- To cite CortiPy, use [CITATION.cff](CITATION.cff).

## Roadmap

- CI/CD hardening: type and coverage gates, release automation to PyPI.
- Streamlit UX: latency display, clearer device status, structured logging.
- Docs: a documentation site with quickstart, device setup, BIDS/SBIDS workflows and troubleshooting.
- Data governance: anonymization guidance, SBOM/dependency provenance for clinical reviews.
- Packaging: signed wheels/sdists.

## License and third-party components

The source code of CortiPy is licensed under the Apache License 2.0, see [LICENSE](LICENSE).

The ActiCHamp device adapter uses the **BrainVision Amplifier SDK** of Brain Products
GmbH. The following files in `cortipy/devices/actichamp` are **not** covered by the
Apache License:

- `AmplifierSDK.dll`, `ActiChamp_x64.dll`, `actiCHampLib.dll`, `ActiChamp.bit` and
  `ActiChampX.bit` are unmodified compiled components of the SDK, version 1.1.0.2,
  Copyright (C) Brain Products GmbH.
- `EEG_SharedMemoryProducer.exe` is built from CortiPy source code with the headers,
  the import library and a sample file of the same SDK.

These files are subject to the BrainVision Amplifier SDK Licensing Terms of Brain
Products GmbH. In particular:

- They may be used for **non-commercial research purposes only**. Selling, renting, lending or leasing them, or software built on them, requires a written agreement with Brain Products.
- They must **not be used for high-risk or medically classified purposes** such as diagnosis or treatment.
- They are provided as is. All warranties and all liability for damages are excluded, as far as the law permits.
- The compiled components must not be modified, decompiled or reverse engineered.

The SDK itself (headers, import library, documentation and samples) is not part of this
repository. It is only needed to rebuild the producer and must be downloaded from Brain
Products, see [cortipy/devices/actichamp/README.md](cortipy/devices/actichamp/README.md).
By using the ActiCHamp device in CortiPy you agree to comply with the Brain Products
licensing terms. All other devices work without these files.

CortiPy is an independent project. It is not affiliated with, endorsed by, or supported
by Brain Products GmbH. "actiCHamp" and "Brain Products" are trademarks of their
respective owners.
