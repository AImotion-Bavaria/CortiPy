# CortiPy Project Context

## Purpose
CortiPy is a Python EEG acquisition and analysis toolkit — a ground-up Python port of a MATLAB-based EEG Analysis Tool. It provides a measurement pipeline, device adapters, processing modules, evaluators, shared signal utilities, and a Streamlit UI.

For commands and development rules, see `CLAUDE.md` (auto-loaded by Claude Code) or `AGENTS.md` (for other coding-agent tools).

## Main flow
Device or imported data -> pipeline -> modules -> evaluation -> UI or exports

## Key abstractions
- `ModuleContext` (`cortipy/core/context.py`): a mutable dataclass — `params` dict, `device`, `data_buffer`, `services` — created once per run and threaded through every module.
- `MeasurementPipeline` (`cortipy/core/pipeline.py`): runs two passes over the configured modules — `connect()` on all modules first (device setup), then `collect()` on all modules (acquire + evaluate).
- `ModuleBase` (`cortipy/modules/base.py`): implements that two-phase lifecycle. Concrete modules (`alpha.py`, `vep.py`, `ssvep.py`, …) only override `collect_measurements()`.
- `EvaluatorBase` (`cortipy/evaluation/base.py`): attached to a module, invoked automatically from `ModuleBase.post_collect()`, mutates `context.params['Evaluation']`.

## Data conventions
- **Array shape**: `params['data']`, `ModuleContext.append_data`, and `CortiDataset.data` are all `(samples, channels)`. MNE `Raw` objects use the opposite convention, `(channels, samples)` — the conversion (`.T`) happens only at the BIDS/MNE boundary, in `cortipy/shared/bids.py` (`_coerce_to_raw_array`, `raw_to_microvolts`).
- **Units**: CortiPy carries EEG in microvolts everywhere internally; MNE requires SI units (volts). Convert only at the MNE boundary with `to_volts()`/`to_microvolts()` (`cortipy/shared/units.py`) — never scale non-voltage channel types (stim/misc, see `VOLT_CHANNEL_TYPES`). `looks_like_microvolts()` guards against legacy exports that wrote µV magnitudes into a volt-labeled container.
- **Channels**: `params['Channels']` only describes the EEG columns of `params['data']` — AUX/trigger columns trail the EEG block and are absent from the list. Use `cortipy.shared.channels` helpers (`channel_labels()`, `resolve_channel_index()`, `resolve_plot_channel()`) to resolve labels/indices rather than re-deriving them; evaluators used to each carry their own (disagreeing) lookup logic.

## Important locations
- `pyproject.toml`: package metadata, dependencies, and Ruff config — the single source of truth. `requirements.txt` is just a one-line `-e .[ui,bids,sbids]` pointer at its extras, with nothing of its own to keep in sync; `environment.yml` mirrors the same extras but also hand-duplicates pinned versions (numpy==1.26.4, matplotlib==3.9.2, pyserial==3.5, …), and is currently missing the `streamlit-plotly-events` pin present in pyproject.toml's `ui` extra — don't assume it's fully in sync.
- `apps/streamlit_app.py`: Streamlit entry point (imports `cortipy.ui_streamlit.session` as `ui`)
- `cortipy/core/`: pipeline, context, and parameter handling
- `cortipy/devices/`: hardware, LSL, dummy, and offline adapters — `dummy.py`'s `DummyDevice` (reached via `Device: "dummy"`/`"sim"`/`"simulation"`) generates synthetic EEG-like data and needs no physical amplifier, useful for running the pipeline/UI without hardware attached
- `cortipy/modules/`: acquisition and processing modules
- `cortipy/evaluation/`: evaluation algorithms
- `cortipy/shared/`: shared signal, format, and reference utilities
- `cortipy/ui/`: headless/programmatic config and session-running API (`config.py`, `runner.py`, `save.py`) — smaller and more standalone than its name suggests; only `SaveManager`/`normalize_params` are wired into the Streamlit UI
- `cortipy/ui_streamlit/`: the Streamlit application itself (session state, sidebar, live plotting, workflow orchestration, device/import config) — not just "chart helpers"; `session.py` is the single largest file in the package
- `tests/`: unit, regression, experiment, and UI tests
- `experiments/`: benchmark suite (experiment1/2/3, datasets/, paper_outputs/, run_all_experiments.py)
- `scripts/`: bin/BIDS/SBIDS conversion scripts, `run_ui.py` (macOS launcher), `bump_version.py` (version bump — currently broken, see `CLAUDE.md`/`AGENTS.md`)
- `docs/`: misc project docs
- `cortipy_runs/` (top-level, gitignored) vs. `apps/cortipy_runs/` (**not** gitignored): default save directory for recorded sessions is `Path.cwd() / "cortipy_runs"`, so its location and git-ignore status depend on the working directory the app was launched from
- `streamlit_app.log`: written by two independent handlers pointed at the same path. `cortipy/evaluation/__init__.py` attaches a `FileHandler` (DEBUG) to the narrow `"cortipy.evaluation"` logger only. `cortipy/ui_streamlit/session.py` separately attaches its own `FileHandler` (INFO) to the **root** logger (`logging.getLogger()`, no name filter) — since it's on root, this is the handler that actually captures device-connection log lines (e.g. `cortipy/devices/actichamp_device.py` logging via `logging.getLogger(__name__)`, messages like "ActiChamp device requires Windows...", "Disconnecting ActiChamp device.", which propagate up to root). Useful for debugging device connections, but only when the Streamlit app (not a bare pipeline run) is what wrote the log.
- `results.json`: written to the process's cwd by `evaluation/alpha.py`'s `plot_alpha_matrix()` as an undocumented debug side effect whenever alpha evaluation renders a plot; gitignored so it won't get committed, but can leave stray files in unexpected directories

## Submodule
- Path: `CortiPy_VEP_Ear_Study`
- Repository: https://github.com/LKreilinger/CortiPy_VEP_Ear_Study.git
- The main repository stores a Git submodule pointer to one exact commit; the submodule's files are maintained in the separate repository.
- Inspect the pinned commit with `git submodule status`; see `AGENTS.md`/`CLAUDE.md` for setup and handling rules.
