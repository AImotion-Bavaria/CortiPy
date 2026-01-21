"""Convert CortiPy experiment `.bin` datasets into per-dataset BIDS folders with EDF files.

Defaults to the D4 synthetic datasets under `experiments/datasets/D4_sereega_evoked_potentials`, writing siblings like
`Oddball_bids` with an EDF export inside the `eeg/` modality directory.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

import logging
import numpy as np
import pandas as pd
import mne


def _load_shared_module() -> ModuleType:
    """Load cortipy.shared.* without requiring an installed package."""
    try:
        import cortipy.shared as shared  # type: ignore
        return shared  # pragma: no cover
    except Exception:
        repo_root = Path(__file__).resolve().parent.parent
        shared_path = repo_root / "cortipy" / "shared" / "__init__.py"
        spec = importlib.util.spec_from_file_location("cortipy.shared", shared_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return module


shared = _load_shared_module()
BIDSLoader = getattr(shared, "BIDSLoader")
ExperimentBinLoader = getattr(shared, "ExperimentBinLoader")
BIDSLoadResult = getattr(shared, "BIDSLoadResult")
_ModuleContext = None
_EVAL_CLASSES: dict[str, type] = {}
METHOD_ALIASES = {
    "odball": "p300",
    "oddball": "p300",
    "abr": "bera",
}


def _safe_label(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "", value)
    return token or "01"


def _discover_datasets(root: Path) -> Sequence[Path]:
    candidates: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "params.json").exists() or list(entry.glob("*.bin")):
            candidates.append(entry)
    return candidates


def convert_dataset(
    loader: ExperimentBinLoader,
    dataset_dir: Path,
    *,
    format: str,
    modality: str,
    subject: str | None,
    task: str | None,
    analyze: bool,
    method_override: str | None,
    show_plots: bool,
    verbose: bool,
    synth_triggers: bool,
    synth_interval: float | None,
    synth_pulse_ms: float,
    overwrite: bool,
) -> Path:
    name = dataset_dir.name
    result = loader.read_bin(name)
    params = result.metadata.get("params", {}) if result.metadata else {}
    channels = _channels_for_bids(result.raw.ch_names, params)

    subject_label = subject or _safe_label(name)
    task_label = task or name.lower()

    output_root = dataset_dir.parent / f"{name}_bids"
    target_loader = BIDSLoader(output_root)

    dataset_description = {
        "Name": f"{name} synthetic export",
        "BIDSVersion": "1.8.0",
        "GeneratedBy": [{"Name": "cortipy bin_to_bids_edf.py"}],
    }

    analysis_summary = None
    if synth_triggers:
        result, params = _inject_triggers(
            result,
            params,
            interval_s=synth_interval,
            pulse_ms=synth_pulse_ms,
        )

    if analyze:
        analysis_summary = _run_analysis(
            params=params,
            data=result.data,
            method_override=method_override,
            dataset_name=name,
            output_root=output_root,
            show_plots=show_plots,
        )
        if verbose:
            print(f"{name}: analysis -> {analysis_summary}")

    out_path = target_loader.to_bids(
        result.raw,
        subject=subject_label,
        task=task_label,
        modality=modality,
        format=format,
        overwrite=overwrite,
        sidecar={"SamplingFrequency": result.sampling_rate},
        dataset_description=dataset_description,
        channels=channels,
    )
    if analyze and analysis_summary is not None:
        summary_file = output_root / f"{name}_analysis.json"
        summary_file.write_text(json.dumps(analysis_summary, indent=2))
    return out_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/datasets/D4_sereega_evoked_potentials"),
        help="Folder containing dataset subdirectories (each with params.json + *.bin)",
    )
    parser.add_argument("--format", default="edf", choices=["edf", "bdf", "fif"], help="BIDS data format to export")
    parser.add_argument("--modality", default="eeg", help="Modality directory name for the BIDS export")
    parser.add_argument("--subject", help="Subject label to use (defaults to dataset folder name)")
    parser.add_argument("--task", help="Task label to use (defaults to dataset folder name)")
    parser.add_argument("--method", help="Override Method used for analysis (defaults to params['Method'] or folder name)")
    parser.add_argument("--analyze", action="store_true", help="Run method-specific analysis before export")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows after analysis")
    parser.add_argument("--verbose", action="store_true", help="Print analysis summaries to stdout")
    parser.add_argument("--debug-eval", action="store_true", help="Enable INFO logging for evaluators")
    parser.add_argument(
        "--synthesize-triggers",
        action="store_true",
        help="Append a synthetic trigger channel (pulses) before analysis/export",
    )
    parser.add_argument(
        "--trigger-interval-s",
        type=float,
        help="Seconds between synthetic trigger pulses (defaults to RecordingTime/Epochs or 1s)",
    )
    parser.add_argument(
        "--trigger-pulse-ms",
        type=float,
        default=5.0,
        help="Pulse width (ms) for synthetic triggers",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.debug_eval:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    root = args.root.expanduser()
    datasets = _discover_datasets(root)
    if not datasets:
        print(f"No datasets found under {root}", file=sys.stderr)
        return 1

    loader = ExperimentBinLoader(root)
    for ds in datasets:
        try:
            out_path = convert_dataset(
                loader,
                ds,
                format=args.format,
                modality=args.modality,
                subject=args.subject,
                task=args.task,
                analyze=args.analyze,
                method_override=args.method,
                show_plots=args.show_plots,
                verbose=args.verbose,
                synth_triggers=args.synthesize_triggers,
                synth_interval=args.trigger_interval_s,
                synth_pulse_ms=args.trigger_pulse_ms,
                overwrite=args.overwrite,
            )
            print(f"{ds.name}: exported to {out_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"{ds.name}: failed - {exc}", file=sys.stderr)
    return 0


def _channels_for_bids(ch_names: Sequence[str], params: dict) -> pd.DataFrame | None:
    """Build a BIDS-compatible channels.tsv frame from params.json contents."""
    entries = params.get("Channels") or []
    rows: list[dict] = []
    for entry in entries:
        name = entry.get("Channel") or entry.get("Position") or entry.get("Name")
        desc_parts = []
        for key in ("Rubrik", "Model", "Position"):
            val = entry.get(key)
            if val and str(val) != str(name):
                desc_parts.append(str(val))

        rows.append(
            {
                "name": str(name) if name else None,
                "type": str(entry.get("Type", "eeg")).lower() if entry.get("Type", "eeg") else "eeg",
                "status": "bad" if entry.get("Active") is False else "good",
                "impedance": entry.get("Impedance"),
                "description": "; ".join(desc_parts) if desc_parts else None,
            }
        )

    if not rows:
        return None

    frame = pd.DataFrame(rows)
    # Align to the Raw channel order, filling missing with defaults.
    frame = frame.drop_duplicates(subset=["name"]).set_index("name", drop=False)
    ordered_rows = []
    for ch in ch_names:
        if ch in frame.index:
            ordered_rows.append(frame.loc[ch])
        else:
            ordered_rows.append(
                {
                    "name": ch,
                    "type": "eeg",
                    "status": "good",
                    "impedance": None,
                    "description": None,
                }
            )
    ordered = pd.DataFrame(ordered_rows)
    cols = [c for c in ("name", "type", "status", "impedance", "description") if c in ordered.columns]
    return ordered[cols]


def _load_evaluators() -> None:
    global _ModuleContext, _EVAL_CLASSES
    if _EVAL_CLASSES:
        return

    _ModuleContext = _load_module("cortipy.core.context")
    eval_modules = {
        "vep": _load_module("cortipy.evaluation.vep"),
        "assr": _load_module("cortipy.evaluation.assr"),
        "p300": _load_module("cortipy.evaluation.p300"),
        "ssvep": _load_module("cortipy.evaluation.ssvep"),
        "alpha": _load_module("cortipy.evaluation.alpha"),
        "bera": _load_module("cortipy.evaluation.bera"),
    }
    _EVAL_CLASSES = {
        "vep": eval_modules["vep"].VepEvaluator,
        "assr": eval_modules["assr"].AssrEvaluator,
        "p300": eval_modules["p300"].P300Evaluator,
        "ssvep": eval_modules["ssvep"].SsvepEvaluator,
        "alpha": eval_modules["alpha"].AlphaEvaluator,
        "bera": eval_modules["bera"].BeraEvaluator,
    }


def _load_module(modname: str) -> ModuleType:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module(modname)
    except Exception:
        rel = Path(*modname.split("."))  # type: ignore[arg-type]
        module_path = repo_root / f"{rel}.py"
        spec = importlib.util.spec_from_file_location(modname, module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return module


def _run_analysis(
    params: dict,
    data,
    *,
    method_override: str | None,
    dataset_name: str,
    output_root: Path,
    show_plots: bool,
) -> dict:
    _load_evaluators()
    method = (method_override or params.get("Method") or dataset_name).strip()
    method_key = method.lower()
    canonical_method = METHOD_ALIASES.get(method_key, method_key)
    evaluator_cls = _EVAL_CLASSES.get(canonical_method)
    if evaluator_cls is None:
        return {"method": method, "status": "skipped", "reason": "no evaluator available"}

    params_copy = _normalize_params_for_eval(copy.deepcopy(params), canonical_method, data)
    skip_reason = params_copy.pop("_skip_evaluator_reason", None)
    if skip_reason:
        return {"method": method, "status": "skipped", "reason": skip_reason}
    params_copy["Method"] = canonical_method.upper() if canonical_method.isalpha() else canonical_method
    params_copy["data"] = data

    ctx = _ModuleContext.ModuleContext(params_copy)  # type: ignore[attr-defined]
    try:
        evaluator = evaluator_cls(show_plots=show_plots)
    except TypeError:
        evaluator = evaluator_cls()

    try:
        evaluator.evaluate(ctx)  # type: ignore[arg-type]
        results = _json_safe(ctx.params.get("Evaluation"))
        if show_plots:
            _show_all_figures()
        return {"method": method, "status": "ok", "evaluation": results}
    except Exception as exc:  # noqa: BLE001
        return {"method": method, "status": "failed", "error": str(exc)}


def _json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)


def _show_all_figures() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        was_interactive = plt.isinteractive()
        plt.ioff()
        figs = plt.get_fignums()
        if figs:
            plt.show(block=True)
        if was_interactive:
            plt.ion()
    except Exception:
        pass


def _normalize_params_for_eval(params: dict, method: str, data) -> dict:
    param_block = params.setdefault("Parameters", {})
    # ReferenceChannel may be a string like "Ref"; fall back to 1 for evaluators.
    try:
        param_block["ReferenceChannel"] = int(param_block.get("ReferenceChannel", 1))
    except (TypeError, ValueError):
        param_block["ReferenceChannel"] = 1

    channel_count = None
    try:
        channel_count = int(getattr(data, "shape", [None, None])[1])
    except Exception:
        channel_count = None

    if channel_count is None:
        try:
            eeg = param_block.get("NumberEEGChannels")
            aux = param_block.get("NumberAUXChannels", 0)
            if eeg is not None:
                channel_count = int(eeg) + int(aux or 0)
        except (TypeError, ValueError):
            channel_count = None

    if channel_count is None and params.get("Channels"):
        channel_count = len(params["Channels"])

    trigger_value = param_block.get("TriggerChannel")
    if trigger_value is None:
        if channel_count:
            param_block["TriggerChannel"] = channel_count
    else:
        try:
            trigger_channel = int(trigger_value)
        except (TypeError, ValueError):
            trigger_channel = channel_count or 1
        trigger_channel = max(1, trigger_channel)
        if channel_count:
            trigger_channel = min(trigger_channel, channel_count)
        param_block["TriggerChannel"] = trigger_channel

    # BERA evaluator only supports ActiCHamp/BIOPAC/Simulated; if not, skip early.
    if method == "bera":
        dev = (params.get("Device") or "").lower()
        if dev not in {"actichamp", "biopac", "simulated", "biopack"}:
            params["_skip_evaluator_reason"] = "BERA supports only ActiCHamp/BIOPAC/Simulated devices"
    return params


def _inject_triggers(
    result,
    params: dict,
    *,
    interval_s: float | None,
    pulse_ms: float,
    channel_name: str = "TRIG",
) -> tuple:
    """Append a synthetic trigger channel to the data/raw and update params."""
    fs = float(params.get("Parameters", {}).get("fs") or result.sampling_rate)
    if fs <= 0:
        raise ValueError("Sampling frequency not available for trigger synthesis.")

    if interval_s is None:
        rec_time = params.get("Parameters", {}).get("RecordingTime")
        epochs = params.get("Parameters", {}).get("Epochs")
        if epochs and rec_time:
            try:
                interval_s = float(rec_time) / float(epochs)
            except (TypeError, ValueError, ZeroDivisionError):
                interval_s = 1.0
        else:
            interval_s = 1.0

    width = max(1, int(round((pulse_ms / 1000.0) * fs)))
    step = max(1, int(round(interval_s * fs)))

    samples = result.data.shape[0]
    trig = np.zeros(samples, dtype=float)
    for start in range(0, samples, step):
        trig[start : start + width] = 1.0

    data = np.column_stack([result.data, trig])
    ch_names = list(result.raw.ch_names) + [channel_name]
    ch_types = ["stim" if name == channel_name else "eeg" for name in ch_names]

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types, verbose=False)
    raw = mne.io.RawArray(data.T, info)

    params = copy.deepcopy(params)
    params.setdefault("Channels", []).append(
        {"Channel": channel_name, "Position": channel_name, "Type": "stim", "Active": True}
    )
    params.setdefault("Parameters", {})["TriggerChannel"] = len(ch_names)

    updated = BIDSLoadResult(
        raw=raw,
        data=data,
        sampling_rate=fs,
        events=result.events,
        channels=result.channels,
        metadata=result.metadata,
        source_path=result.source_path,
        ancillary_files=getattr(result, "ancillary_files", []),
    )
    return updated, params


if __name__ == "__main__":
    raise SystemExit(main())
