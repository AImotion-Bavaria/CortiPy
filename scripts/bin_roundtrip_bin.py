"""Load CortiPy `.bin` datasets, optionally run analysis, and re-save to a new `.bin` folder.

Intended to sanity-check the load/analysis/save loop and surface any bit-loss by
round-tripping through memory before writing out a fresh params.json + .bin.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

import hashlib
import mne
import numpy as np
import pandas as pd


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
BIDSLoadResult = getattr(shared, "BIDSLoadResult")
ExperimentBinLoader = getattr(shared, "ExperimentBinLoader")

_ModuleContext = None
_EVAL_CLASSES: dict[str, type] = {}
METHOD_ALIASES = {
    "odball": "p300",
    "oddball": "p300",
    "abr": "bera",
}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/synData"),
        help="Folder containing dataset subdirectories (each with params.json + *.bin)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Where to write the new bin exports (defaults to <root>_binrt next to root)",
    )
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
    parser.add_argument("--verify", action="store_true", help="Reload written bin and report max abs diff to detect drift")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--summary-csv", type=Path, help="Write per-dataset summary table to CSV")
    parser.add_argument("--summary-json", type=Path, help="Write raw summary payload to JSON")
    parser.add_argument("--summary-plot", type=Path, help="Save a per-dataset plot of roundtrip quality")
    parser.add_argument("--summary-title", help="Optional title for printed/plot summaries")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.debug_eval:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    root = args.root.expanduser().resolve()
    out_root = args.output_root.expanduser() if args.output_root else root.parent / f"{root.name}_binrt"
    datasets = _discover_datasets(root)
    print(f"Source root: {root}")
    print(f"Output root: {out_root}")
    if not datasets:
        print(f"No datasets found under {root}", file=sys.stderr)
        return 1
    print(f"Discovered {len(datasets)} dataset(s).")

    loader = ExperimentBinLoader(root)
    summary: list[dict] = []
    for ds in datasets:
        try:
            result = loader.read_bin(ds)
            params = result.metadata.get("params", {}) if result.metadata else {}
            source_hash = _sha256(result.source_path)

            if args.synthesize_triggers:
                result, params = _inject_triggers(
                    result,
                    params,
                    interval_s=args.trigger_interval_s,
                    pulse_ms=args.trigger_pulse_ms,
                )

            analysis_summary = None
            if args.analyze:
                analysis_summary = _run_analysis(
                    params=params,
                    data=result.data,
                    method_override=args.method,
                    dataset_name=ds.name,
                    show_plots=args.show_plots,
                )
                if args.verbose:
                    summary_str = f"{analysis_summary.get('method')} {analysis_summary.get('status')}"
                    eval_payload = analysis_summary.get("evaluation")
                    if isinstance(eval_payload, dict):
                        summary_str += f" keys={list(eval_payload.keys())}"
                    if analysis_summary.get("error"):
                        summary_str += f" error={analysis_summary.get('error')}"
                    if analysis_summary.get("reason"):
                        summary_str += f" reason={analysis_summary.get('reason')}"
                    print(f"{ds.name}: analysis -> {summary_str}")

            target_dir = out_root / ds.name
            data_path, params_path = loader.write_bin(
                result.raw,
                target_dir,
                params=params,
                overwrite=args.overwrite,
            )
            output_hash = _sha256(data_path)

            verify = ""
            if args.verify:
                reloaded = loader.read_bin(target_dir)
                min_len = min(result.data.shape[0], reloaded.data.shape[0])
                min_ch = min(result.data.shape[1], reloaded.data.shape[1])
                diff = np.max(np.abs(result.data[:min_len, :min_ch] - reloaded.data[:min_len, :min_ch]))
                reload_hash = _sha256(reloaded.source_path)
                verify = (
                    f" (max_abs_diff={diff:.3g}, "
                    f"sha_src={source_hash[:8]}, sha_out={output_hash[:8]}, sha_reload={reload_hash[:8]}, "
                    f"src==out={source_hash==output_hash})"
                )
            else:
                verify = f" (sha_src={source_hash[:8]}, sha_out={output_hash[:8]}, src==out={source_hash==output_hash})"

            print(f"{ds.name}: saved {data_path} and {params_path}{verify}")
            summary.append(
                {
                    "name": ds.name,
                    "src_hash": source_hash,
                    "out_hash": output_hash,
                    "reload_hash": reload_hash if args.verify else output_hash,
                    "src_eq_out": source_hash == output_hash,
                    "out_eq_reload": output_hash == (reload_hash if args.verify else output_hash),
                    "max_abs_diff": diff if args.verify else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{ds.name}: failed - {exc}", file=sys.stderr)
            summary.append({"name": ds.name, "error": str(exc)})
    frame = _summary_dataframe(summary)
    _print_summary(summary, frame=frame, title=args.summary_title)
    _export_summary(summary, frame, csv_path=args.summary_csv, json_path=args.summary_json)
    _plot_roundtrip_summary(
        frame,
        save_path=args.summary_plot,
        show=args.show_plots,
        title=args.summary_title,
    )
    return 0


def _discover_datasets(root: Path) -> Sequence[Path]:
    candidates: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "params.json").exists() or list(entry.glob("*.bin")):
            candidates.append(entry)
    return candidates


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
        events=getattr(result, "events", None),
        channels=getattr(result, "channels", None),
        metadata=getattr(result, "metadata", {}),
        source_path=result.source_path,
        ancillary_files=getattr(result, "ancillary_files", []),
    )
    return updated, params


def _summary_dataframe(summary: list[dict]) -> pd.DataFrame:
    """Convert run summary payloads to a tidy DataFrame for reporting/export."""
    records: list[dict] = []
    for item in summary:
        records.append(
            {
                "Dataset": item.get("name"),
                "Status": "error" if "error" in item else "ok",
                "Source==Out": item.get("src_eq_out"),
                "Out==Reload": item.get("out_eq_reload"),
                "MaxAbsDiff": item.get("max_abs_diff"),
                "SourceHash": item.get("src_hash"),
                "OutputHash": item.get("out_hash"),
                "ReloadHash": item.get("reload_hash"),
                "Error": item.get("error"),
            }
        )
    return pd.DataFrame.from_records(records)


def _print_summary(summary: list[dict], frame: pd.DataFrame | None = None, *, title: str | None = None) -> None:
    if not summary:
        return
    frame = frame if frame is not None else _summary_dataframe(summary)
    if frame.empty:
        print("No summary data to display.")
        return

    ok_mask = frame["Status"] == "ok"
    hash_match_mask = frame["Out==Reload"].fillna(False)
    ok_count = int((ok_mask & hash_match_mask).sum())
    error_count = int((frame["Status"] == "error").sum())
    total = len(frame)
    header = title or "Roundtrip summary"
    print(f"{header}: {ok_count}/{total} outputs match their reload hash; {error_count} failed.")

    display = frame.copy()
    display["Source==Out"] = display["Source==Out"].apply(_format_bool)
    display["Out==Reload"] = display["Out==Reload"].apply(_format_bool)
    display["MaxAbsDiff"] = display["MaxAbsDiff"].apply(_format_float)
    display["Status"] = display["Status"].str.upper()

    cols = ["Dataset", "Status", "Source==Out", "Out==Reload", "MaxAbsDiff", "Error"]
    print(display[cols].to_string(index=False))


def _format_bool(value) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _format_float(value) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.3g}"
    except Exception:
        return "n/a"


def _export_summary(
    raw_summary: list[dict],
    frame: pd.DataFrame,
    *,
    csv_path: Path | None,
    json_path: Path | None,
) -> None:
    if csv_path:
        csv_path = csv_path.expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)
        print(f"Wrote summary CSV -> {csv_path}")
    if json_path:
        json_path = json_path.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw_summary, f, indent=2)
        print(f"Wrote summary JSON -> {json_path}")


def _plot_roundtrip_summary(
    frame: pd.DataFrame,
    *,
    save_path: Path | None,
    show: bool,
    title: str | None,
) -> None:
    if frame.empty:
        return

    plot_frame = frame[frame["Status"] != "error"].copy()
    if plot_frame.empty:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting optional
        print(f"Plotting skipped: {exc}", file=sys.stderr)
        return

    values = plot_frame["MaxAbsDiff"].fillna(0.0).astype(float)
    x = np.arange(len(values))
    colors = ["#2e7d32" if ok else "#c62828" for ok in plot_frame["Out==Reload"].fillna(False)]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.75 * len(values)), 4.5))
    bars = ax.bar(x, values, color=colors, edgecolor="#424242")

    plot_title = title or "Roundtrip verification"
    ax.set_title(plot_title)
    ax.set_ylabel("Max abs diff (source vs reload)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_frame["Dataset"], rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    max_height = max(values.max(), 0.0)
    pad = max_height * 0.05 if max_height else 0.05
    for bar, ok in zip(bars, plot_frame["Out==Reload"].fillna(False)):
        label = "match" if ok else "mismatch"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + pad,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#424242",
        )
    ax.set_ylim(0, max_height + 2 * pad)

    fig.tight_layout()
    if save_path:
        save_path = save_path.expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved summary plot -> {save_path}")

    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
