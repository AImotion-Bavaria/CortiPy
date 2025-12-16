"""Experiment 1 – Streaming & Numerical Validation.

Runs automated parts of Experiment 1:
  1.1 Bitwise identity check (I/O correctness)
  1.2 Pipeline equivalence surrogate (synthetic sine + trigger in CortiPy)
  1.3/1.4 are documented placeholders (hardware / EEGLAB dependent).

Run with no arguments:
  PYTHONPATH=. python experiments/format_benchmark/run_experiment1.py

Outputs are written under experiments/format_benchmark/results_exp1/
"""

from __future__ import annotations

import hashlib
import copy
import contextlib
import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np

from cortipy.shared import CortiDataset  # type: ignore
from cortipy.shared.bids import ExperimentBinLoader
from cortipy.shared.plotting import apply_standard_montage, topomap_info_from_labels

RESULTS_DIR = Path(__file__).resolve().parent / "results_exp1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Where to pull a sample BIN for bitwise check
DEFAULT_BIN = Path(__file__).resolve().parents[2] / "experiments" / "synData" / "ABR"

DATASETS = {
    "ABR": {
        "bin_dir": "ABR",
        "trace_channel": "T8",
        "trace_label": "ABR Wave V (T8)",
        "topo_time_ms": 7,
    },
    "ASSR": {
        "bin_dir": "ASSR",
        "trace_channel": "T8",
        "trace_label": "ASSR PSD (T8)",
        "topo_freq_hz": 40,
    },
    "Oddball": {
        "bin_dir": "Oddball",
        "trace_channel": "Pz",
        "trace_label": "Oddball ERP (Pz)",
        "topo_time_ms": 300,
        "data_path": "Oddball_scalpdata.bin",
    },
    "VEP": {
        "bin_dir": "VEP",
        "trace_channel": "Oz",
        "trace_label": "VEP (Oz)",
        "topo_time_ms": 100,
    },
    "SSVEP": {
        "bin_dir": "SSVEP",
        "trace_channel": "Oz",
        "trace_label": "SSVEP PSD (Oz)",
        "topo_freq_hz": 10,
    },
    "Sine10Hz": {
        "bin_dir": "ContinuousSine",
        "trace_channel": "Cz",
        "trace_label": "Sine 10 Hz (Cz)",
        "topo_freq_hz": 10,
        "data_path": "ContinuousSine_continuous.bin",
        "sampling_rate": 1000.0,
        "channel_count": 64,
    },
}

EEGLAB_FILES = {
    "ABR": [
        "EEGlab_ABR_WaveV_Topography.pdf",
        "EEGlab_ABR_T8_ERP_vector.pdf",
    ],
    "ASSR": [
        "EEGlab_ASSR_40Hz_Topography.pdf",
        "EEGlab_T8_PSD_vector.pdf",
    ],
    "Oddball": [
        "Oddball_ERP_Pz.pdf",
        "EEGlab_Oddball_Target_Topography.pdf",
    ],
    "VEP": [
        "EEGlab_VEP_Topography_100ms.pdf",
        "EEGlab_Oz_VEP_vector.pdf",
    ],
    "SSVEP": [
        "EEGlab_SSVEP_10Hz_Topography.pdf",
        "EEGlab_Oz_PSD_vector.pdf",
    ],
}

# Matching hints for comparisons: which EEGLAB file goes with which CortiPy kind
COMPARE_RULES = {
    "ABR": {"topomap": "Topography", "trace": "ERP_vector"},
    "ASSR": {"topomap": "Topography", "psd": "PSD_vector"},
    "Oddball": {"topomap": "Topography", "trace": "ERP_Pz"},
    "VEP": {"topomap": "Topography", "trace": "VEP_vector"},
    "SSVEP": {"topomap": "Topography", "psd": "PSD_vector"},
    "Sine10Hz": {"psd": None, "topomap": None, "trace": None},
}


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def bitwise_identity(bin_dir: Path) -> Dict[str, Any]:
    loader = ExperimentBinLoader(bin_dir)
    params_path = bin_dir / "params.json"
    data_path = None
    # pick first .bin
    bins = sorted(bin_dir.glob("*.bin"))
    if bins:
        data_path = bins[0]
    if data_path is None or not params_path.exists():
        raise FileNotFoundError(f"BIN dataset incomplete under {bin_dir}")

    original_hash = _hash_file(data_path)

    ds = CortiDataset.from_bin(bin_dir, data_path=data_path)
    out_dir = RESULTS_DIR / "bitwise_identity"
    out_dir.mkdir(parents=True, exist_ok=True)
    reexport_path, _ = ds.to_bin(out_dir, overwrite=True)
    reexport_hash = _hash_file(reexport_path)

    report_path = out_dir / "bitwise_report.txt"
    report_lines = [
        "Bitwise Identity Check",
        "======================",
        f"Dataset root: {bin_dir}",
        f"Params file:  {params_path}",
        f"Input BIN:    {data_path}",
        f"Input SHA256: {original_hash}",
        f"Output BIN:   {reexport_path}",
        f"Output SHA256:{reexport_hash}",
        f"Identical:    {original_hash == reexport_hash}",
        "",
    ]
    report_path.write_text("\n".join(report_lines))

    return {
        "dataset": str(bin_dir),
        "original": str(data_path),
        "reexport": str(reexport_path),
        "original_hash": original_hash,
        "reexport_hash": reexport_hash,
        "identical": original_hash == reexport_hash,
        "report": str(report_path),
    }


def _epoch_and_metrics(raw: mne.io.BaseRaw, tmin: float = -0.2, tmax: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = mne.find_events(raw, stim_channel="TRIG", shortest_event=1)
    epochs = mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True, picks=["Cz"])
    evoked = epochs.average().data.squeeze()
    data_ep = epochs.get_data().squeeze()
    n_times = data_ep.shape[-1]
    psd, freqs = mne.time_frequency.psd_array_welch(
        data_ep,
        sfreq=raw.info["sfreq"],
        fmin=1.0,
        fmax=50.0,
        average="mean",
        n_fft=min(128, n_times),
        n_per_seg=min(128, n_times),
    )
    psd_mean = psd.mean(axis=0) if psd.ndim > 1 else psd
    times = epochs.times
    return evoked, psd_mean, freqs


def pipeline_equivalence_surrogate() -> Dict[str, Any]:
    ds = CortiDataset.synthetic_sine_trigger()
    raw = ds.raw
    evoked, psd_mean, freqs = _epoch_and_metrics(raw)

    out_dir = RESULTS_DIR / "pipeline_equivalence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot average and PSD
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(evoked.size) / raw.info["sfreq"] + (-0.2), evoked, label="Evoked (Cz)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.title("Synthetic Evoked (CortiPy)")
    plt.tight_layout()
    avg_path = out_dir / "evoked_cortipy.png"
    plt.savefig(avg_path, dpi=150)
    plt.savefig(avg_path.with_suffix(".pdf"))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.semilogy(freqs, psd_mean, label="PSD (Cz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (uV^2/Hz)")
    plt.title("Synthetic PSD (CortiPy)")
    plt.tight_layout()
    psd_path = out_dir / "psd_cortipy.png"
    plt.savefig(psd_path, dpi=150)
    plt.savefig(psd_path.with_suffix(".pdf"))
    plt.close()

    # Placeholder correlation vs EEGLAB (needs numeric export)
    correlations = {
        "evoked_vs_eeglab": None,
        "psd_vs_eeglab": None,
        "note": "Provide EEGLAB numeric exports to compute correlation.",
    }

    return {
        "evoked_plot": str(avg_path),
        "psd_plot": str(psd_path),
        "correlations": correlations,
    }


def document_placeholders() -> Dict[str, Any]:
    return {
        "live_acquisition": "Manual step: connect signal generator 10 Hz / 20 mV to UNICORN and ActiChamp, inspect time-series and PSD visually in CortiPy.",
        "topography_comparison": "Manual step: generate CortiPy topoplots at specified latencies/frequencies and compare to EEGLAB PDFs in synData folders.",
    }


def run_dataset_plots() -> Dict[str, Any]:
    """Run CortiPy evaluators with their built-in plotting instead of local matplotlib code."""
    syn_root = Path(__file__).resolve().parents[2] / "experiments" / "synData"
    out_root = RESULTS_DIR / "cortipy_plots"
    out_root.mkdir(parents=True, exist_ok=True)
    refs_root = RESULTS_DIR / "eeglab_refs"

    outputs: Dict[str, Any] = {}
    for name, cfg in DATASETS.items():
        bin_dir = syn_root / cfg["bin_dir"]
        try:
            ds = CortiDataset.from_bin(
                bin_dir,
                data_path=bin_dir / cfg["data_path"] if cfg.get("data_path") else None,
                channel_count=cfg.get("channel_count"),
                sampling_rate=cfg.get("sampling_rate"),
            )
            params_from_meta = _params_from_metadata(ds.metadata)
            if name == "VEP":
                ds.inject_synthetic_trigger(step_ms=params_from_meta.get("Parameters", {}).get("EpochLength"))
            apply_standard_montage(ds.raw, params=params_from_meta)
            _ensure_plot_channel(params_from_meta, ds.raw, cfg.get("trace_channel"))
            # Persist updated params (e.g., PlotChannelLabel) back into metadata so evaluators use them.
            meta_copy = dict(ds.metadata or {})
            meta_copy["params"] = params_from_meta
            ds.result.metadata = meta_copy

            ds_dir = out_root / name
            ds_dir.mkdir(parents=True, exist_ok=True)
            method_key = _method_for_dataset(name)
            if method_key is None:
                outputs[name] = {"error": "No evaluator mapping"}
                continue

            with _suppress_matplotlib_show():
                evaluation = ds.evaluate(
                    method_key,
                    show_plots=False,
                    save_plots=True,
                    save_dir=ds_dir,
                    figure_prefix=name,
                    store_result=False,
                )

            saved = {}
            if isinstance(evaluation, dict):
                saved_raw = evaluation.get("_figures_saved")
                if isinstance(saved_raw, dict):
                    saved = saved_raw
            if not saved:
                saved = {str(p.name): [str(p)] for p in sorted(ds_dir.glob("*.png"))}
            refs_dir = refs_root / name
            refs = sorted(str(p) for p in refs_dir.glob("*.png")) if refs_dir.exists() else []
            outputs[name] = {
                "plots": saved,
                "eeglab_refs": refs,
                "evaluation_keys": sorted(evaluation.keys()) if isinstance(evaluation, dict) else [],
            }
        except Exception as exc:
            outputs[name] = {"error": f"Failed: {exc}"}
            plt.close("all")
            continue

    return outputs


def _params_from_metadata(meta: Any) -> dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    params = meta.get("params")
    if isinstance(params, dict):
        return copy.deepcopy(params)
    return copy.deepcopy(meta)


def _method_for_dataset(name: str) -> Optional[str]:
    mapping = {
        "ABR": "bera",
        "ASSR": "assr",
        "Oddball": "p300",
        "VEP": "vep",
        "SSVEP": "ssvep",
        "Sine10Hz": "ssvep",
    }
    return mapping.get(name)


def _ensure_plot_channel(params: dict, raw: mne.io.BaseRaw, preferred_label: Optional[str]) -> None:
    """Set PlotChannelLabel/ChannelIpsi if a preferred label is given and found."""
    if not preferred_label or not isinstance(preferred_label, str):
        return
    params_block = params.setdefault("Parameters", {})
    params_block["PlotChannelLabel"] = preferred_label
    try:
        names_lower = [ch.lower() for ch in raw.ch_names]
        idx = names_lower.index(preferred_label.lower())
        params_block["ChannelIpsi"] = idx + 1  # 1-based for evaluators
    except Exception:
        return


@contextlib.contextmanager
def _suppress_matplotlib_show():
    """Prevent evaluator plots from blocking via plt.show()."""
    original = getattr(plt, "show", None)
    try:
        plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
        yield
    finally:
        if original is not None:
            plt.show = original  # type: ignore[assignment]


def copy_eeglab_refs() -> None:
    syn_root = Path(__file__).resolve().parents[2] / "experiments" / "synData"
    dest_root = RESULTS_DIR / "eeglab_refs"
    dest_root.mkdir(parents=True, exist_ok=True)
    for dataset, files in EEGLAB_FILES.items():
        src_dir = syn_root / dataset
        dst_dir = dest_root / dataset
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, dst_dir / fname)
                out_png = dst_dir / Path(fname).with_suffix(".png")
                _pdf_to_png(src, out_png)


def _pdf_to_png(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    # Try PyMuPDF
    try:
        import fitz  # type: ignore

        pdf = fitz.open(src)
        pix = pdf.load_page(0).get_pixmap(dpi=200)
        pix.save(dst)
        return
    except Exception:
        pass
    # Fallback: macOS sips if available
    if shutil.which("sips"):
        try:
            subprocess.run(["sips", "-s", "format", "png", str(src), "--out", str(dst)], check=True)
            return
        except Exception:
            pass


def _pick_best_corti_image(directory: Path, kind: str) -> Optional[Path]:
    """Select the most relevant CortiPy image for a kind when multiple exist."""
    candidates = sorted(directory.glob(f"*{kind}*.png"))
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int]:
        name = path.name.lower()
        penalty = 0
        if "dbg" in name:
            penalty += 5
        if "figure" in name and kind != "figure":
            penalty += 3
        if "psd_2" in name or "psd-2" in name or "psd_3" in name or "psd-3" in name:
            penalty += 2
        # Prefer specific peak topomaps for VEP (so we match P100 vs N75/N135)
        if "p100" in name:
            penalty -= 3 if kind == "topomap" else 0
        if "n75" in name or "n135" in name:
            penalty += 1
        if name.count(kind) > 1:
            penalty += 1
        return penalty, len(name)

    return sorted(candidates, key=score)[0]


def make_side_by_side() -> None:
    """Create side-by-side panels of CortiPy vs EEGLAB for quick comparison."""
    import matplotlib.image as mpimg

    corti_root = RESULTS_DIR / "cortipy_plots"
    eeglab_root = RESULTS_DIR / "eeglab_refs"
    dest_root = RESULTS_DIR / "comparisons"
    dest_root.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS.keys():
        c_dir = corti_root / dataset
        e_dir = eeglab_root / dataset
        if not c_dir.exists() or not e_dir.exists():
            continue
        rules = COMPARE_RULES.get(dataset, {})
        # pick topomap and trace/psd if present
        for kind in ("topomap", "trace", "psd"):
            corti_img = _pick_best_corti_image(c_dir, kind)
            if corti_img is None:
                continue

            pattern = rules.get(kind)
            if not pattern:
                continue
            eeg_candidates = sorted([p for p in e_dir.glob("*.png") if pattern.lower() in p.name.lower()])
            if not eeg_candidates:
                continue
            eeg_img = eeg_candidates[0]

            try:
                img1 = mpimg.imread(corti_img)
                img2 = mpimg.imread(eeg_img)
            except Exception:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(img1)
            axes[0].axis("off")
            axes[0].set_title(f"CortiPy {dataset} {kind}")
            axes[1].imshow(img2)
            axes[1].axis("off")
            axes[1].set_title(f"EEGLAB {dataset} {kind}")
            fig.tight_layout()
            out_path = dest_root / f"{dataset}_{kind}_comparison.png"
            plt.savefig(out_path, dpi=150)
            plt.savefig(out_path.with_suffix(".pdf"))
            plt.close()


def main() -> None:
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)

    results: Dict[str, Any] = {}

    # Copy EEGLAB reference PDFs into results_exp1 for side-by-side comparison
    copy_eeglab_refs()

    # 1.1 Bitwise identity
    try:
        results["bitwise_identity"] = bitwise_identity(DEFAULT_BIN)
    except Exception as exc:
        results["bitwise_identity_error"] = str(exc)

    # 1.2 Pipeline surrogate
    try:
        results["pipeline_equivalence"] = pipeline_equivalence_surrogate()
    except Exception as exc:
        results["pipeline_equivalence_error"] = str(exc)

    # 1.3 / 1.4 placeholders
    results["manual_steps"] = document_placeholders()

    # 1.x Dataset comparisons matching EEGLAB plots
    results["dataset_plots"] = {}
    try:
        results["dataset_plots"] = run_dataset_plots()
    except Exception as exc:
        results["dataset_plots_error"] = str(exc)

    make_side_by_side()

    out_path = RESULTS_DIR / "experiment1_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote Experiment 1 results to {out_path}")


if __name__ == "__main__":
    main()
