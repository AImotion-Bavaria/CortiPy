"""Experiment 2 – File Format Storage & Latency Benchmark.

Loads each synthetic BIN dataset under experiments/synData (plus optionally generated
synthetic variations), exports to BIDS or SBIDS in a single chosen container/format,
and measures size + read/write latency (mean/std over runs). Synthetic datasets can
be generated one at a time to conserve disk space; pass --keep-artifacts to retain
exports. Run with no arguments (defaults to BIDS EDF):
`PYTHONPATH=. python experiments/format_benchmark/run_experiment2.py`
Results are stored as JSON under the sibling `results_exp2/` folder.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import shutil
import numpy as np

import matplotlib.pyplot as plt

from cortipy.shared import CortiDataset  # type: ignore


BIN_DATASETS: Sequence[dict[str, Any]] = (
    {"label": "ABR", "folder": "ABR", "kind": "bin"},
    {"label": "ASSR", "folder": "ASSR", "kind": "bin"},
    {"label": "Oddball", "folder": "Oddball", "kind": "bin"},
    {"label": "VEP", "folder": "VEP", "kind": "bin"},
    {"label": "SSVEP", "folder": "SSVEP", "kind": "bin"},
    {
        "label": "Sine10Hz",
        "folder": "ContinuousSine",
        "kind": "bin",
        "data_path": "ContinuousSine_continuous.bin",
        "channel_count": 64,
        "sampling_rate": 1000.0,
    },
)

SYNTHETIC_DATASETS: Sequence[dict[str, Any]] = (
    {
        "label": "Synth19ch_250Hz_1h",
        "kind": "synthetic",
        "channel_count": 19,
        "sampling_rate": 250.0,
        "duration_s": 3600.0,
        "seed": 7,
        "runs": 1,
        "dtype": "float32",
    },
    {
        "label": "Synth64ch_1000Hz_1h",
        "kind": "synthetic",
        "channel_count": 64,
        "sampling_rate": 1000.0,
        "duration_s": 3600.0,
        "seed": 11,
        "runs": 1,
        "dtype": "float32",
    },
    {
        "label": "Synth256ch_2000Hz_1h",
        "kind": "synthetic",
        "channel_count": 256,
        "sampling_rate": 2000.0,
        "duration_s": 3600.0,
        "seed": 13,
        "runs": 1,
        "dtype": "float32",
    },
)

DATASETS = [*BIN_DATASETS, *SYNTHETIC_DATASETS]

FORMATS = ["parquet", "edf", "zarr", "hdf5"]
RUNS = 1
ALLOWED_FORMATS = ("parquet", "edf", "zarr", "hdf5")
ALLOWED_CONTAINERS = ("bids", "sbids")


def _size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _measure(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return result, t1 - t0


def _mean_std(samples: List[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    if len(samples) == 1:
        return samples[0], 0.0
    return statistics.mean(samples), statistics.pstdev(samples)


def _log_skip(msg: str) -> None:
    print(f"[SKIP] {msg}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _dataset_stats(ds: CortiDataset, *, dataset_kind: str, source: Path) -> Dict[str, Any]:
    raw = ds.result.raw
    sampling_rate = float(raw.info.get("sfreq", ds.result.sampling_rate))
    channel_count = int(raw.info.get("nchan", len(raw.ch_names)))
    duration_s = float(raw.n_times / sampling_rate) if sampling_rate else None
    data_points = int(channel_count * raw.n_times) if duration_s is not None else None

    dtype = getattr(ds.result.data, "dtype", None)
    itemsize = int(dtype.itemsize) if dtype is not None else 8
    approx_raw_bytes = int(data_points * itemsize) if data_points is not None else None
    approx_raw_gb = (approx_raw_bytes / (1024**3)) if approx_raw_bytes is not None else None

    return {
        "dataset_kind": dataset_kind,
        "channel_count": channel_count,
        "sampling_rate": sampling_rate,
        "duration_s": duration_s,
        "data_points": data_points,
        "approx_raw_bytes": approx_raw_bytes,
        "approx_raw_gb": approx_raw_gb,
        "source": str(source),
    }


def _sanitize_task_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _load_dataset(cfg: dict[str, Any], syn_root: Path) -> Tuple[CortiDataset, Dict[str, Any], int, str]:
    label = cfg["label"]
    if cfg["kind"] == "synthetic":
        ds = CortiDataset.from_synthetic(
            sampling_rate=float(cfg["sampling_rate"]),
            duration_s=float(cfg["duration_s"]),
            channel_count=int(cfg["channel_count"]),
            seed=cfg.get("seed"),
            dataset_name=label,
            dtype=cfg.get("dtype"),
        )
        stats = _dataset_stats(ds, dataset_kind="synthetic", source=Path(f"{label}_synthetic"))
    else:
        bin_dir = syn_root / cfg["folder"]
        data_path = cfg.get("data_path")
        ds = CortiDataset.from_bin(
            bin_dir,
            data_path=bin_dir / data_path if data_path else None,
            channel_count=cfg.get("channel_count"),
            sampling_rate=cfg.get("sampling_rate"),
        )
        stats = _dataset_stats(ds, dataset_kind="bin", source=bin_dir)
    runs = int(cfg.get("runs", RUNS))
    task = _sanitize_task_name(cfg.get("task", label))
    return ds, stats, runs, task


def run(
    *,
    containers: Sequence[str] = ALLOWED_CONTAINERS,
    formats: Sequence[str] = ALLOWED_FORMATS,
    keep_artifacts: bool = False,
    purge_outputs: bool = True,
) -> None:
    # Avoid HDF5 file locking issues on shared filesystems.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Not setting position of .* misc channel.*", category=RuntimeWarning)

    repo_root = Path(__file__).resolve().parents[2]
    syn_root = repo_root / "experiments" / "synData"
    out_root = Path(__file__).resolve().parent / "results_exp2"
    out_root.mkdir(parents=True, exist_ok=True)

    containers = [c.lower() for c in containers]
    formats = [f.lower() for f in formats]
    for c in containers:
        if c not in ALLOWED_CONTAINERS:
            raise ValueError(f"container must be one of {ALLOWED_CONTAINERS}, got {c}")
    for f in formats:
        if f not in ALLOWED_FORMATS:
            raise ValueError(f"fmt must be one of {ALLOWED_FORMATS}, got {f}")

    results: List[Dict[str, Any]] = []
    edf_baseline: dict[str, int] = {}

    for cfg in DATASETS:
        label = cfg["label"]
        ds, stats, runs, task = _load_dataset(cfg, syn_root)
        print(f"Processing {label} ({stats['dataset_kind']})")

        dataset_dir = out_root / label
        if purge_outputs and dataset_dir.exists() and not keep_artifacts:
            shutil.rmtree(dataset_dir, ignore_errors=True)

        for fmt in formats:
            for container in containers:
                if container == "bids":
                    bids_root = dataset_dir / "bids" / fmt
                    bids_root.mkdir(parents=True, exist_ok=True)
                    write_times = []
                    bids_data_path = None
                    try:
                        for _ in range(runs):
                            bids_data_path, dt = _measure(
                                ds.to_bids,
                                bids_root,
                                subject="01",
                                task=task,
                                format=fmt,
                                overwrite=True,
                            )
                            write_times.append(dt)
                    except RuntimeError as exc:
                        if "requires the optional dependency 'zarr'" in str(exc).lower():
                            _log_skip(f"BIDS {label} {fmt}: {exc}")
                            results.append(
                                {
                                    "dataset": label,
                                    "container": "bids",
                                    "format": fmt,
                                    "error": str(exc),
                                }
                            )
                            continue
                        raise
                    read_times = []
                    for _ in range(runs):
                        try:
                            _, dt = _measure(
                                CortiDataset.from_bids,
                                bids_root,
                                subject="01",
                                task=task,
                                preload=True,
                                allowed_file_structures=(f".{fmt}",),
                            )
                            read_times.append(dt)
                        except FileNotFoundError as exc:
                            _log_skip(f"BIDS read {label} {fmt}: {exc}")
                            results.append(
                                {
                                    "dataset": label,
                                    "container": "bids",
                                    "format": fmt,
                                    "error": str(exc),
                                }
                            )
                            read_times = []
                            break
                    size = _size_bytes(bids_data_path) if bids_data_path else 0
                    if fmt == "edf":
                        edf_baseline[label] = size
                    res_bids = {
                        "dataset": label,
                        "container": "bids",
                        "format": fmt,
                        "task": task,
                        "runs": runs,
                        "write_times": write_times,
                        "read_times": read_times,
                        "write_mean": _mean_std(write_times)[0],
                        "write_std": _mean_std(write_times)[1],
                        "read_mean": _mean_std(read_times)[0],
                        "read_std": _mean_std(read_times)[1],
                        "size_bytes": size,
                        "codec": "snappy (default)" if fmt == "parquet" else "n/a",
                        **stats,
                    }
                    results.append(res_bids)
                    if not keep_artifacts:
                        shutil.rmtree(bids_root, ignore_errors=True)

                elif container == "sbids":
                    sbids_dir = dataset_dir / "sbids"
                    sbids_dir.mkdir(parents=True, exist_ok=True)
                    sbids_meta = sbids_dir / f"{label}_{fmt}.jsonld"
                    write_times = []
                    raw_path = None
                    try:
                        for _ in range(runs):
                            raw_path, dt = _measure(ds.to_sbids, sbids_meta, export_format=fmt)
                            write_times.append(dt)
                    except RuntimeError as exc:
                        if "requires the optional dependency 'zarr'" in str(exc).lower():
                            _log_skip(f"SBIDS {label} {fmt}: {exc}")
                            results.append(
                                {
                                    "dataset": label,
                                    "container": "sbids",
                                    "format": fmt,
                                    "error": str(exc),
                                }
                            )
                            continue
                        raise
                    read_times = []
                    for _ in range(runs):
                        try:
                            _, dt = _measure(
                                CortiDataset.from_sbids,
                                sbids_meta,
                                data_roots=[sbids_dir],
                                preload=True,
                            )
                            read_times.append(dt)
                        except FileNotFoundError as exc:
                            _log_skip(f"SBIDS read {label} {fmt}: {exc}")
                            results.append(
                                {
                                    "dataset": label,
                                    "container": "sbids",
                                    "format": fmt,
                                    "error": str(exc),
                                }
                            )
                            read_times = []
                            break
                    size = _size_bytes(raw_path) if raw_path else 0
                    res_sbids = {
                        "dataset": label,
                        "container": "sbids",
                        "format": fmt,
                        "task": task,
                        "runs": runs,
                        "write_times": write_times,
                        "read_times": read_times,
                        "write_mean": _mean_std(write_times)[0],
                        "write_std": _mean_std(write_times)[1],
                        "read_mean": _mean_std(read_times)[0],
                        "read_std": _mean_std(read_times)[1],
                        "size_bytes": size,
                        "codec": "snappy (default)" if fmt == "parquet" else "n/a",
                        **stats,
                    }
                    results.append(res_sbids)
                    if not keep_artifacts:
                        shutil.rmtree(sbids_dir, ignore_errors=True)

        del ds

        if stats.get("dataset_kind") == "synthetic" and not keep_artifacts:
            dataset_dir = out_root / label
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir, ignore_errors=True)
                print(f"[CLEANUP] removed artifacts for {label}")

    # Compute reductions vs EDF baseline per dataset
    for entry in results:
        base = edf_baseline.get(entry["dataset"])
        size = entry.get("size_bytes")
        entry["reduction_vs_edf"] = (
            1.0 - (size / base) if base and base and size is not None and size > 0 else None
        )

    out_path = out_root / "experiment2_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"Wrote results to {out_path}")

    _plot_metrics(results, out_root)
    _plot_latency_boxplots(results, out_root)
    _plot_differences(results, out_root)
    _plot_size_trend(results, out_root)


def _plot_metrics(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Emit simple comparison plots for size and latency."""
    def plot_metric(metric: str, error_key: str | None, fname: str, ylabel: str) -> None:
        filtered = [r for r in results if metric in r and isinstance(r.get(metric), (int, float))]
        if not filtered:
            return
        labels = [f"{r['dataset']}-{r['container']}-{r['format']}" for r in filtered]
        values = [r[metric] for r in filtered]
        errors = [r.get(error_key, 0.0) if error_key else 0.0 for r in filtered]

        plt.figure(figsize=(max(6, len(values) * 0.5), 4))
        plt.bar(range(len(values)), values, yerr=errors if error_key else None, alpha=0.8, color="steelblue")
        plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel(ylabel)
        plt.title(f"Experiment 2 – {ylabel}")
        plt.tight_layout()
        out_file = out_root / fname
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")

    plot_metric("size_bytes", None, "sizes.png", "Size (bytes)")
    plot_metric("write_mean", "write_std", "write_latency.png", "Write latency (s)")
    plot_metric("read_mean", "read_std", "read_latency.png", "Read latency (s)")


def _plot_differences(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Plot SBIDS minus BIDS differences (green faster/smaller, red slower/larger)."""
    pairs: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for r in results:
        key = (r.get("dataset"), r.get("format"))
        container = r.get("container")
        if not key[0] or not key[1] or container not in {"bids", "sbids"}:
            continue
        bucket = pairs.setdefault(key, {})
        bucket[container] = r

    diffs_size = []
    labels_size = []
    diffs_write = []
    labels_write = []
    diffs_read = []
    labels_read = []

    for (dataset, fmt), buckets in pairs.items():
        b = buckets.get("bids")
        s = buckets.get("sbids")
        if not b or not s:
            continue
        if isinstance(b.get("size_bytes"), (int, float)) and isinstance(s.get("size_bytes"), (int, float)):
            diffs_size.append(s["size_bytes"] - b["size_bytes"])
            labels_size.append(f"{dataset}-{fmt}")
        if isinstance(b.get("write_mean"), (int, float)) and isinstance(s.get("write_mean"), (int, float)):
            diffs_write.append(s["write_mean"] - b["write_mean"])
            labels_write.append(f"{dataset}-{fmt}")
        if isinstance(b.get("read_mean"), (int, float)) and isinstance(s.get("read_mean"), (int, float)):
            diffs_read.append(s["read_mean"] - b["read_mean"])
            labels_read.append(f"{dataset}-{fmt}")

    def plot_diff(vals: List[float], labels: List[str], fname: str, ylabel: str) -> None:
        if not vals:
            return
        colors = ["green" if v < 0 else "red" for v in vals]
        plt.figure(figsize=(max(6, len(vals) * 0.5), 4))
        plt.bar(range(len(vals)), vals, color=colors, alpha=0.8)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(range(len(vals)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel(ylabel)
        plt.title(f"SBIDS minus BIDS ({ylabel})")
        plt.tight_layout()
        out_file = out_root / fname
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")

    plot_diff(diffs_size, labels_size, "diff_size.png", "Size difference (bytes)")
    plot_diff(diffs_write, labels_write, "diff_write_latency.png", "Write latency difference (s)")
    plot_diff(diffs_read, labels_read, "diff_read_latency.png", "Read latency difference (s)")


def _plot_latency_boxplots(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Boxplots of read/write times grouped by container/format."""
    metrics = [("write_times", "Write latency (s)", "latency_box_write.png"), ("read_times", "Read latency (s)", "latency_box_read.png")]
    for key, ylabel, fname in metrics:
        buckets: Dict[tuple[str, str], List[float]] = {}
        for r in results:
            if not isinstance(r.get(key), list):
                continue
            container = r.get("container")
            fmt = r.get("format")
            if container not in {"bids", "sbids"} or not fmt:
                continue
            buckets.setdefault((container, fmt), []).extend([v for v in r[key] if isinstance(v, (int, float))])
        if not buckets:
            continue
        labels = [f"{c.upper()}-{f}" for (c, f) in buckets]
        data = [buckets[k] for k in buckets]
        plt.figure(figsize=(max(6, len(data) * 1.2), 5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Experiment 2 – {ylabel} (boxplots)")
        plt.tight_layout()
        out_file = out_root / fname
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _plot_size_trend(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Line chart comparing BIDS vs SBIDS sizes as dataset scale grows."""
    def plot_subset(entries: List[Dict[str, Any]], suffix: str) -> None:
        if not entries:
            return
        plt.figure(figsize=(10, 6))
        grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for r in entries:
            key = (r["format"], r["container"])
            grouped.setdefault(key, []).append(r)

        for (fmt, container), items in grouped.items():
            items_sorted = sorted(items, key=lambda x: x.get("approx_raw_gb", 0.0))
            x = [i["approx_raw_gb"] for i in items_sorted]
            y = [i["size_bytes"] / (1024**3) for i in items_sorted]
            labels = [i["dataset"] for i in items_sorted]
            plt.plot(x, y, marker="o", label=f"{fmt.upper()} {container.upper()}")
            for xi, yi, lbl in zip(x, y, labels):
                plt.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        plt.xlabel("Approx. uncompressed raw size (GB)")
        plt.ylabel("Exported size (GB)")
        plt.title(f"BIDS vs SBIDS size scaling across datasets ({suffix})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        out_file = out_root / f"size_trend_{suffix}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")

    base_entries = [
        r
        for r in results
        if r.get("container") in {"bids", "sbids"}
        and isinstance(r.get("size_bytes"), (int, float))
        and isinstance(r.get("approx_raw_gb"), (int, float))
    ]
    plot_subset(base_entries, "all")

    synth_entries = [r for r in base_entries if r.get("dataset_kind") == "synthetic"]
    bin_entries = [r for r in base_entries if r.get("dataset_kind") == "bin"]
    plot_subset(synth_entries, "synthetic")
    plot_subset(bin_entries, "bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2 – format storage & latency benchmark")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep per-dataset BIDS/SBIDS exports (synthetic datasets are deleted by default).",
    )
    parser.add_argument(
        "--containers",
        type=str,
        default=",".join(ALLOWED_CONTAINERS),
        help="Comma-separated containers to export/read (any of bids,sbids). Defaults to both.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=",".join(ALLOWED_FORMATS),
        help="Comma-separated formats to export/read (parquet,edf,zarr,hdf5). Defaults to all.",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Do not purge existing outputs for the dataset before writing new ones.",
    )
    args = parser.parse_args()
    containers = [c.strip() for c in args.containers.split(",") if c.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    run(
        containers=containers,
        formats=formats,
        keep_artifacts=args.keep_artifacts,
        purge_outputs=not args.no_purge,
    )
