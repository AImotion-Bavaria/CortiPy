"""Experiment 2 – File Format Storage & Latency Benchmark.

Loads optionally generated synthetic variations, exports to BIDS or SBIDS in a single chosen container/format,
and measures size + read/write latency (mean/std over runs). Select datasets via --datasets.
Synthetic datasets can
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

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

import matplotlib.pyplot as plt

from cortipy.shared import CortiDataset  # type: ignore

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

DATASETS = [*SYNTHETIC_DATASETS]

FORMATS = ["parquet", "edf", "zarr", "hdf5"]
RUNS = 1
ALLOWED_FORMATS = ("parquet", "edf", "zarr", "hdf5")
ALLOWED_CONTAINERS = ("bids", "sbids")


def _size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_dir():
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total
    return path.stat().st_size


def _mean_std(samples: List[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    if len(samples) == 1:
        return samples[0], 0.0
    return statistics.mean(samples), statistics.pstdev(samples)


def _log_skip(msg: str) -> None:
    print(f"[SKIP] {msg}")


def _measure(fn, *args, **kwargs):
    proc = psutil.Process() if psutil else None
    t0 = time.perf_counter()
    cpu0 = proc.cpu_times() if proc else None
    rss0 = proc.memory_info().rss if proc else None
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    cpu1 = proc.cpu_times() if proc else None
    rss1 = proc.memory_info().rss if proc else None

    dt = t1 - t0
    cpu_delta = None
    if cpu0 and cpu1:
        cpu_delta = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    rss_peak = None
    if rss0 is not None and rss1 is not None:
        rss_peak = max(rss0, rss1)
    return result, dt, cpu_delta, rss_peak


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
    bytes_per_sample = (approx_raw_bytes / data_points) if (approx_raw_bytes and data_points) else None

    return {
        "dataset_kind": dataset_kind,
        "channel_count": channel_count,
        "sampling_rate": sampling_rate,
        "duration_s": duration_s,
        "data_points": data_points,
        "approx_raw_bytes": approx_raw_bytes,
        "approx_raw_gb": approx_raw_gb,
        "bytes_per_sample": bytes_per_sample,
        "source": str(source),
    }

def _filter_by_scope(results: List[Dict[str, Any]], scope: str) -> List[Dict[str, Any]]:
    if scope == "all":
        filtered = results
    else:
        filtered = [r for r in results if r.get("dataset_kind") == scope]
    for r in filtered:
        r["_scope"] = scope
    return filtered


def _sanitize_task_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _find_sbids_raw(sbids_dir: Path, fmt: str, stem: str) -> Path | None:
    raw_dir = sbids_dir / "raw_data"
    if not raw_dir.exists():
        return None
    ext_map = {"parquet": "parquet", "edf": "edf", "hdf5": "hdf5", "zarr": "zarr"}
    ext = ext_map.get(fmt, fmt)
    if ext == "zarr":
        candidates = list(raw_dir.glob("*.zarr"))
    else:
        candidates = list(raw_dir.glob(f"*.{ext}"))
    if not candidates:
        return None
    # Prefer matching stem if present
    for c in candidates:
        if stem in c.stem:
            return c
    return candidates[0]


def _load_dataset(cfg: dict[str, Any], default_runs: int) -> Tuple[CortiDataset, Dict[str, Any], int, str]:
    label = cfg["label"]
    ds = CortiDataset.from_synthetic(
        sampling_rate=float(cfg["sampling_rate"]),
        duration_s=float(cfg["duration_s"]),
        channel_count=int(cfg["channel_count"]),
        seed=cfg.get("seed"),
        dataset_name=label,
        dtype=cfg.get("dtype"),
    )
    stats = _dataset_stats(ds, dataset_kind="synthetic", source=Path(f"{label}_synthetic"))
    runs = int(cfg.get("runs", default_runs))
    task = _sanitize_task_name(cfg.get("task", label))
    return ds, stats, runs, task


def run(
    *,
    containers: Sequence[str] = ALLOWED_CONTAINERS,
    formats: Sequence[str] = ALLOWED_FORMATS,
    datasets: Sequence[str] | None = None,
    runs_override: int | None = None,
    keep_artifacts: bool = False,
    purge_outputs: bool = True,
    plot_scope: str = "all",
    plots_only: bool = False,
    results_path: str | Path | None = None,
) -> None:
    # Avoid HDF5 file locking issues on shared filesystems.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Not setting position of .* misc channel.*", category=RuntimeWarning)

    out_root = Path(__file__).resolve().parent / "results_exp2"
    out_root.mkdir(parents=True, exist_ok=True)
    res_path = Path(results_path) if results_path else out_root / "experiment2_results.json"

    if plots_only:
        if not res_path.exists():
            raise FileNotFoundError(f"Results JSON not found: {res_path}")
        results = json.loads(res_path.read_text())
        filtered_results = _filter_by_scope(results, plot_scope)
        _plot_metrics(filtered_results, out_root)
        _plot_latency_boxplots(filtered_results, out_root)
        _plot_differences(filtered_results, out_root)
        _plot_latency_pairs(filtered_results, out_root)
        _plot_size_trend(filtered_results, out_root)
        _plot_latency_size_trend(results, out_root)
        _plot_synth_breakdowns(results, out_root)
        _plot_speedup_heatmap(results, out_root)
        _plot_resource_correlation(results, out_root)
        _plot_failure_summary(results, out_root)
        _plot_efficiency_bars(results, out_root)
        _plot_synth_multistrip(results, out_root)
        _plot_rw_throughput(results, out_root)
        _plot_extension_box(results, out_root)
        return

    containers = [c.lower() for c in containers]
    formats = [f.lower() for f in formats]
    for c in containers:
        if c not in ALLOWED_CONTAINERS:
            raise ValueError(f"container must be one of {ALLOWED_CONTAINERS}, got {c}")
    for f in formats:
        if f not in ALLOWED_FORMATS:
            raise ValueError(f"fmt must be one of {ALLOWED_FORMATS}, got {f}")
    if plot_scope not in {"all", "synthetic", "bin"}:
        raise ValueError("plot_scope must be one of: all, synthetic, bin")

    default_runs = runs_override if runs_override is not None else RUNS
    selected = DATASETS if datasets is None else [cfg for cfg in DATASETS if cfg["label"] in datasets]
    missing = set(datasets or []) - {cfg["label"] for cfg in DATASETS}
    if missing:
        print(f"[WARN] Unknown dataset labels requested: {', '.join(sorted(missing))}")

    results: List[Dict[str, Any]] = []
    edf_baseline: dict[str, int] = {}

    for cfg in selected:
        label = cfg["label"]
        ds, stats, runs, task = _load_dataset(cfg, default_runs=default_runs)
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
                    write_cpu = []
                    write_rss = []
                    bids_data_path = None
                    try:
                        for _ in range(runs):
                            bids_data_path, dt, cpu_dt, rss_peak = _measure(
                                ds.to_bids,
                                bids_root,
                                subject="01",
                                task=task,
                                format=fmt,
                                overwrite=True,
                            )
                            write_times.append(dt)
                            if cpu_dt is not None:
                                write_cpu.append(cpu_dt)
                            if rss_peak is not None:
                                write_rss.append(rss_peak)
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
                    read_cpu = []
                    read_rss = []
                    for _ in range(runs):
                        try:
                            _, dt, cpu_dt, rss_peak = _measure(
                                CortiDataset.from_bids,
                                bids_root,
                                subject="01",
                                task=task,
                                preload=True,
                                allowed_file_structures=(f".{fmt}",),
                            )
                            read_times.append(dt)
                            if cpu_dt is not None:
                                read_cpu.append(cpu_dt)
                            if rss_peak is not None:
                                read_rss.append(rss_peak)
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
                    write_thr = (size / sum(write_times)) if size and write_times and sum(write_times) > 0 else None
                    read_thr = (size / sum(read_times)) if size and read_times and sum(read_times) > 0 else None
                    res_bids = {
                        "dataset": label,
                        "container": "bids",
                        "format": fmt,
                        "task": task,
                        "runs": runs,
                        "write_times": write_times,
                        "read_times": read_times,
                        "write_cpu_times": write_cpu,
                        "read_cpu_times": read_cpu,
                        "write_rss_list": write_rss,
                        "read_rss_list": read_rss,
                        "write_mean": _mean_std(write_times)[0],
                        "write_std": _mean_std(write_times)[1],
                        "read_mean": _mean_std(read_times)[0],
                        "read_std": _mean_std(read_times)[1],
                        "write_total": sum(write_times),
                        "read_total": sum(read_times),
                        "write_throughput": write_thr,
                        "read_throughput": read_thr,
                        "write_cv": (_mean_std(write_times)[1] / _mean_std(write_times)[0]) if _mean_std(write_times)[0] else None,
                        "read_cv": (_mean_std(read_times)[1] / _mean_std(read_times)[0]) if _mean_std(read_times)[0] else None,
                        "write_cpu_mean": _mean_std(write_cpu)[0] if write_cpu else None,
                        "read_cpu_mean": _mean_std(read_cpu)[0] if read_cpu else None,
                        "write_cpu_total": sum(write_cpu) if write_cpu else None,
                        "read_cpu_total": sum(read_cpu) if read_cpu else None,
                        "write_rss_max": max(write_rss) if write_rss else None,
                        "read_rss_max": max(read_rss) if read_rss else None,
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
                    write_cpu = []
                    write_rss = []
                    raw_path = None
                    try:
                        for _ in range(runs):
                            raw_path, dt, cpu_dt, rss_peak = _measure(ds.to_sbids, sbids_meta, export_format=fmt)
                            write_times.append(dt)
                            if cpu_dt is not None:
                                write_cpu.append(cpu_dt)
                            if rss_peak is not None:
                                write_rss.append(rss_peak)
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
                    read_cpu = []
                    read_rss = []
                    for _ in range(runs):
                        try:
                            _, dt, cpu_dt, rss_peak = _measure(
                                CortiDataset.from_sbids,
                                sbids_meta,
                                data_roots=[sbids_dir],
                                preload=True,
                            )
                            read_times.append(dt)
                            if cpu_dt is not None:
                                read_cpu.append(cpu_dt)
                            if rss_peak is not None:
                                read_rss.append(rss_peak)
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
                    raw_data_path = _find_sbids_raw(sbids_dir, fmt, Path(sbids_meta).stem)
                    size = _size_bytes(raw_data_path) if raw_data_path else 0
                    write_thr = (size / sum(write_times)) if size and write_times and sum(write_times) > 0 else None
                    read_thr = (size / sum(read_times)) if size and read_times and sum(read_times) > 0 else None
                    res_sbids = {
                        "dataset": label,
                        "container": "sbids",
                        "format": fmt,
                        "task": task,
                        "runs": runs,
                        "write_times": write_times,
                        "read_times": read_times,
                        "write_cpu_times": write_cpu,
                        "read_cpu_times": read_cpu,
                        "write_rss_list": write_rss,
                        "read_rss_list": read_rss,
                        "write_mean": _mean_std(write_times)[0],
                        "write_std": _mean_std(write_times)[1],
                        "read_mean": _mean_std(read_times)[0],
                        "read_std": _mean_std(read_times)[1],
                        "write_total": sum(write_times),
                        "read_total": sum(read_times),
                        "write_throughput": write_thr,
                        "read_throughput": read_thr,
                        "write_cv": (_mean_std(write_times)[1] / _mean_std(write_times)[0]) if _mean_std(write_times)[0] else None,
                        "read_cv": (_mean_std(read_times)[1] / _mean_std(read_times)[0]) if _mean_std(read_times)[0] else None,
                        "write_cpu_mean": _mean_std(write_cpu)[0] if write_cpu else None,
                        "read_cpu_mean": _mean_std(read_cpu)[0] if read_cpu else None,
                        "write_cpu_total": sum(write_cpu) if write_cpu else None,
                        "read_cpu_total": sum(read_cpu) if read_cpu else None,
                        "write_rss_max": max(write_rss) if write_rss else None,
                        "read_rss_max": max(read_rss) if read_rss else None,
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

    out_path = res_path
    out_path.write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"Wrote results to {out_path}")

    filtered_results = _filter_by_scope(results, plot_scope)
    _plot_metrics(filtered_results, out_root)
    _plot_latency_boxplots(filtered_results, out_root)
    _plot_differences(filtered_results, out_root)
    _plot_latency_pairs(filtered_results, out_root)
    _plot_size_trend(filtered_results, out_root)
    _plot_latency_size_trend(results, out_root)
    _plot_synth_breakdowns(results, out_root)
    _plot_speedup_heatmap(results, out_root)
    _plot_resource_correlation(results, out_root)
    _plot_failure_summary(results, out_root)
    _plot_efficiency_bars(results, out_root)
    _plot_synth_multistrip(results, out_root)
    _plot_rw_throughput(results, out_root)
    _plot_extension_box(results, out_root)


def _plot_metrics(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Emit simple comparison plots for size and latency."""
    if not results:
        return

    def plot_metric(metric: str, error_key: str | None, fname: str, ylabel: str, scope: str, scale: float = 1.0) -> None:
        filtered = [r for r in results if metric in r and isinstance(r.get(metric), (int, float))]
        if not filtered:
            return
        labels = [f"{r['dataset']}-{r['container']}-{r['format']}" for r in filtered]
        values = [r[metric] * scale for r in filtered]
        errors = [r.get(error_key, 0.0) if error_key else 0.0 for r in filtered]

        plt.figure(figsize=(max(6, len(values) * 0.5), 4))
        plt.bar(range(len(values)), values, yerr=errors if error_key else None, alpha=0.8, color="steelblue")
        plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel(ylabel)
        plt.title(f"Experiment 2 – {ylabel}")
        plt.tight_layout()
        suffix = f"_{scope}" if scope != "all" else ""
        out_file = out_root / f"{fname[:-4]}{suffix}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")

    scope = results[0].get("_scope", "all")
    plot_metric("size_bytes", None, "sizes.png", "Size (bytes)", scope)
    plot_metric("write_mean", "write_std", "write_latency.png", "Write latency (s)", scope)
    plot_metric("read_mean", "read_std", "read_latency.png", "Read latency (s)", scope)
    plot_metric("write_total", None, "write_total.png", "Write total (s)", scope)
    plot_metric("read_total", None, "read_total.png", "Read total (s)", scope)
    plot_metric("write_throughput", None, "write_throughput.png", "Write throughput (bytes/s)", scope)
    plot_metric("read_throughput", None, "read_throughput.png", "Read throughput (bytes/s)", scope)
    plot_metric("write_throughput", None, "write_throughput_mb.png", "Write throughput (MB/s)", scope, scale=1 / (1024 * 1024))
    plot_metric("read_throughput", None, "read_throughput_mb.png", "Read throughput (MB/s)", scope, scale=1 / (1024 * 1024))
    plot_metric("write_cv", None, "write_cv.png", "Write coefficient of variation", scope)
    plot_metric("read_cv", None, "read_cv.png", "Read coefficient of variation", scope)
    plot_metric("write_cpu_total", None, "write_cpu_total.png", "CPU time write (s)", scope)
    plot_metric("read_cpu_total", None, "read_cpu_total.png", "CPU time read (s)", scope)
    plot_metric("write_rss_max", None, "write_rss_max.png", "Max RSS write (bytes)", scope)
    plot_metric("read_rss_max", None, "read_rss_max.png", "Max RSS read (bytes)", scope)


def _plot_differences(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Plot SBIDS minus BIDS differences (green faster/smaller, red slower/larger)."""
    if not results:
        return
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
    diffs_write_total = []
    labels_write_total = []
    diffs_read_total = []
    labels_read_total = []
    diffs_write_thr = []
    labels_write_thr = []
    diffs_read_thr = []
    labels_read_thr = []

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
        if isinstance(b.get("write_total"), (int, float)) and isinstance(s.get("write_total"), (int, float)):
            diffs_write_total.append(s["write_total"] - b["write_total"])
            labels_write_total.append(f"{dataset}-{fmt}")
        if isinstance(b.get("read_total"), (int, float)) and isinstance(s.get("read_total"), (int, float)):
            diffs_read_total.append(s["read_total"] - b["read_total"])
            labels_read_total.append(f"{dataset}-{fmt}")
        if isinstance(b.get("write_throughput"), (int, float)) and isinstance(s.get("write_throughput"), (int, float)):
            diffs_write_thr.append(s["write_throughput"] - b["write_throughput"])
            labels_write_thr.append(f"{dataset}-{fmt}")
        if isinstance(b.get("read_throughput"), (int, float)) and isinstance(s.get("read_throughput"), (int, float)):
            diffs_read_thr.append(s["read_throughput"] - b["read_throughput"])
            labels_read_thr.append(f"{dataset}-{fmt}")

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

    scope = results[0].get("_scope", "all") if results else "all"
    suffix = f"_{scope}" if scope != "all" else ""
    plot_diff(diffs_size, labels_size, f"diff_size{suffix}.png", "Size difference (bytes)")
    plot_diff(diffs_write, labels_write, f"diff_write_latency{suffix}.png", "Write latency difference (s)")
    plot_diff(diffs_read, labels_read, f"diff_read_latency{suffix}.png", "Read latency difference (s)")
    plot_diff(diffs_write_total, labels_write_total, f"diff_write_total{suffix}.png", "Write total difference (s)")
    plot_diff(diffs_read_total, labels_read_total, f"diff_read_total{suffix}.png", "Read total difference (s)")
    plot_diff(diffs_write_thr, labels_write_thr, f"diff_write_throughput{suffix}.png", "Write throughput difference (bytes/s)")
    plot_diff(diffs_read_thr, labels_read_thr, f"diff_read_throughput{suffix}.png", "Read throughput difference (bytes/s)")


def _plot_latency_boxplots(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Boxplots of read/write times grouped by container/format."""
    if not results:
        return
    metrics = [
        ("write_times", lambda r: r.get("write_times"), "Write latency (s)", "latency_box_write.png", 1.0),
        ("read_times", lambda r: r.get("read_times"), "Read latency (s)", "latency_box_read.png", 1.0),
        ("write_times_total", lambda r: [sum(r.get("write_times", []))] if r.get("write_times") else None, "Write total (s)", "latency_box_write_total.png", 1.0),
        ("read_times_total", lambda r: [sum(r.get("read_times", []))] if r.get("read_times") else None, "Read total (s)", "latency_box_read_total.png", 1.0),
        (
            "write_thr_runs_mb",
            lambda r: [
                (r.get("size_bytes") / t) / (1024 * 1024)
                for t in r.get("write_times", [])
                if t and r.get("size_bytes")
            ]
            if r.get("write_times")
            else None,
            "Write throughput (MB/s)",
            "latency_box_write_thr_mb.png",
            1.0,
        ),
        (
            "read_thr_runs_mb",
            lambda r: [
                (r.get("size_bytes") / t) / (1024 * 1024)
                for t in r.get("read_times", [])
                if t and r.get("size_bytes")
            ]
            if r.get("read_times")
            else None,
            "Read throughput (MB/s)",
            "latency_box_read_thr_mb.png",
            1.0,
        ),
        (
            "write_thr_mb",
            lambda r: [(r.get("size_bytes") / t) / (1024 * 1024) for t in r.get("write_times", []) if t and r.get("size_bytes")]
            if r.get("write_times")
            else None,
            "Write throughput (MB/s)",
            "latency_box_write_thr_mb.png",
            1.0,
        ),
        (
            "read_thr_mb",
            lambda r: [(r.get("size_bytes") / t) / (1024 * 1024) for t in r.get("read_times", []) if t and r.get("size_bytes")]
            if r.get("read_times")
            else None,
            "Read throughput (MB/s)",
            "latency_box_read_thr_mb.png",
            1.0,
        ),
        ("write_cpu_times", lambda r: r.get("write_cpu_times"), "CPU time write (s)", "latency_box_write_cpu.png", 1.0),
        ("read_cpu_times", lambda r: r.get("read_cpu_times"), "CPU time read (s)", "latency_box_read_cpu.png", 1.0),
        ("write_cpu_total", lambda r: [sum(r.get("write_cpu_times", []))] if r.get("write_cpu_times") else None, "CPU time write total (s)", "latency_box_write_cpu_total.png", 1.0),
        ("read_cpu_total", lambda r: [sum(r.get("read_cpu_times", []))] if r.get("read_cpu_times") else None, "CPU time read total (s)", "latency_box_read_cpu_total.png", 1.0),
        ("write_rss_list", lambda r: r.get("write_rss_list"), "Max RSS write (MB)", "latency_box_write_rss.png", 1 / (1024 * 1024)),
        ("read_rss_list", lambda r: r.get("read_rss_list"), "Max RSS read (MB)", "latency_box_read_rss.png", 1 / (1024 * 1024)),
        ("write_rss_max_total", lambda r: [max(r.get("write_rss_list", []))] if r.get("write_rss_list") else None, "Peak RSS write (MB)", "latency_box_write_rss_peak.png", 1 / (1024 * 1024)),
        ("read_rss_max_total", lambda r: [max(r.get("read_rss_list", []))] if r.get("read_rss_list") else None, "Peak RSS read (MB)", "latency_box_read_rss_peak.png", 1 / (1024 * 1024)),
    ]
    for _name, extractor, ylabel, fname, factor in metrics:
        buckets: Dict[tuple[str, str], List[float]] = {}
        for r in results:
            values = extractor(r)
            if values is None:
                continue
            container = r.get("container")
            fmt = r.get("format")
            if container not in {"bids", "sbids"} or not fmt:
                continue
            vals = [v * factor for v in values if isinstance(v, (int, float))]
            if not vals:
                continue
            buckets.setdefault((container, fmt), []).extend(vals)
        if not buckets:
            continue
        labels = [f"{c.upper()}-{f}" for (c, f) in buckets]
        data = [buckets[k] for k in buckets]
        plt.figure(figsize=(max(6, len(data) * 1.2), 5))
        plt.boxplot(data, tick_labels=labels, showfliers=False)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Experiment 2 – {ylabel} (boxplots)")
        plt.tight_layout()
        scope = results[0].get("_scope", "all") if results else "all"
        suffix = f"_{scope}" if scope != "all" else ""
        out_file = out_root / f"{fname[:-4]}{suffix}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _plot_size_trend(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Line chart comparing BIDS vs SBIDS sizes as dataset scale grows."""
    entries = [
        r
        for r in results
        if r.get("container") in {"bids", "sbids"}
        and isinstance(r.get("size_bytes"), (int, float))
        and isinstance(r.get("approx_raw_gb"), (int, float))
    ]
    if not entries:
        return

    scope = results[0].get("_scope", "all") if results else "all"
    suffix = f"_{scope}" if scope != "all" else ""

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
    plt.title("BIDS vs SBIDS size scaling across datasets")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_file = out_root / f"size_trend{suffix}.png"
    plt.savefig(out_file, dpi=150)
    plt.savefig(out_file.with_suffix(".pdf"))
    plt.close()
    print(f"Wrote plot: {out_file}")


def _plot_latency_size_trend(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Scatter/line plots of latency vs exported size for synthetic datasets."""
    entries = [
        r
        for r in results
        if r.get("dataset_kind") == "synthetic"
        and r.get("container") in {"bids", "sbids"}
        and isinstance(r.get("size_bytes"), (int, float))
    ]
    if not entries:
        return

    def plot(metric: str, title: str, fname: str) -> None:
        pts = [e for e in entries if isinstance(e.get(metric), (int, float))]
        if not pts:
            return
        plt.figure(figsize=(10, 6))
        grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for e in pts:
            key = (e["format"], e["container"])
            grouped.setdefault(key, []).append(e)
        for (fmt, container), items in grouped.items():
            items_sorted = sorted(items, key=lambda x: x.get("size_bytes", 0))
            x = [i["size_bytes"] / (1024**3) for i in items_sorted]
            y = [i[metric] for i in items_sorted]
            labels = [i["dataset"] for i in items_sorted]
            plt.plot(x, y, marker="o", label=f"{fmt.upper()} {container.upper()}")
            for xi, yi, lbl in zip(x, y, labels):
                plt.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        plt.xlabel("Exported size (GB)")
        plt.ylabel(title)
        plt.title(f"Synthetic datasets – {title} vs size")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        out_file = out_root / fname
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")

    plot("write_mean", "Write latency (s)", "latency_vs_size_write_synthetic.png")
    plot("read_mean", "Read latency (s)", "latency_vs_size_read_synthetic.png")
    plot("write_total", "Write total (s)", "total_vs_size_write_synthetic.png")
    plot("read_total", "Read total (s)", "total_vs_size_read_synthetic.png")
    plot("write_cpu_total", "CPU time write (s)", "cpu_vs_size_write_synthetic.png")
    plot("read_cpu_total", "CPU time read (s)", "cpu_vs_size_read_synthetic.png")
    plot("write_rss_max", "Max RSS write (bytes)", "rss_vs_size_write_synthetic.png")
    plot("read_rss_max", "Max RSS read (bytes)", "rss_vs_size_read_synthetic.png")
    plot("write_throughput", "Write throughput (bytes/s)", "throughput_vs_size_write_synthetic.png")
    plot("read_throughput", "Read throughput (bytes/s)", "throughput_vs_size_read_synthetic.png")
    plot("write_throughput", "Write throughput (MB/s)", "throughput_mb_vs_size_write_synthetic.png")
    plot("read_throughput", "Read throughput (MB/s)", "throughput_mb_vs_size_read_synthetic.png")
    plot("write_cpu_total", "CPU time write (s)", "cpu_vs_size_write_synthetic.png")
    plot("read_cpu_total", "CPU time read (s)", "cpu_vs_size_read_synthetic.png")


def _plot_synth_breakdowns(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Per-synthetic-dataset and per-format breakdown charts (synthetic only)."""
    synth = [
        r
        for r in results
        if r.get("dataset_kind") == "synthetic"
        and r.get("container") in {"bids", "sbids"}
        and isinstance(r.get("size_bytes"), (int, float))
    ]
    if not synth:
        return

    metrics = [
        ("size_bytes", None, "Size (bytes)", "size"),
        ("write_mean", "write_std", "Write latency (s)", "write_mean"),
        ("read_mean", "read_std", "Read latency (s)", "read_mean"),
        ("write_total", None, "Write total (s)", "write_total"),
        ("read_total", None, "Read total (s)", "read_total"),
        ("write_throughput", None, "Write throughput (bytes/s)", "write_throughput"),
        ("read_throughput", None, "Read throughput (bytes/s)", "read_throughput"),
        ("write_cpu_total", None, "CPU time write (s)", "write_cpu"),
        ("read_cpu_total", None, "CPU time read (s)", "read_cpu"),
        ("write_rss_max", None, "Max RSS write (bytes)", "write_rss"),
        ("read_rss_max", None, "Max RSS read (bytes)", "read_rss"),
    ]

    def plot_group(entries: List[Dict[str, Any]], label_prefix: str, fname_prefix: str) -> None:
        for metric, err_key, ylabel, slug in metrics:
            filtered = [e for e in entries if isinstance(e.get(metric), (int, float))]
            if not filtered:
                continue
            labels = [f"{e['container']}-{e['format']}" for e in filtered]
            values = [e[metric] for e in filtered]
            errors = [e.get(err_key, 0.0) if err_key else 0.0 for e in filtered]
            plt.figure(figsize=(max(6, len(values) * 0.7), 4))
            plt.bar(range(len(values)), values, yerr=errors if err_key else None, alpha=0.8, color="steelblue")
            plt.xticks(range(len(values)), labels, rotation=30, ha="right")
            plt.ylabel(ylabel)
            plt.title(f"{label_prefix} – {ylabel}")
            plt.tight_layout()
            out_file = out_root / f"{fname_prefix}_{slug}.png"
            plt.savefig(out_file, dpi=150)
            plt.savefig(out_file.with_suffix(".pdf"))
            plt.close()
            print(f"Wrote plot: {out_file}")

    # Per synthetic dataset (all formats/containers)
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for e in synth:
        by_dataset.setdefault(e["dataset"], []).append(e)
    for ds_name, entries in by_dataset.items():
        plot_group(entries, f"Synth {ds_name}", f"synth_{ds_name}")

    # Per format across synthetic datasets (all containers)
    by_format: Dict[str, List[Dict[str, Any]]] = {}
    for e in synth:
        by_format.setdefault(e["format"], []).append(e)
    for fmt, entries in by_format.items():
        plot_group(entries, f"Synth format {fmt.upper()}", f"synth_format_{fmt}")


def _plot_resource_correlation(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Scatter plots CPU time vs wall time and RSS vs wall time."""
    if not results:
        return
    metrics = [
        ("write", "write_total", "write_cpu_total", "write_rss_max", "Write"),
        ("read", "read_total", "read_cpu_total", "read_rss_max", "Read"),
    ]
    scope = results[0].get("_scope", "all")
    suffix = f"_{scope}" if scope != "all" else ""

    for key, wall_key, cpu_key, rss_key, title in metrics:
        pts = [r for r in results if isinstance(r.get(wall_key), (int, float))]
        cpu_vals = [r.get(cpu_key) for r in pts if isinstance(r.get(cpu_key), (int, float))]
        wall_vals = [r.get(wall_key) for r in pts if isinstance(r.get(cpu_key), (int, float))]
        rss_vals = [r.get(rss_key) for r in pts if isinstance(r.get(rss_key), (int, float))]
        wall_vals_rss = [r.get(wall_key) for r in pts if isinstance(r.get(rss_key), (int, float))]

        has_cpu = cpu_vals and wall_vals and len(cpu_vals) == len(wall_vals)
        has_rss = rss_vals and wall_vals_rss and len(rss_vals) == len(wall_vals_rss)
        if not (has_cpu or has_rss):
            continue

        cols = 1 + int(has_cpu and has_rss)
        plt.figure(figsize=(10, 5))
        subplot_idx = 1
        if has_cpu:
            plt.subplot(1, cols, subplot_idx)
            plt.scatter(wall_vals, cpu_vals, alpha=0.7, c="steelblue")
            plt.xlabel("Wall time (s)")
            plt.ylabel("CPU time (s)")
            plt.title(f"{title}: CPU vs wall")
            plt.grid(True, alpha=0.3)
            subplot_idx += 1
        if has_rss:
            plt.subplot(1, cols, subplot_idx)
            plt.scatter(wall_vals_rss, [v / (1024 * 1024) for v in rss_vals], alpha=0.7, c="orange")
            plt.xlabel("Wall time (s)")
            plt.ylabel("Max RSS (MB)")
            plt.title(f"{title}: RSS vs wall")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        out_file = out_root / f"resource_corr_{key}{suffix}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _plot_failure_summary(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Stacked bar of successes vs errors by container/format."""
    if not results:
        return
    counts: Dict[tuple[str, str], Dict[str, int]] = {}
    for r in results:
        container = r.get("container")
        fmt = r.get("format")
        if container not in {"bids", "sbids"} or not fmt:
            continue
        key = (container, fmt)
        bucket = counts.setdefault(key, {"ok": 0, "fail": 0})
        if "error" in r:
            bucket["fail"] += 1
        else:
            bucket["ok"] += 1
    if not counts:
        return
    keys = sorted(counts.keys())
    ok_vals = [counts[k]["ok"] for k in keys]
    fail_vals = [counts[k]["fail"] for k in keys]
    labels = [f"{c.upper()}-{f}" for c, f in keys]

    plt.figure(figsize=(max(6, len(labels) * 0.7), 4))
    idx = np.arange(len(labels))
    plt.bar(idx, ok_vals, label="Success", color="seagreen")
    plt.bar(idx, fail_vals, bottom=ok_vals, label="Fail/Skip", color="salmon")
    plt.xticks(idx, labels, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Export/load outcomes by container/format")
    plt.legend()
    plt.tight_layout()
    out_file = out_root / "failure_summary.png"
    plt.savefig(out_file, dpi=150)
    plt.savefig(out_file.with_suffix(".pdf"))
    plt.close()
    print(f"Wrote plot: {out_file}")


def _plot_speedup_heatmap(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Heatmap of SBIDS/BIDS speedup ratios (latency and throughput) per dataset/format."""
    if not results:
        return
    pairs: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for r in results:
        key = (r.get("dataset"), r.get("format"))
        container = r.get("container")
        if not key[0] or not key[1] or container not in {"bids", "sbids"}:
            continue
        pairs.setdefault(key, {})[container] = r

    metrics = [
        ("read_mean", "Read latency (s)"),
        ("write_mean", "Write latency (s)"),
        ("read_throughput", "Read throughput (bytes/s)"),
        ("write_throughput", "Write throughput (bytes/s)"),
    ]

    for metric, title in metrics:
        ds_list = sorted({k[0] for k in pairs})
        fmt_list = sorted({k[1] for k in pairs})
        if not ds_list or not fmt_list:
            continue
        matrix = np.full((len(ds_list), len(fmt_list)), np.nan)
        for (ds, fmt), bucket in pairs.items():
            b = bucket.get("bids")
            s = bucket.get("sbids")
            if not b or not s:
                continue
            bval = b.get(metric)
            sval = s.get(metric)
            if not isinstance(bval, (int, float)) or not isinstance(sval, (int, float)) or bval == 0:
                continue
            ratio = sval / bval
            i = ds_list.index(ds)
            j = fmt_list.index(fmt)
            matrix[i, j] = ratio
        if np.all(np.isnan(matrix)):
            continue
        plt.figure(figsize=(1.5 * len(fmt_list) + 2, 0.6 * len(ds_list) + 2))
        cmap = plt.cm.RdYlGn_r
        vmax = np.nanmax(matrix)
        vmin = np.nanmin(matrix)
        im = plt.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="SBIDS / BIDS ratio")
        plt.xticks(range(len(fmt_list)), fmt_list)
        plt.yticks(range(len(ds_list)), ds_list)
        plt.title(f"SBIDS vs BIDS speedup – {title}")
        plt.tight_layout()
        out_file = out_root / f"heatmap_speedup_{metric}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _plot_efficiency_bars(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Bars for bytes per sample per channel and reduction vs EDF baseline."""
    if not results:
        return
    scope = results[0].get("_scope", "all") if results else "all"
    suffix = f"_{scope}" if scope != "all" else ""

    # Bytes per sample per channel (normalized footprint)
    entries = [
        r
        for r in results
        if isinstance(r.get("size_bytes"), (int, float))
        and isinstance(r.get("data_points"), (int, float))
        and isinstance(r.get("channel_count"), (int, float))
    ]
    if entries:
        labels = [f"{r['dataset']}-{r['container']}-{r['format']}" for r in entries]
        vals = [r["size_bytes"] / (r["data_points"] * r["channel_count"]) for r in entries]
        plt.figure(figsize=(max(6, len(vals) * 0.5), 4))
        plt.bar(range(len(vals)), vals, color="steelblue", alpha=0.8)
        plt.xticks(range(len(vals)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel("Bytes per sample per channel")
        plt.title("Disk efficiency (normalized)")
        plt.tight_layout()
        out_file = out_root / f"bytes_per_sample_channel{suffix}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _plot_synth_multistrip(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Small multiples per synthetic dataset covering key metrics across format/container."""
    synth = [
        r
        for r in results
        if r.get("dataset_kind") == "synthetic"
        and r.get("container") in {"bids", "sbids"}
    ]
    if not synth:
        return

    metrics = [
        ("size_bytes", "Size (bytes)"),
        ("write_mean", "Write latency (s)"),
        ("read_mean", "Read latency (s)"),
        ("write_total", "Write total (s)"),
        ("read_total", "Read total (s)"),
        ("write_throughput", "Write throughput (bytes/s)"),
        ("read_throughput", "Read throughput (bytes/s)"),
        ("write_cpu_total", "CPU time write (s)"),
        ("read_cpu_total", "CPU time read (s)"),
        ("write_rss_max", "Max RSS write (bytes)"),
        ("read_rss_max", "Max RSS read (bytes)"),
    ]

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in synth:
        by_dataset.setdefault(r["dataset"], []).append(r)

    for ds, entries in by_dataset.items():
        cols = 3
        rows = int(np.ceil(len(metrics) / cols))
        plt.figure(figsize=(5 * cols, 3 * rows))
        for idx, (metric, label) in enumerate(metrics, start=1):
            plt.subplot(rows, cols, idx)
            vals = [e.get(metric) for e in entries if isinstance(e.get(metric), (int, float))]
            if not vals:
                plt.axis("off")
                continue
            labels = [f"{e['container']}-{e['format']}" for e in entries if isinstance(e.get(metric), (int, float))]
            plt.bar(range(len(vals)), vals, color="steelblue", alpha=0.8)
            plt.xticks(range(len(vals)), labels, rotation=30, ha="right", fontsize=8)
            plt.ylabel(label)
            plt.title(metric)
        plt.tight_layout()
        out_file = out_root / f"synth_multistrip_{ds}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _plot_rw_throughput(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Bars placing read vs write throughput side by side per dataset/container/format."""
    entries = [
        r
        for r in results
        if r.get("container") in {"bids", "sbids"}
        and isinstance(r.get("write_throughput"), (int, float))
        and isinstance(r.get("read_throughput"), (int, float))
    ]
    if not entries:
        return
    scope = results[0].get("_scope", "all") if results else "all"
    suffix = f"_{scope}" if scope != "all" else ""

    labels = [f"{r['dataset']}-{r['container']}-{r['format']}" for r in entries]
    write_vals = [r["write_throughput"] / (1024 * 1024) for r in entries]
    read_vals = [r["read_throughput"] / (1024 * 1024) for r in entries]
    idx = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(max(6, len(labels) * 0.7), 4))
    plt.bar(idx - width / 2, write_vals, width, label="Write", color="steelblue")
    plt.bar(idx + width / 2, read_vals, width, label="Read", color="orange")
    plt.xticks(idx, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Throughput (MB/s)")
    plt.title("Read vs Write throughput")
    plt.legend()
    plt.tight_layout()
    out_file = out_root / f"throughput_rw{suffix}.png"
    plt.savefig(out_file, dpi=150)
    plt.savefig(out_file.with_suffix(".pdf"))
    plt.close()
    print(f"Wrote plot: {out_file}")


def _plot_extension_box(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Box plot of throughput by file extension (collapsed across BIDS/SBIDS)."""
    entries = [
        r
        for r in results
        if r.get("container") in {"bids", "sbids"}
        and isinstance(r.get("read_throughput"), (int, float))
    ]
    if not entries:
        return
    buckets: Dict[str, List[float]] = {}
    for r in entries:
        fmt = r.get("format")
        if not fmt:
            continue
        buckets.setdefault(fmt, []).append(r["read_throughput"] / (1024 * 1024))
    if not buckets:
        return
    labels = list(buckets.keys())
    data = [buckets[k] for k in labels]

    plt.figure(figsize=(max(6, len(labels) * 1.2), 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel("Read throughput (MB/s)")
    plt.title("Experiment 2 – Read throughput by extension")
    plt.tight_layout()
    out_file = out_root / "extension_throughput_box.png"
    plt.savefig(out_file, dpi=150)
    plt.savefig(out_file.with_suffix(".pdf"))
    plt.close()
    print(f"Wrote plot: {out_file}")


def _plot_latency_pairs(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Side-by-side BIDS vs SBIDS bars per dataset/format for timing totals/means."""
    if not results:
        return
    pairs: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for r in results:
        key = (r.get("dataset"), r.get("format"))
        container = r.get("container")
        if not key[0] or not key[1] or container not in {"bids", "sbids"}:
            continue
        pairs.setdefault(key, {})[container] = r

    metrics = [
        ("write_mean", "Write latency (s)", "paired_write_latency"),
        ("read_mean", "Read latency (s)", "paired_read_latency"),
        ("write_total", "Write total (s)", "paired_write_total"),
        ("read_total", "Read total (s)", "paired_read_total"),
        ("write_throughput", "Write throughput (bytes/s)", "paired_write_throughput"),
        ("read_throughput", "Read throughput (bytes/s)", "paired_read_throughput"),
    ]

    for metric, ylabel, slug in metrics:
        labels = []
        bids_vals = []
        sbids_vals = []
        for (ds, fmt), bucket in pairs.items():
            b = bucket.get("bids")
            s = bucket.get("sbids")
            if not b or not s:
                continue
            if not isinstance(b.get(metric), (int, float)) or not isinstance(s.get(metric), (int, float)):
                continue
            labels.append(f"{ds}-{fmt}")
            bids_vals.append(b[metric])
            sbids_vals.append(s[metric])
        if not labels:
            continue
        idx = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(max(6, len(labels) * 0.7), 4))
        plt.bar(idx - width / 2, bids_vals, width, label="BIDS", color="steelblue")
        plt.bar(idx + width / 2, sbids_vals, width, label="SBIDS", color="orange")
        plt.xticks(idx, labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel(ylabel)
        plt.title(f"BIDS vs SBIDS – {ylabel}")
        plt.legend()
        plt.tight_layout()
        scope = results[0].get("_scope", "all")
        suffix = f"_{scope}" if scope != "all" else ""
        out_file = out_root / f"{slug}{suffix}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2 – format storage & latency benchmark")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep per-dataset BIDS/SBIDS exports (synthetic datasets are deleted by default).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(cfg["label"] for cfg in DATASETS),
        help="Comma-separated dataset labels to run (default: all synthetic datasets).",
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
    parser.add_argument(
        "--plot-scope",
        choices=["all", "synthetic", "bin"],
        default="all",
        help="Which dataset kinds to include in plots (does not affect JSON).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override run count per dataset (default uses dataset config / RUNS).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip running conversions; just read results JSON and regenerate plots.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Path to experiment2_results.json to use when plots-only.",
    )
    args = parser.parse_args()
    containers = [c.strip() for c in args.containers.split(",") if c.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    run(
        containers=containers,
        formats=formats,
        datasets=datasets,
        runs_override=args.runs,
        keep_artifacts=args.keep_artifacts,
        purge_outputs=not args.no_purge,
        plot_scope=args.plot_scope,
        plots_only=args.plots_only,
        results_path=args.results_json,
    )
