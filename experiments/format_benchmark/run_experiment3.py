"""Experiment 3 – Streaming Access Benchmark.

Exports synthetic datasets to BIDS or SBIDS in the chosen formats, then measures
streaming-style reads without preloading the full recording. For each exported
file we time:
  - opening the file handle (no preload)
  - reading an entire single channel
  - reading a random 10 s window from one channel
  - reading a random 10 s window from all channels

Wall time, CPU time (if psutil is available), and max RSS across runs are saved
as JSON under `experiments/format_benchmark/results_exp3/`.

Run with defaults (all synthetic datasets, BIDS+SBIDS, all formats):
    PYTHONPATH=. python experiments/format_benchmark/run_experiment3.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

import mne

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
RUNS = 3
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
    for c in candidates:
        if stem in c.stem:
            return c
    return candidates[0]


def _load_dataset(
    cfg: dict[str, Any],
    default_runs: int,
    *,
    runs_override: int | None = None,
) -> Tuple[CortiDataset, Dict[str, Any], int, str]:
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
    runs = int(runs_override if runs_override is not None else cfg.get("runs", default_runs))
    task = _sanitize_task_name(cfg.get("task", label))
    return ds, stats, runs, task


@dataclass
class StreamHandle:
    fmt: str
    n_samples: int
    channel_count: int
    sampling_rate: float
    raw: Any | None = None
    parquet: Any | None = None
    columns: list[str] | None = None
    dataset: Any | None = None
    resources: dict[str, Any] = field(default_factory=dict)


def _open_stream_handle(data_path: Path, fmt: str, stats: Dict[str, Any]) -> StreamHandle:
    fmt_lower = fmt.lower()
    channel_count = int(stats.get("channel_count") or 0)
    sampling_rate = float(stats.get("sampling_rate") or 0.0)
    duration_s = float(stats.get("duration_s") or 0.0)
    n_samples = int(duration_s * sampling_rate) if sampling_rate and duration_s else 0

    if fmt_lower == "edf":
        raw = mne.io.read_raw_edf(data_path, preload=False, verbose="ERROR")
        channel_count = len(raw.ch_names)
        sampling_rate = float(raw.info.get("sfreq", sampling_rate))
        n_samples = raw.n_times
        return StreamHandle(
            fmt=fmt_lower,
            n_samples=n_samples,
            channel_count=channel_count,
            sampling_rate=sampling_rate,
            raw=raw,
            columns=list(raw.ch_names),
        )

    if fmt_lower == "parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("Reading parquet requires the optional dependency 'pyarrow'.") from exc
        pf = pq.ParquetFile(data_path)
        if pf.metadata and pf.metadata.num_rows is not None:
            n_samples = int(pf.metadata.num_rows)
        columns = list(pf.schema.names)
        if not channel_count:
            channel_count = len(columns)
        return StreamHandle(
            fmt=fmt_lower,
            n_samples=n_samples,
            channel_count=channel_count,
            sampling_rate=sampling_rate,
            parquet=pf,
            columns=columns,
        )

    if fmt_lower in {"hdf5", "h5"}:
        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("Reading HDF5 requires the optional dependency 'h5py'.") from exc
        h5file = h5py.File(data_path, "r")
        dataset = h5file["data"]
        n_samples = dataset.shape[0]
        channel_count = dataset.shape[1] if dataset.ndim > 1 else 1
        sfreq_attr = dataset.attrs.get("sfreq")
        if sampling_rate <= 0 and sfreq_attr is not None:
            try:
                sampling_rate = float(sfreq_attr)
            except (TypeError, ValueError):
                sampling_rate = sampling_rate
        return StreamHandle(
            fmt=fmt_lower,
            n_samples=int(n_samples),
            channel_count=int(channel_count),
            sampling_rate=float(sampling_rate),
            dataset=dataset,
            resources={"h5": h5file},
        )

    if fmt_lower == "zarr":
        try:
            import zarr  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("Reading zarr requires the optional dependency 'zarr'.") from exc
        store = zarr.open(str(data_path), mode="r")
        dataset = store["data"] if "data" in store else store
        n_samples = dataset.shape[0]
        channel_count = dataset.shape[1] if dataset.ndim > 1 else 1
        sfreq_attr = getattr(dataset, "attrs", {}).get("sfreq") if hasattr(dataset, "attrs") else None
        if sampling_rate <= 0 and sfreq_attr is not None:
            try:
                sampling_rate = float(sfreq_attr)
            except (TypeError, ValueError):
                sampling_rate = sampling_rate
        return StreamHandle(
            fmt=fmt_lower,
            n_samples=int(n_samples),
            channel_count=int(channel_count),
            sampling_rate=float(sampling_rate),
            dataset=dataset,
            resources={"zarr_store": store},
        )

    raise ValueError(f"Unsupported format for streaming: {fmt}")


def _close_handle(handle: StreamHandle) -> None:
    if handle.raw is not None and hasattr(handle.raw, "close"):
        try:
            handle.raw.close()
        except Exception:
            pass
    h5file = handle.resources.get("h5") if handle.resources else None
    if h5file is not None:
        try:
            h5file.close()
        except Exception:
            pass


def _parquet_slice(pf, columns: list[str], start: int, stop: int) -> np.ndarray:
    import pyarrow as pa  # type: ignore

    if start < 0:
        start = 0
    if stop <= start:
        return np.empty((0, len(columns)))

    tables = []
    offset = 0
    for idx in range(pf.num_row_groups):
        rg = pf.metadata.row_group(idx)
        rg_rows = rg.num_rows
        if offset + rg_rows <= start:
            offset += rg_rows
            continue
        if offset >= stop:
            break
        slice_start = max(0, start - offset)
        slice_len = min(rg_rows - slice_start, stop - offset - slice_start)
        table = pf.read_row_group(idx, columns=columns)
        if slice_start or slice_len != rg_rows:
            table = table.slice(slice_start, slice_len)
        tables.append(table)
        offset += rg_rows
        if offset >= stop:
            break
    if not tables:
        return np.empty((0, len(columns)))
    combined = pa.concat_tables(tables)
    return combined.to_pandas().to_numpy()


def _read_channel_data(handle: StreamHandle, channel_idx: int, start: int | None = None, stop: int | None = None) -> np.ndarray:
    fmt = handle.fmt
    start_idx = 0 if start is None else int(start)
    stop_idx = handle.n_samples if stop is None else int(stop)
    if stop_idx < start_idx:
        stop_idx = start_idx
    if fmt == "edf":
        if handle.raw is None:
            return np.array([])
        data = handle.raw.get_data(picks=[channel_idx], start=start_idx, stop=stop_idx)
        return np.asarray(data)
    if fmt == "parquet":
        if handle.parquet is None or not handle.columns:
            return np.array([])
        col_name = handle.columns[channel_idx]
        if start is None and stop is None:
            table = handle.parquet.read(columns=[col_name])
            arr = table.column(0).to_numpy()
            return np.asarray(arr)
        sliced = _parquet_slice(handle.parquet, [col_name], start_idx, stop_idx)
        return np.asarray(sliced).reshape(-1, 1) if sliced.size else np.array([])
    if fmt in {"hdf5", "h5", "zarr"}:
        if handle.dataset is None:
            return np.array([])
        data = handle.dataset[start_idx:stop_idx, channel_idx]
        return np.asarray(data)
    raise ValueError(f"Unsupported format for channel read: {fmt}")


def _read_window_all_channels(handle: StreamHandle, start: int, stop: int) -> np.ndarray:
    fmt = handle.fmt
    if fmt == "edf":
        if handle.raw is None:
            return np.array([])
        return np.asarray(handle.raw.get_data(start=start, stop=stop))
    if fmt == "parquet":
        if handle.parquet is None or not handle.columns:
            return np.array([])
        return _parquet_slice(handle.parquet, handle.columns, start, stop)
    if fmt in {"hdf5", "h5", "zarr"}:
        if handle.dataset is None:
            return np.array([])
        return np.asarray(handle.dataset[start:stop, :])
    raise ValueError(f"Unsupported format for window read: {fmt}")


def _metric_bucket() -> Dict[str, List[float]]:
    return {"times": [], "cpu": [], "rss": []}


def _push_metric(bucket: Dict[str, List[float]], dt: float, cpu_dt: float | None, rss_peak: float | None) -> None:
    bucket["times"].append(dt)
    if cpu_dt is not None:
        bucket["cpu"].append(cpu_dt)
    if rss_peak is not None:
        bucket["rss"].append(rss_peak)


def _summarize_metrics(bucket: Dict[str, List[float]]) -> Dict[str, Any]:
    times = bucket.get("times", [])
    cpu = bucket.get("cpu", [])
    rss = bucket.get("rss", [])
    mean, std = _mean_std(times)
    cpu_mean, cpu_std = _mean_std(cpu) if cpu else (None, None)
    return {
        "times": times,
        "mean": mean,
        "std": std,
        "total": sum(times),
        "cpu_times": cpu,
        "cpu_mean": cpu_mean,
        "cpu_std": cpu_std,
        "cpu_total": sum(cpu) if cpu else None,
        "rss_list": rss,
        "rss_max": max(rss) if rss else None,
    }


def _benchmark_streaming(
    data_path: Path,
    fmt: str,
    stats: Dict[str, Any],
    *,
    runs: int,
    window_s: float,
    channel_idx: int = 0,
) -> Dict[str, Any]:
    open_metrics = _metric_bucket()
    ch_full = _metric_bucket()
    window_single = _metric_bucket()
    window_all = _metric_bucket()
    errors: List[str] = []

    for _ in range(runs):
        try:
            handle, dt_open, cpu_open, rss_open = _measure(_open_stream_handle, data_path, fmt, stats)
        except Exception as exc:
            errors.append(str(exc))
            break
        _push_metric(open_metrics, dt_open, cpu_open, rss_open)

        window_samples = None
        if handle.sampling_rate > 0 and handle.n_samples > 0 and window_s > 0:
            window_samples = max(1, int(handle.sampling_rate * window_s))
        if window_samples is None:
            start = 0
            stop = handle.n_samples
        else:
            max_start = max(0, handle.n_samples - window_samples)
            start = random.randint(0, max_start) if max_start > 0 else 0
            stop = min(handle.n_samples, start + window_samples)

        try:
            _, dt, cpu_dt, rss_peak = _measure(_read_channel_data, handle, channel_idx, None, None)
            _push_metric(ch_full, dt, cpu_dt, rss_peak)

            _, dt, cpu_dt, rss_peak = _measure(_read_channel_data, handle, channel_idx, start, stop)
            _push_metric(window_single, dt, cpu_dt, rss_peak)

            _, dt, cpu_dt, rss_peak = _measure(_read_window_all_channels, handle, start, stop)
            _push_metric(window_all, dt, cpu_dt, rss_peak)
        except Exception as exc:
            errors.append(str(exc))
        finally:
            _close_handle(handle)

    result: Dict[str, Any] = {
        "open": _summarize_metrics(open_metrics),
        "single_channel_full": _summarize_metrics(ch_full),
        "window_single_channel": _summarize_metrics(window_single),
        "window_all_channels": _summarize_metrics(window_all),
        "window_s": window_s,
        "channel_idx": channel_idx,
    }
    if errors:
        result["errors"] = errors
    return result


def _stream_entries(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in results if isinstance(r.get("stream_metrics"), dict)]


def _get_metric_mean(entry: Dict[str, Any], metric: str) -> float | None:
    sm = entry.get("stream_metrics", {})
    m = sm.get(metric, {})
    val = m.get("mean")
    return float(val) if isinstance(val, (int, float)) else None


def _get_metric_values(entry: Dict[str, Any], metric: str, key: str) -> List[float]:
    sm = entry.get("stream_metrics", {})
    m = sm.get(metric, {})
    vals = m.get(key, [])
    return [float(v) for v in vals if isinstance(v, (int, float))]


def _plot_stream_latency_bars(results: List[Dict[str, Any]], out_root: Path) -> None:
    entries = _stream_entries(results)
    if not entries:
        return
    metrics = [
        ("open", "Open handle (s)"),
        ("single_channel_full", "Read full single channel (s)"),
        ("window_single_channel", "Read random window single channel (s)"),
        ("window_all_channels", "Read random window all channels (s)"),
    ]
    for metric, ylabel in metrics:
        labels = []
        values = []
        for e in entries:
            val = _get_metric_mean(e, metric)
            if val is None:
                continue
            labels.append(f"{e['dataset']}-{e['container']}-{e['format']}")
            values.append(val)
        if not labels:
            continue
        plt.figure(figsize=(max(6, len(values) * 0.6), 4))
        plt.bar(range(len(values)), values, color="steelblue", alpha=0.8)
        plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel(ylabel)
        plt.title(f"Streaming latency – {ylabel}")
        plt.tight_layout()
        out_file = out_root / f"stream_bar_{metric}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _plot_stream_latency_boxplots(results: List[Dict[str, Any]], out_root: Path) -> None:
    entries = _stream_entries(results)
    if not entries:
        return
    metrics = [
        ("open", "Open handle (s)"),
        ("single_channel_full", "Read full single channel (s)"),
        ("window_single_channel", "Read random window single channel (s)"),
        ("window_all_channels", "Read random window all channels (s)"),
    ]
    for metric, ylabel in metrics:
        buckets: Dict[tuple[str, str], List[float]] = {}
        for e in entries:
            vals = _get_metric_values(e, metric, "times")
            if not vals:
                continue
            key = (e.get("container"), e.get("format"))
            if not key[0] or not key[1]:
                continue
            buckets.setdefault(key, []).extend(vals)
        if not buckets:
            continue
        labels = [f"{c.upper()}-{f}" for (c, f) in buckets]
        data = [buckets[k] for k in buckets]
        plt.figure(figsize=(max(6, len(data) * 1.2), 5))
        plt.boxplot(data, tick_labels=labels, showfliers=False)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Streaming latency – {ylabel} (boxplots)")
        plt.tight_layout()
        out_file = out_root / f"stream_box_{metric}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _bytes_per_sample(entry: Dict[str, Any]) -> float | None:
    bps = entry.get("bytes_per_sample")
    if isinstance(bps, (int, float)) and bps > 0:
        return float(bps)
    approx = entry.get("approx_raw_bytes")
    data_points = entry.get("data_points")
    if isinstance(approx, (int, float)) and isinstance(data_points, (int, float)) and data_points > 0:
        return float(approx) / float(data_points)
    return None


def _raw_sample_count(entry: Dict[str, Any]) -> int | None:
    data_points = entry.get("data_points")
    channel_count = entry.get("channel_count")
    if isinstance(data_points, (int, float)) and isinstance(channel_count, (int, float)) and channel_count > 0:
        return int(data_points / float(channel_count))
    sr = entry.get("sampling_rate")
    duration = entry.get("duration_s")
    if isinstance(sr, (int, float)) and isinstance(duration, (int, float)) and sr > 0 and duration > 0:
        return int(sr * duration)
    return None


def _window_sample_count(entry: Dict[str, Any]) -> int | None:
    sr = entry.get("sampling_rate")
    window_s = entry.get("window_s")
    if isinstance(sr, (int, float)) and isinstance(window_s, (int, float)) and sr > 0 and window_s > 0:
        return max(1, int(sr * window_s))
    return None


def _estimate_bytes_for_metric(entry: Dict[str, Any], metric: str) -> float | None:
    bps = _bytes_per_sample(entry)
    if bps is None or bps <= 0:
        return None
    n_full = _raw_sample_count(entry)
    n_window = _window_sample_count(entry)
    ch = entry.get("channel_count")
    if metric == "single_channel_full" and n_full is not None:
        return bps * n_full
    if metric == "window_single_channel" and n_window is not None:
        return bps * n_window
    if metric == "window_all_channels" and n_window is not None and isinstance(ch, (int, float)):
        return bps * n_window * float(ch)
    return None


def _plot_stream_throughput_bars(results: List[Dict[str, Any]], out_root: Path) -> None:
    entries = _stream_entries(results)
    if not entries:
        return
    metrics = [
        ("single_channel_full", "Throughput full single channel (MB/s)"),
        ("window_single_channel", "Throughput window single channel (MB/s)"),
        ("window_all_channels", "Throughput window all channels (MB/s)"),
    ]
    for metric, ylabel in metrics:
        labels = []
        values = []
        for e in entries:
            mean = _get_metric_mean(e, metric)
            if mean is None or mean <= 0:
                continue
            bytes_est = _estimate_bytes_for_metric(e, metric)
            if bytes_est is None or bytes_est <= 0:
                continue
            thr = bytes_est / mean / (1024 * 1024)
            labels.append(f"{e['dataset']}-{e['container']}-{e['format']}")
            values.append(thr)
        if not labels:
            continue
        plt.figure(figsize=(max(6, len(values) * 0.6), 4))
        plt.bar(range(len(values)), values, color="seagreen", alpha=0.8)
        plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel(ylabel)
        plt.title(f"Streaming throughput – {ylabel}")
        plt.tight_layout()
        out_file = out_root / f"stream_throughput_{metric}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _plot_stream_differences(results: List[Dict[str, Any]], out_root: Path) -> None:
    entries = _stream_entries(results)
    if not entries:
        return
    pairs: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for r in entries:
        key = (r.get("dataset"), r.get("format"))
        container = r.get("container")
        if not key[0] or not key[1] or container not in {"bids", "sbids"}:
            continue
        pairs.setdefault(key, {})[container] = r

    metrics = [
        ("open", "Open handle (s)"),
        ("single_channel_full", "Read full single channel (s)"),
        ("window_single_channel", "Read random window single channel (s)"),
        ("window_all_channels", "Read random window all channels (s)"),
    ]

    for metric, ylabel in metrics:
        vals = []
        labels = []
        for (ds, fmt), bucket in pairs.items():
            b = bucket.get("bids")
            s = bucket.get("sbids")
            if not b or not s:
                continue
            bval = _get_metric_mean(b, metric)
            sval = _get_metric_mean(s, metric)
            if bval is None or sval is None:
                continue
            vals.append(sval - bval)
            labels.append(f"{ds}-{fmt}")
        if not vals:
            continue
        colors = ["green" if v < 0 else "red" for v in vals]
        plt.figure(figsize=(max(6, len(vals) * 0.6), 4))
        plt.bar(range(len(vals)), vals, color=colors, alpha=0.85)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(range(len(vals)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel("SBIDS minus BIDS (s)")
        plt.title(f"Streaming latency diff – {ylabel}")
        plt.tight_layout()
        out_file = out_root / f"stream_diff_{metric}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _plot_stream_resource_boxplots(results: List[Dict[str, Any]], out_root: Path) -> None:
    entries = _stream_entries(results)
    if not entries:
        return
    metrics = ["open", "single_channel_full", "window_single_channel", "window_all_channels"]
    resource_specs = [
        ("cpu_times", "CPU time (s)", "cpu", 1.0),
        ("rss_list", "Max RSS (MB)", "rss", 1 / (1024 * 1024)),
    ]
    for metric in metrics:
        for key, ylabel, slug, factor in resource_specs:
            buckets: Dict[tuple[str, str], List[float]] = {}
            for e in entries:
                vals = _get_metric_values(e, metric, key)
                if not vals:
                    continue
                scaled = [v * factor for v in vals]
                pair = (e.get("container"), e.get("format"))
                if not pair[0] or not pair[1]:
                    continue
                buckets.setdefault(pair, []).extend(scaled)
            if not buckets:
                continue
            labels = [f"{c.upper()}-{f}" for (c, f) in buckets]
            data = [buckets[k] for k in buckets]
            plt.figure(figsize=(max(6, len(data) * 1.2), 5))
            plt.boxplot(data, tick_labels=labels, showfliers=False)
            plt.xticks(rotation=30, ha="right")
            plt.ylabel(ylabel)
            plt.title(f"Streaming resources – {metric} {ylabel}")
            plt.tight_layout()
            out_file = out_root / f"stream_resource_{metric}_{slug}.png"
            plt.savefig(out_file, dpi=150)
            plt.savefig(out_file.with_suffix(".pdf"))
            plt.close()
            print(f"Wrote plot: {out_file}")


def _plot_export_latency_boxplots(results: List[Dict[str, Any]], out_root: Path) -> None:
    entries = [r for r in results if isinstance(r.get("export_time"), (int, float))]
    if not entries:
        return
    buckets: Dict[tuple[str, str], List[float]] = {}
    for e in entries:
        key = (e.get("container"), e.get("format"))
        if not key[0] or not key[1]:
            continue
        buckets.setdefault(key, []).append(float(e["export_time"]))
    if not buckets:
        return
    labels = [f"{c.upper()}-{f}" for (c, f) in buckets]
    data = [buckets[k] for k in buckets]
    plt.figure(figsize=(max(6, len(data) * 1.2), 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Export latency (s)")
    plt.title("Write latency (export) by container/format")
    plt.tight_layout()
    out_file = out_root / "stream_export_box.png"
    plt.savefig(out_file, dpi=150)
    plt.savefig(out_file.with_suffix(".pdf"))
    plt.close()
    print(f"Wrote plot: {out_file}")


def _plot_read_write_pairs(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Side-by-side bars for export (write) vs streaming read latency per dataset/format for key read metrics."""
    entries = _stream_entries(results)
    if not entries:
        return
    metrics = [
        ("single_channel_full", "Read full single channel (s)"),
        ("window_single_channel", "Read window single channel (s)"),
        ("window_all_channels", "Read window all channels (s)"),
    ]
    for metric, title in metrics:
        labels = []
        read_vals = []
        write_vals = []
        for e in entries:
            read_mean = _get_metric_mean(e, metric)
            write_mean = e.get("export_time")
            if not isinstance(write_mean, (int, float)) or read_mean is None:
                continue
            labels.append(f"{e['dataset']}-{e['container']}-{e['format']}")
            read_vals.append(read_mean)
            write_vals.append(write_mean)
        if not labels:
            continue
        idx = np.arange(len(labels))
        width = 0.4
        plt.figure(figsize=(max(6, len(labels) * 0.8), 5))
        plt.bar(idx - width / 2, read_vals, width, label="Read (stream)", color="steelblue")
        plt.bar(idx + width / 2, write_vals, width, label="Write (export)", color="orange")
        plt.xticks(idx, labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel("Latency (s)")
        plt.title(f"Read vs write latency – {title}")
        plt.legend()
        plt.tight_layout()
        out_file = out_root / f"stream_read_write_pairs_{metric}.png"
        plt.savefig(out_file, dpi=150)
        plt.savefig(out_file.with_suffix(".pdf"))
        plt.close()
        print(f"Wrote plot: {out_file}")


def _plot_read_write_boxpairs(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Grouped boxplots placing read and write side-by-side per container/format."""
    entries = _stream_entries(results)
    if not entries:
        return
    metrics = [
        ("single_channel_full", "Read full single channel (s)"),
        ("window_single_channel", "Read window single channel (s)"),
        ("window_all_channels", "Read window all channels (s)"),
    ]
    for metric, ylabel in metrics:
        buckets_read: Dict[tuple[str, str], List[float]] = {}
        buckets_write: Dict[tuple[str, str], List[float]] = {}
        for e in entries:
            key = (e.get("container"), e.get("format"))
            if not key[0] or not key[1]:
                continue
            read_vals = _get_metric_values(e, metric, "times")
            write_val = e.get("export_time")
            if read_vals:
                buckets_read.setdefault(key, []).extend(read_vals)
            if isinstance(write_val, (int, float)):
                buckets_write.setdefault(key, []).append(float(write_val))
        if not buckets_read and not buckets_write:
            continue
        keys = sorted(set(buckets_read.keys()) | set(buckets_write.keys()))
        if not keys:
            continue
        positions = []
        data = []
        labels = []
        colors = []
        for idx, key in enumerate(keys):
            base = idx * 3
            r_vals = buckets_read.get(key, [])
            w_vals = buckets_write.get(key, [])
            if r_vals:
                positions.append(base + 1)
                data.append(r_vals)
                labels.append(f"{key[0].upper()}-{key[1]} read")
                colors.append("steelblue")
            if w_vals:
                positions.append(base + 2)
                data.append(w_vals)
                labels.append(f"{key[0].upper()}-{key[1]} write")
                colors.append("orange")
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(keys) * 2.5), 5))
        bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_xticks([idx * 3 + 1.5 for idx in range(len(keys))])
        ax.set_xticklabels([f"{k[0].upper()}-{k[1]}" for k in keys], rotation=30, ha="right")
        ax.set_ylabel("Latency (s)")
        ax.set_title(f"Read vs write latency (box) – {ylabel}")
        handles = [
            plt.Line2D([0], [0], color="steelblue", lw=4, label="Read"),
            plt.Line2D([0], [0], color="orange", lw=4, label="Write"),
        ]
        ax.legend(handles=handles, loc="best")
        fig.tight_layout()
        out_file = out_root / f"stream_read_write_box_{metric}.png"
        fig.savefig(out_file, dpi=150)
        fig.savefig(out_file.with_suffix(".pdf"))
        plt.close(fig)
        print(f"Wrote plot: {out_file}")


def _plot_subset_read_box_bar(results: List[Dict[str, Any]], out_root: Path) -> None:
    """Custom comparison for Exp 3: SBIDS+parquet, SBIDS+edf, BIDS+edf read latency (single channel full)."""
    entries = _stream_entries(results)
    if not entries:
        return
    combos = [("sbids", "parquet"), ("sbids", "edf"), ("bids", "edf")]
    labels = [f"{c.upper()}+{f}" for c, f in combos]
    data = []
    means = []
    valid_labels = []
    for container, fmt in combos:
        vals: list[float] = []
        mean_val = None
        for e in entries:
            if e.get("container") == container and e.get("format") == fmt:
                vals = _get_metric_values(e, "single_channel_full", "times")
                mean_val = _get_metric_mean(e, "single_channel_full")
                break
        if vals:
            data.append(vals)
            means.append(mean_val if mean_val is not None else float(np.mean(vals)))
            valid_labels.append(f"{container.upper()}+{fmt}")
    if not data:
        return
    # Boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, tick_labels=valid_labels, showfliers=False)
    plt.ylabel("Read latency (s)")
    plt.title("Streaming latency – full single channel")
    plt.tight_layout()
    out_box = out_root / "stream_subset_box_single_channel_full.png"
    plt.savefig(out_box, dpi=150)
    plt.savefig(out_box.with_suffix(".pdf"))
    plt.close()
    print(f"Wrote plot: {out_box}")
    # Bar plot of means
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(valid_labels)), means, color="steelblue", alpha=0.85)
    plt.xticks(range(len(valid_labels)), valid_labels, rotation=20)
    plt.ylabel("Read latency (s)")
    plt.title("Streaming latency – full single channel (means)")
    plt.tight_layout()
    out_bar = out_root / "stream_subset_bar_single_channel_full.png"
    plt.savefig(out_bar, dpi=150)
    plt.savefig(out_bar.with_suffix(".pdf"))
    plt.close()
    print(f"Wrote plot: {out_bar}")


def _emit_plots(results: List[Dict[str, Any]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    _plot_stream_latency_bars(results, out_root)
    _plot_stream_latency_boxplots(results, out_root)
    _plot_stream_throughput_bars(results, out_root)
    _plot_stream_differences(results, out_root)
    _plot_stream_resource_boxplots(results, out_root)
    _plot_export_latency_boxplots(results, out_root)
    _plot_read_write_pairs(results, out_root)
    _plot_read_write_boxpairs(results, out_root)
    _plot_subset_read_box_bar(results, out_root)


def run(
    *,
    containers: Sequence[str] = ALLOWED_CONTAINERS,
    formats: Sequence[str] = ALLOWED_FORMATS,
    datasets: Sequence[str] | None = None,
    runs_override: int | None = None,
    keep_artifacts: bool = False,
    purge_outputs: bool = True,
    window_s: float = 10.0,
    channel_idx: int = 0,
) -> None:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Not setting position of .* misc channel.*", category=RuntimeWarning)

    out_root = Path(__file__).resolve().parent / "results_exp3"
    out_root.mkdir(parents=True, exist_ok=True)

    containers = [c.lower() for c in containers]
    formats = [f.lower() for f in formats]
    for c in containers:
        if c not in ALLOWED_CONTAINERS:
            raise ValueError(f"container must be one of {ALLOWED_CONTAINERS}, got {c}")
    for f in formats:
        if f not in ALLOWED_FORMATS:
            raise ValueError(f"fmt must be one of {ALLOWED_FORMATS}, got {f}")

    default_runs = runs_override if runs_override is not None else RUNS
    selected = DATASETS if datasets is None else [cfg for cfg in DATASETS if cfg["label"] in datasets]
    missing = set(datasets or []) - {cfg["label"] for cfg in DATASETS}
    if missing:
        print(f"[WARN] Unknown dataset labels requested: {', '.join(sorted(missing))}")

    results: List[Dict[str, Any]] = []

    for cfg in selected:
        label = cfg["label"]
        ds, stats, runs, task = _load_dataset(cfg, default_runs=default_runs, runs_override=runs_override)
        print(f"Processing {label} ({stats['dataset_kind']}) for streaming benchmark")

        dataset_dir = out_root / label
        if purge_outputs and dataset_dir.exists() and not keep_artifacts:
            shutil.rmtree(dataset_dir, ignore_errors=True)

        for fmt in formats:
            for container in containers:
                data_path: Path | None = None
                meta_path: Path | None = None
                export_time = None
                export_cpu = None
                export_rss = None

                if container == "bids":
                    bids_root = dataset_dir / "bids" / fmt
                    bids_root.mkdir(parents=True, exist_ok=True)
                    try:
                        data_path, export_time, export_cpu, export_rss = _measure(
                            ds.to_bids,
                            bids_root,
                            subject="01",
                            task=task,
                            format=fmt,
                            overwrite=True,
                        )
                        meta_path = data_path.with_suffix(".json") if isinstance(data_path, Path) else None
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
                elif container == "sbids":
                    sbids_dir = dataset_dir / "sbids"
                    sbids_dir.mkdir(parents=True, exist_ok=True)
                    meta_path = sbids_dir / f"{label}_{fmt}.jsonld"
                    try:
                        _, export_time, export_cpu, export_rss = _measure(
                            ds.to_sbids, meta_path, export_format=fmt
                        )
                        data_path = _find_sbids_raw(sbids_dir, fmt, meta_path.stem)
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
                else:
                    _log_skip(f"Unsupported container {container}")
                    continue

                if data_path is None or not Path(data_path).exists():
                    results.append(
                        {
                            "dataset": label,
                            "container": container,
                            "format": fmt,
                            "error": f"Data path not found for {container} {fmt}",
                        }
                    )
                    continue

                size = _size_bytes(Path(data_path))
                stream_metrics = _benchmark_streaming(
                    Path(data_path),
                    fmt,
                    stats,
                    runs=runs,
                    window_s=window_s,
                    channel_idx=channel_idx,
                )

                results.append(
                    {
                        "dataset": label,
                        "container": container,
                        "format": fmt,
                        "task": task,
                        "runs": runs,
                        "export_time": export_time,
                        "export_cpu": export_cpu,
                        "export_rss": export_rss,
                        "size_bytes": size,
                        "stream_metrics": stream_metrics,
                        "data_path": str(data_path),
                        "meta_path": str(meta_path) if meta_path else None,
                        "window_s": window_s,
                        "channel_idx": channel_idx,
                        **stats,
                    }
                )

                if not keep_artifacts:
                    shutil.rmtree(dataset_dir, ignore_errors=True)

        del ds

    out_path = out_root / "experiment3_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"Wrote streaming benchmark results to {out_path}")
    _emit_plots(results, out_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3 – streaming access benchmark")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep per-dataset exports (deleted by default to save space).",
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
        "--runs",
        type=int,
        default=None,
        help="Override run count per dataset (default uses dataset config / RUNS).",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=10.0,
        help="Window duration in seconds for random-window streaming tests.",
    )
    parser.add_argument(
        "--channel-idx",
        type=int,
        default=0,
        help="Channel index to probe for single-channel benchmarks (0-based).",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Path to an existing experiment3_results.json to plot without rerunning benchmarks.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip benchmarking and only emit plots from an existing results JSON (default location if not provided).",
    )
    args = parser.parse_args()
    # Reuse existing results without rerunning benchmarks
    if args.results_json or args.plots_only:
        if args.results_json:
            res_path = Path(args.results_json).expanduser().resolve()
        else:
            res_path = Path(__file__).resolve().parent / "results_exp3" / "experiment3_results.json"
        if not res_path.exists():
            raise FileNotFoundError(f"results JSON not found at {res_path}. Provide --results-json or rerun experiment.")
        results_loaded = json.loads(res_path.read_text())
        out_dir = res_path.parent
        print(f"[PLOTS ONLY] Using existing results from {res_path}")
        _emit_plots(results_loaded, out_dir)
    else:
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
            window_s=args.window_s,
            channel_idx=args.channel_idx,
        )
