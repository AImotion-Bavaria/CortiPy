"""BIDS loading performance experiment.

Measures wall-clock time, CPU time/peaks, and memory usage while loading a BIDS
recording via `cortipy.shared.BIDSLoader`. Results can be printed or written to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cortipy.shared import BIDSLoader  # noqa: E402

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency guard
    HAS_MPL = False

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


@dataclass
class ResourceStats:
    wall_time_s: float
    cpu_time_s: float
    cpu_util_avg_pct: Optional[float]
    cpu_util_peak_pct: Optional[float]
    rss_peak_mb: Optional[float]
    rss_delta_mb: Optional[float]


class _ResourceSampler:
    """Background sampler for CPU% and RSS, using psutil when available."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._running = False
        self._rss_peaks: list[float] = []
        self._cpu_samples: list[float] = []
        self._proc = psutil.Process(os.getpid()) if psutil else None

    def start(self) -> None:
        if not self._proc:
            return
        self._running = True
        self._proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[Optional[float], Optional[float]]:
        if not self._proc:
            return None, None
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval * 2)
        avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else None
        peak_rss = max(self._rss_peaks) if self._rss_peaks else None
        return avg_cpu, peak_rss

    def _loop(self) -> None:
        assert self._proc is not None
        while self._running:
            try:
                self._cpu_samples.append(self._proc.cpu_percent(interval=None))
                self._rss_peaks.append(self._proc.memory_info().rss / (1024 * 1024))
            except Exception:
                break
            time.sleep(self.interval)


def _mb(value: float) -> float:
    return value / (1024 * 1024)


def benchmark_bids_load(
    bids_root: Path,
    *,
    allowed_file_structures: Sequence[str] | None = None,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    modality_dirs: Sequence[str] = ("eeg", "ieeg"),
    preload: bool = True,
    sample_interval: float = 0.05,
    all_recordings: bool = False,
) -> dict[str, Any]:
    loader = BIDSLoader(bids_root)
    sampler = _ResourceSampler(interval=sample_interval)

    allowed = tuple(allowed_file_structures) if allowed_file_structures else (
        ".edf",
        ".bdf",
        ".vhdr",
        ".set",
        ".fif",
        ".eeg",
        ".parquet",
        ".h5",
        ".hdf5",
        ".zarr",
    )

    if all_recordings:
        recordings = loader._collect_recordings(
            allowed, modality_dirs, subject=subject, session=session, task=task, run=run
        )
        recordings = _dedupe_recordings(recordings)
        if not recordings:
            raise FileNotFoundError("No BIDS recordings found for the provided filters.")

        per_recording: list[dict[str, Any]] = []
        for rec in recordings:
            per_recording.append(
                _measure_single_recording(
                    loader,
                    rec,
                    preload=preload,
                    sample_interval=sample_interval,
                )
            )
        summary = _aggregate_results(per_recording)
        output = {
            "recordings": per_recording,
            "summary": summary,
            "config": {
                "bids_root": str(bids_root),
                "allowed_file_structures": list(allowed),
                "subject": subject,
                "session": session,
                "task": task,
                "run": run,
                "preload": preload,
                "sample_interval": sample_interval,
                "psutil_available": psutil is not None,
                "mode": "all_recordings",
            },
        }
        return output

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    sampler.start()
    result = loader.read_bids(
        allowed_file_structures=allowed,
        subject=subject,
        session=session,
        task=task,
        run=run,
        modality_dirs=modality_dirs,
        preload=preload,
    )
    avg_cpu, peak_rss = sampler.stop()
    cpu_end = time.process_time()
    wall_end = time.perf_counter()

    wall_time_s = wall_end - wall_start
    cpu_time_s = cpu_end - cpu_start
    rss_start = None
    rss_end = None
    rss_delta_mb = None
    if psutil:
        proc = psutil.Process(os.getpid())
        try:
            rss_end = proc.memory_info().rss
            rss_start = rss_end  # best-effort; sampler gives the peaks
            if peak_rss is not None:
                rss_delta_mb = peak_rss - (rss_start / (1024 * 1024))
        except Exception:
            rss_end = None

    stats = ResourceStats(
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        cpu_util_avg_pct=avg_cpu,
        cpu_util_peak_pct=max(sampler._cpu_samples) if sampler._cpu_samples else None,
        rss_peak_mb=peak_rss,
        rss_delta_mb=rss_delta_mb,
    )

    data_bytes = int(result.data.nbytes)
    samples, channels = result.data.shape
    output = {
        "metrics": asdict(stats),
        "dataset": {
            "samples": int(samples),
            "channels": int(channels),
            "sampling_rate": float(result.sampling_rate),
            "data_bytes": data_bytes,
            "data_mb": data_bytes / (1024 * 1024),
            "events": None if result.events is None else len(result.events),
            "channels_meta": result.channels is not None,
            "source_path": str(result.source_path),
        },
        "config": {
            "bids_root": str(bids_root),
            "allowed_file_structures": list(allowed),
            "subject": subject,
            "session": session,
            "task": task,
            "run": run,
            "preload": preload,
            "sample_interval": sample_interval,
            "psutil_available": psutil is not None,
            "mode": "single",
        },
    }
    return output


def _measure_single_recording(
    loader: BIDSLoader,
    recording: Path,
    *,
    preload: bool,
    sample_interval: float,
) -> dict[str, Any]:
    sampler = _ResourceSampler(interval=sample_interval)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    sampler.start()
    result = loader._load_single_recording(recording, preload=preload)
    avg_cpu, peak_rss = sampler.stop()
    cpu_end = time.process_time()
    wall_end = time.perf_counter()

    wall_time_s = wall_end - wall_start
    cpu_time_s = cpu_end - cpu_start
    rss_start = None
    rss_end = None
    rss_delta_mb = None
    if psutil:
        proc = psutil.Process(os.getpid())
        try:
            rss_end = proc.memory_info().rss
            rss_start = rss_end
            if peak_rss is not None:
                rss_delta_mb = peak_rss - (rss_start / (1024 * 1024))
        except Exception:
            rss_end = None

    stats = ResourceStats(
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        cpu_util_avg_pct=avg_cpu,
        cpu_util_peak_pct=max(sampler._cpu_samples) if sampler._cpu_samples else None,
        rss_peak_mb=peak_rss,
        rss_delta_mb=rss_delta_mb,
    )

    data_bytes = int(result.data.nbytes)
    samples, channels = result.data.shape
    events_len = None
    if result.events is not None:
        try:
            events_len = len(result.events)
        except Exception:
            events_len = None

    dataset_block = {
        "samples": int(samples),
        "channels": int(channels),
        "sampling_rate": float(result.sampling_rate),
        "data_bytes": data_bytes,
        "data_mb": data_bytes / (1024 * 1024),
        "events": events_len,
        "channels_meta": result.channels is not None,
        "source_path": str(recording),
    }

    # Free memory aggressively.
    try:
        if hasattr(result.raw, "close"):
            result.raw.close()  # type: ignore[attr-defined]
    except Exception:
        pass
    result = None  # type: ignore

    return {"metrics": asdict(stats), "dataset": dataset_block}


def _aggregate_results(per_recording: Sequence[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "recording_count": len(per_recording),
        "total_wall_time_s": 0.0,
        "total_cpu_time_s": 0.0,
        "cpu_util_avg_pct_mean": None,
        "cpu_util_peak_pct_max": None,
        "rss_peak_mb_max": None,
        "data_mb_total": 0.0,
        "samples_total": 0,
        "channels_max": 0,
        "events_total": 0,
    }
    cpu_avgs: list[float] = []
    cpu_peaks: list[float] = []
    rss_peaks: list[float] = []
    for rec in per_recording:
        metrics = rec.get("metrics", {})
        dataset = rec.get("dataset", {})
        totals["total_wall_time_s"] += float(metrics.get("wall_time_s", 0.0))
        totals["total_cpu_time_s"] += float(metrics.get("cpu_time_s", 0.0))
        if metrics.get("cpu_util_avg_pct") is not None:
            cpu_avgs.append(float(metrics["cpu_util_avg_pct"]))
        if metrics.get("cpu_util_peak_pct") is not None:
            cpu_peaks.append(float(metrics["cpu_util_peak_pct"]))
        if metrics.get("rss_peak_mb") is not None:
            rss_peaks.append(float(metrics["rss_peak_mb"]))
        totals["data_mb_total"] += float(dataset.get("data_mb", 0.0))
        totals["samples_total"] += int(dataset.get("samples", 0))
        totals["channels_max"] = max(totals["channels_max"], int(dataset.get("channels", 0)))
        events_val = dataset.get("events")
        if events_val is not None:
            try:
                totals["events_total"] += int(events_val)
            except Exception:
                pass
    if cpu_avgs:
        totals["cpu_util_avg_pct_mean"] = sum(cpu_avgs) / len(cpu_avgs)
    if cpu_peaks:
        totals["cpu_util_peak_pct_max"] = max(cpu_peaks)
    if rss_peaks:
        totals["rss_peak_mb_max"] = max(rss_peaks)
    return totals


def _dedupe_recordings(recordings: Sequence[Path]) -> list[Path]:
    """Prefer BrainVision headers when pairs exist; otherwise keep one per stem."""
    priority = {".vhdr": 0, ".vmrk": 1, ".eeg": 2}
    chosen: dict[tuple[Path, str], Path] = {}
    for rec in recordings:
        stem_key = (rec.parent, rec.stem)
        rank = priority.get(rec.suffix.lower(), 10)
        prev = chosen.get(stem_key)
        if prev is None or priority.get(prev.suffix.lower(), 10) > rank:
            chosen[stem_key] = rec
    return list(chosen.values())


def _fmt(value: Any, suffix: str = "", precision: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}{suffix}"
    except Exception:
        return str(value)


def _short_name(path: Any) -> str:
    try:
        return Path(path).name
    except Exception:
        return str(path) if path is not None else "recording"


def print_summary(results: dict[str, Any]) -> None:
    if "recordings" in results:
        summary = results.get("summary", {})
        config = results.get("config", {})
        records = results.get("recordings", [])
        lines = []
        lines.append("=== BIDS Load Performance (All Recordings) ===")
        lines.append(f"Count: {summary.get('recording_count', len(records))}")
        lines.append(
            f"Totals: wall {_fmt(summary.get('total_wall_time_s'), ' s')}, CPU {_fmt(summary.get('total_cpu_time_s'), ' s')}, "
            f"CPU avg {_fmt(summary.get('cpu_util_avg_pct_mean'), ' %')}, CPU peak {_fmt(summary.get('cpu_util_peak_pct_max'), ' %')}, "
            f"RSS peak {_fmt(summary.get('rss_peak_mb_max'), ' MB')}"
        )
        lines.append(
            f"Data: {_fmt(summary.get('data_mb_total'), ' MB')} across {summary.get('samples_total', 0)} samples; "
            f"events total {summary.get('events_total', 0)}; max channels {summary.get('channels_max', 0)}"
        )
        lines.append(
            f"Root: {config.get('bids_root')} | Subject: {config.get('subject')} | Session: {config.get('session')} | Task: {config.get('task')} | Run: {config.get('run')}"
        )
        lines.append(
            f"Preload: {config.get('preload')} | Sample interval: {_fmt(config.get('sample_interval'), ' s')} | psutil: {config.get('psutil_available')}"
        )
        lines.append("--- Top recordings by wall time ---")
        top = sorted(records, key=lambda r: r.get("metrics", {}).get("wall_time_s", 0.0), reverse=True)[:5]
        for rec in top:
            m = rec.get("metrics", {})
            d = rec.get("dataset", {})
            lines.append(
                f"  {_short_name(d.get('source_path'))}: wall {_fmt(m.get('wall_time_s'), ' s')}, "
                f"CPU {_fmt(m.get('cpu_time_s'), ' s')}, RSS {_fmt(m.get('rss_peak_mb'), ' MB')}, "
                f"{d.get('samples')} x {d.get('channels')} @ {_fmt(d.get('sampling_rate'))} Hz"
            )
        print("\n".join(lines))
        return

    metrics = results.get("metrics", {})
    dataset = results.get("dataset", {})
    config = results.get("config", {})

    lines = []
    lines.append("=== BIDS Load Performance ===")
    lines.append("Metrics:")
    lines.append(
        f"  Wall time: {_fmt(metrics.get('wall_time_s'), ' s')} | CPU time: {_fmt(metrics.get('cpu_time_s'), ' s')}"
    )
    lines.append(
        f"  CPU util avg/peak: {_fmt(metrics.get('cpu_util_avg_pct'), ' %')} / {_fmt(metrics.get('cpu_util_peak_pct'), ' %')}"
    )
    lines.append(
        f"  RSS peak: {_fmt(metrics.get('rss_peak_mb'), ' MB')} | RSS delta: {_fmt(metrics.get('rss_delta_mb'), ' MB')}"
    )
    lines.append("Dataset:")
    lines.append(
        f"  Samples x Ch: {dataset.get('samples')} x {dataset.get('channels')} | SR: {_fmt(dataset.get('sampling_rate'))} Hz"
    )
    lines.append(
        f"  Size: {_fmt(dataset.get('data_mb'), ' MB')} | Events: {dataset.get('events')} | Channels meta: {dataset.get('channels_meta')}"
    )
    lines.append(f"  Source: {dataset.get('source_path')}")
    lines.append("Config:")
    lines.append(
        f"  Root: {config.get('bids_root')} | Subject: {config.get('subject')} | Session: {config.get('session')} | Task: {config.get('task')} | Run: {config.get('run')}"
    )
    lines.append(
        f"  Preload: {config.get('preload')} | Sample interval: {_fmt(config.get('sample_interval'), ' s')} | psutil: {config.get('psutil_available')}"
    )
    print("\n".join(lines))


def export_pdf(results: dict[str, Any], pdf_path: Path) -> None:
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for PDF export. Install it and retry.")

    metrics = results.get("metrics", {})
    recordings = results.get("recordings")
    dataset = results.get("dataset", {})
    config = results.get("config", {})
    summary = results.get("summary", {})

    with PdfPages(pdf_path) as pdf:
        if recordings:
            labels = [_short_name(rec.get("dataset", {}).get("source_path")) for rec in recordings]
            walls = [rec.get("metrics", {}).get("wall_time_s", 0.0) for rec in recordings]
            cpus = [rec.get("metrics", {}).get("cpu_time_s", 0.0) for rec in recordings]
            rss = [rec.get("metrics", {}).get("rss_peak_mb", 0.0) for rec in recordings]

            def _bar_page(values: list[float], ylabel: str, title: str) -> None:
                fig, ax = plt.subplots(figsize=(8.5, 5))
                ax.bar(range(len(values)), values, color="#4C78A8")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=90)
                for idx, val in enumerate(values):
                    ax.text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            _bar_page(walls, "Seconds", "Wall Time per Recording")
            _bar_page(cpus, "Seconds", "CPU Time per Recording")
            _bar_page(rss, "MB", "RSS Peak per Recording")

            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.axis("off")
            text = [
                "BIDS Load Performance Summary (All Recordings)",
                "",
                f"Root: {config.get('bids_root')}",
                f"Count: {summary.get('recording_count')}",
                f"Wall total: {_fmt(summary.get('total_wall_time_s'), ' s')} | CPU total: {_fmt(summary.get('total_cpu_time_s'), ' s')}",
                f"CPU mean/peak: {_fmt(summary.get('cpu_util_avg_pct_mean'), ' %')} / {_fmt(summary.get('cpu_util_peak_pct_max'), ' %')}",
                f"RSS peak (max): {_fmt(summary.get('rss_peak_mb_max'), ' MB')}",
                f"Data total: {_fmt(summary.get('data_mb_total'), ' MB')} | Samples total: {summary.get('samples_total')} | Max channels: {summary.get('channels_max')}",
                f"Events total: {summary.get('events_total')}",
            ]
            ax.text(0.01, 0.95, "\n".join(text), va="top", ha="left", fontsize=10)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            return

        fig, ax = plt.subplots(figsize=(8.5, 4))
        times = [metrics.get("wall_time_s", 0.0), metrics.get("cpu_time_s", 0.0)]
        ax.bar(["Wall", "CPU"], times, color=["#4C78A8", "#F58518"])
        ax.set_ylabel("Seconds")
        ax.set_title("Load Time")
        for idx, val in enumerate(times):
            ax.text(idx, val, f"{val:.3f}s", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4))
        cpu_vals = [
            metrics.get("cpu_util_avg_pct") or 0.0,
            metrics.get("cpu_util_peak_pct") or 0.0,
        ]
        ax.bar(["CPU avg", "CPU peak"], cpu_vals, color="#E45756")
        ax.set_ylabel("Percent")
        ax.set_title("CPU Utilization")
        for idx, val in enumerate(cpu_vals):
            ax.text(idx, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4))
        mem_vals = [metrics.get("rss_peak_mb") or 0.0]
        ax.bar(["RSS peak"], mem_vals, color="#72B7B2")
        ax.set_ylabel("MB")
        ax.set_title("Memory Usage")
        for idx, val in enumerate(mem_vals):
            ax.text(idx, val, f"{val:.1f} MB", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4))
        ax.axis("off")
        text = [
            "BIDS Load Performance Summary",
            "",
            f"Root: {config.get('bids_root')}",
            f"Recording: {dataset.get('source_path')}",
            f"Samples x Channels: {dataset.get('samples')} x {dataset.get('channels')}",
            f"Sampling rate: {_fmt(dataset.get('sampling_rate'))} Hz",
            f"Data size: {_fmt(dataset.get('data_mb'), ' MB')}",
            f"Events: {dataset.get('events')} | Channels meta: {dataset.get('channels_meta')}",
            "",
            "Metrics:",
            f"  Wall: {_fmt(metrics.get('wall_time_s'), ' s')} | CPU: {_fmt(metrics.get('cpu_time_s'), ' s')}",
            f"  CPU avg/peak: {_fmt(metrics.get('cpu_util_avg_pct'), ' %')} / {_fmt(metrics.get('cpu_util_peak_pct'), ' %')}",
            f"  RSS peak: {_fmt(metrics.get('rss_peak_mb'), ' MB')} | RSS delta: {_fmt(metrics.get('rss_delta_mb'), ' MB')}",
        ]
        ax.text(0.01, 0.95, "\n".join(text), va="top", ha="left", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bids_root", type=Path, help="Path to the BIDS dataset root.")
    parser.add_argument("--subject", help="sub-<id> filter.")
    parser.add_argument("--session", help="ses-<id> filter.")
    parser.add_argument("--task", help="task-<name> filter.")
    parser.add_argument("--run", help="run-<id> filter.")
    parser.add_argument(
        "--allowed-ext",
        nargs="+",
        default=None,
        help="Allowed file extensions (default: common EEG formats).",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Disable MNE preload to measure lazy-loading paths.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.05,
        help="Sampling interval (seconds) for CPU/memory metrics (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write results into.",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF report path with charts (requires matplotlib).",
    )
    parser.add_argument(
        "--all-recordings",
        action="store_true",
        help="Benchmark all recordings matching the filters (default: first match).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    results = benchmark_bids_load(
        bids_root=args.bids_root,
        allowed_file_structures=args.allowed_ext,
        subject=args.subject,
        session=args.session,
        task=args.task,
        run=args.run,
        preload=not args.no_preload,
        sample_interval=args.sample_interval,
        all_recordings=args.all_recordings,
    )
    print_summary(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Wrote results to {args.output}")
    if args.pdf_output:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        export_pdf(results, args.pdf_output)
        print(f"Wrote PDF report to {args.pdf_output}")
    if not args.output:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
