"""Run all experiments and collect paper-ready outputs.

Default behavior emits only the paper-target artifacts into `experiments/paper_outputs/`.
Use `--plot-all` to keep each experiment's full plot suite as it currently does.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "paper_outputs"


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    mpl_dir = Path(os.environ.get("MPLCONFIGDIR", "")) if os.environ.get("MPLCONFIGDIR") else None
    if not mpl_dir:
        mpl_dir = Path("/tmp") / "cortipy_mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(mpl_dir)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _pick_existing(base_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    fallback = sorted(base_dir.glob("*_topomap_*.pdf"))
    if fallback:
        return fallback[0]
    tried = ", ".join(candidates)
    raise FileNotFoundError(f"No topomap PDF found in {base_dir} (tried: {tried})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Keep each experiment's full plot suite (default only generates paper outputs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to collect paper outputs (default: experiments/paper_outputs).",
    )
    parser.add_argument(
        "--d1-output-root",
        type=Path,
        default=None,
        help="Override output root for D1 exports (e.g., external disk).",
    )
    args = parser.parse_args(argv)

    plot_mode = "all" if args.plot_all else "paper"
    d1_args: list[str] = []
    if args.d1_output_root:
        d1_args = ["--d1-output-root", str(args.d1_output_root)]

    # Experiment 1 (topomaps)
    _run([sys.executable, "experiments/experiment1/run_experiment1.py", "--plot-mode", plot_mode])

    # Experiment 2 (format throughput plot)
    _run(
        [
            sys.executable,
            "experiments/experiment2/run_experiment2.py",
            "--plot-mode",
            plot_mode,
            *d1_args,
        ]
    )

    # Experiment 3 (streaming summary table)
    exp3_cmd = [sys.executable, "experiments/experiment3/run_experiment3.py", "--plot-mode", plot_mode, *d1_args]
    if plot_mode == "paper":
        exp3_cmd += ["--formats", "parquet,edf", "--containers", "sbids,bids", "--keep-artifacts"]
    _run(exp3_cmd)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect required PDFs.
    exp2_root = Path("experiments/experiment2/results")
    _copy(
        REPO_ROOT / exp2_root / "extension_rw_throughput_box.pdf",
        out_dir / "extension_rw_throughput_box.pdf",
    )

    exp1_root = Path("experiments/experiment1/results/cortipy_plots")
    topomap_map = {
        "ABR": (
            [
                "ABR_topomap_6ms.pdf",
                "ABR_topomap_abr-topomap-60-ms.pdf",
                "ABR_topomap_abr-topomap-60-ms_2.pdf",
                "ABR_topomap_7ms.pdf",
            ],
            "CortiPy_ABR_topomap_6ms.pdf",
        ),
        "ASSR": (
            [
                "ASSR_topomap_40Hz.pdf",
                "ASSR_topomap_assr-topomap-40-0-hz.pdf",
                "ASSR_topomap_assr-topomap-40-0-hz_2.pdf",
            ],
            "CortiPy_ASSR_topomap_40Hz.pdf",
        ),
        "Oddball": (
            [
                "Oddball_topomap_300ms.pdf",
                "Oddball_topomap_p300-topomap-300-ms.pdf",
                "Oddball_topomap_p300-topomap-300-ms_2.pdf",
            ],
            "CortiPy_Oddball_topomap_300ms.pdf",
        ),
        "VEP": (
            [
                "VEP_topomap_100ms.pdf",
                "VEP_topomap_vep-topomap-100-ms.pdf",
                "VEP_topomap_p100-topography.pdf",
            ],
            "CortiPy_VEP_topomap_100ms.pdf",
        ),
        "SSVEP": (
            [
                "SSVEP_topomap_10Hz.pdf",
                "SSVEP_topomap_ssvep-topomap-10-0-hz.pdf",
                "SSVEP_topomap_ssvep-topomap-10-0-hz_2.pdf",
            ],
            "CortiPy_SSVEP_topomap_10Hz.pdf",
        ),
    }
    for ds, (candidates, dst_name) in topomap_map.items():
        src = _pick_existing(REPO_ROOT / exp1_root / ds, candidates)
        _copy(src, out_dir / dst_name)

    _copy(
        REPO_ROOT / "experiments/experiment3/results" / "D1_streaming_single_channel_latency_table.pdf",
        out_dir / "D1_streaming_single_channel_latency_table.pdf",
    )

    print(f"Wrote paper outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
