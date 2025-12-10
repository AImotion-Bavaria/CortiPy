"""Convert BIN -> SBIDS and run all available evaluations (Alpha, SSVEP, BERA, P300, VEP, ASSR)."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cortipy.shared import CortiDataset  # type: ignore


def _summarize_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Reduce evaluation payload to concise statistics (mean of numeric fields)."""
    import numpy as np

    if not isinstance(evaluation, dict) or not evaluation:
        return {}

    summary: dict[str, Any] = {"keys": sorted(evaluation.keys())}
    means: dict[str, float] = {}
    for key, value in evaluation.items():
        try:
            arr = np.asarray(value, dtype=float)
            if arr.size == 0:
                continue
            val = float(np.nanmean(arr))
            if np.isfinite(val):
                means[key] = val
        except Exception:
            continue
    if means:
        summary["means"] = means
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bin_dir", type=Path, help="Path to BIN dataset directory (params.json + data.bin).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SBIDS JSON-LD path (default: sbids_meta_<dataset>.jsonld under bin_dir).",
    )
    parser.add_argument(
        "--export-format",
        default="parquet",
        choices=["parquet", "npz", "copy", "edf", "hdf5", "zarr"],
        help="Container to store raw data in SBIDS raw_data/ (default: %(default)s).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["alpha", "ssvep", "bera", "p300", "vep", "assr"],
        choices=["alpha", "ssvep", "bera", "p300", "vep", "assr"],
        help="Which evaluators to run (default: all).",
    )
    parser.add_argument(
        "--stim-freq",
        type=float,
        nargs="+",
        default=None,
        help="Override stimulus frequencies (Hz) for SSVEP evaluation.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib plots during evaluations.",
    )
    return parser


def _prepare_params(base: dict[str, Any], method: str, stim_freq: Optional[list[float]]) -> dict[str, Any]:
    params = dict(base) if isinstance(base, dict) else {}
    params["Method"] = method.upper()
    if method == "ssvep" and stim_freq is not None:
        params.setdefault("Parameters", {})
        params["Parameters"]["StimFreq"] = list(stim_freq)
    return params


def _default_params() -> dict[str, Any]:
    return {"Method": "ALPHA", "Parameters": {}}


def main(argv: Optional[list[str]] = None) -> None:
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Not setting position of .* misc channel.*", category=RuntimeWarning)

    parser = _build_parser()
    args = parser.parse_args(argv)

    ds = CortiDataset.from_bin(args.bin_dir)
    base_params = ds.metadata.get("params", {}) if isinstance(ds.metadata, dict) else {}
    if not isinstance(base_params, dict):
        base_params = _default_params()

    summaries: dict[str, dict[str, Any]] = {}
    for method in args.methods:
        method_key = method.lower()
        params = _prepare_params(base_params, method_key, args.stim_freq)
        eval_result = ds.evaluate(method_key, params=params, show_plots=not args.no_plots, store_result=True)
        summaries[method_key] = _summarize_evaluation(eval_result)

    output = args.output or (args.bin_dir / f"sbids_meta_{args.bin_dir.name}.jsonld")
    sbids_path = ds.to_sbids(output, export_format=args.export_format)

    print("Evaluations completed (means shown where available):")
    print(json.dumps(summaries, indent=2))
    print(f"Wrote SBIDS JSON-LD to {sbids_path}")

    if not args.no_plots:
        print("Close plot windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
