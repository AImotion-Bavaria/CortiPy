"""Convert BIN -> SBIDS and run BERA evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cortipy.shared import CortiDataset  # type: ignore


def _summarize_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Reduce evaluation payload to concise statistics."""
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
        "--no-plots",
        action="store_true",
        help="Disable matplotlib plots during BERA evaluation.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Not setting position of .* misc channel.*", category=RuntimeWarning)

    parser = _build_parser()
    args = parser.parse_args(argv)

    ds = CortiDataset.from_bin(args.bin_dir)

    evaluation = ds.evaluate_bera(show_plots=not args.no_plots)
    summary = _summarize_evaluation(evaluation)
    print("BERA evaluation completed. Summary:")
    print(json.dumps(summary, indent=2))

    output = args.output or (args.bin_dir / f"sbids_meta_{args.bin_dir.name}.jsonld")
    sbids_path = ds.to_sbids(output, export_format=args.export_format)
    print(f"Wrote SBIDS JSON-LD to {sbids_path}")

    if not args.no_plots:
        print("Close plot windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
