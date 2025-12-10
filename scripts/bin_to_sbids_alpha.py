"""Convert BIN -> SBIDS and run Alpha evaluation in one go."""

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


def _json_safe(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


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
        help="Disable matplotlib plots during Alpha evaluation.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Not setting position of .* misc channel.*", category=RuntimeWarning)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load BIN
    ds = CortiDataset.from_bin(args.bin_dir)

    # Alpha evaluation (pandas-like flow)
    evaluation = ds.evaluate_alpha(show_plots=not args.no_plots)
    summary = {k: _json_safe(v) for k, v in list(evaluation.items())[:5]}
    print("ALPHA evaluation completed. Key metrics:")
    print(json.dumps(summary, indent=2))

    # Export SBIDS
    output = args.output or (args.bin_dir / f"sbids_meta_{args.bin_dir.name}.jsonld")
    sbids_path = ds.to_sbids(output, export_format=args.export_format)
    print(f"Wrote SBIDS JSON-LD to {sbids_path}")

    if not args.no_plots:
        print("Close plot windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
