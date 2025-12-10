"""CLI wrapper for converting BIDS datasets to SBIDS using the unified CortiDataset API."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cortipy.shared import CortiDataset, to_sbids  # type: ignore


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
        help="Disable MNE preload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SBIDS JSON-LD path (default: sbids_meta_<dataset>.jsonld under the BIDS root).",
    )
    parser.add_argument(
        "--all-recordings",
        action="store_true",
        help="Export all recordings in the dataset (default: only those matching filters).",
    )
    parser.add_argument(
        "--export-format",
        default="parquet",
        choices=["parquet", "npz", "copy", "edf", "hdf5", "zarr"],
        help="Container to store raw data in SBIDS raw_data/ (default: %(default)s).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Not setting position of .* misc channel.*", category=RuntimeWarning)
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_path = to_sbids(
        bids_root=args.bids_root,
        subject=args.subject,
        session=args.session,
        task=args.task,
        run=args.run,
        allowed_ext=args.allowed_ext,
        preload=not args.no_preload,
        output=args.output,
        all_recordings=args.all_recordings,
        export_format=args.export_format,
    )
    print(f"Wrote SBIDS JSON-LD to {output_path}")


if __name__ == "__main__":
    main()
