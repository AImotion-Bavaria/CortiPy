"""CLI wrapper to convert a CortiPy BIN recording into SBIDS JSON-LD via CortiDataset."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cortipy.shared import CortiDataset  # type: ignore # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bin_dir", type=Path, help="Path to BIN dataset directory (contains params.json + data.bin).")
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
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Not setting position of .* misc channel.*", category=RuntimeWarning)

    parser = _build_parser()
    args = parser.parse_args(argv)

    ds = CortiDataset.from_bin(args.bin_dir)
    output = args.output or (args.bin_dir / f"sbids_meta_{args.bin_dir.name}.jsonld")
    path = ds.to_sbids(output, export_format=args.export_format)
    print(f"Wrote SBIDS JSON-LD to {path}")


if __name__ == "__main__":
    main()
