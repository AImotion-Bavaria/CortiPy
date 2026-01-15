"""CLI wrapper to convert SBIDS JSON-LD into BIDS via CortiDataset."""

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
    parser.add_argument("sbids_path", type=Path, help="Path to SBIDS JSON-LD.")
    parser.add_argument("bids_root", type=Path, help="Destination BIDS root.")
    parser.add_argument("--subject", required=True, help="sub-<id> for export.")
    parser.add_argument("--session", help="ses-<id> for export.")
    parser.add_argument("--task", help="task-<name> for export.")
    parser.add_argument("--run", help="run-<id> for export.")
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        help="Additional roots to resolve raw data (can be passed multiple times).",
    )
    parser.add_argument(
        "--format",
        default="fif",
        help="Export format for raw data (default: %(default)s).",
    )
    parser.add_argument(
        "--recording-id",
        default=None,
        help="Specific recording @id to export (default: first).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting destination files.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Not setting position of .* misc channel.*", category=RuntimeWarning)

    parser = _build_parser()
    args = parser.parse_args(argv)

    ds = CortiDataset.from_sbids(args.sbids_path, data_roots=args.data_root, recording_id=args.recording_id)
    path = ds.to_bids(
        args.bids_root,
        subject=args.subject,
        session=args.session,
        task=args.task,
        run=args.run,
        format=args.format,
        overwrite=args.overwrite,
    )
    print(f"Wrote BIDS data to {path}")


if __name__ == "__main__":
    main()
