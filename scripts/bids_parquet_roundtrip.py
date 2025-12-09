"""Load a BIDS dataset and re-export it to a Parquet-based BIDS layout."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional


def _load_bidsloader() -> type:
    """Load BIDSLoader without pulling in the full cortipy package (avoids pylsl lib issues)."""
    try:
        from cortipy.shared.bids import BIDSLoader as Loader  # type: ignore

        return Loader
    except Exception:
        repo_root = Path(__file__).resolve().parent.parent
        bids_path = repo_root / "cortipy" / "shared" / "bids.py"
        spec = importlib.util.spec_from_file_location("cortipy.shared.bids", bids_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return getattr(module, "BIDSLoader")


BIDSLoader = _load_bidsloader()


DEFAULT_ALLOWED = [
    ".fif",
    ".edf",
    ".bdf",
    ".vhdr",
    ".set",
    ".eeg",
    ".parquet",
    ".h5",
    ".hdf5",
    ".zarr",
]


def _infer_token(stem: str, prefix: str) -> Optional[str]:
    for token in stem.split("_"):
        if token.startswith(f"{prefix}-"):
            return token.split("-", 1)[1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bids_root", type=Path, help="Path to the source BIDS dataset")
    parser.add_argument("output_root", type=Path, help="Where to write the Parquet BIDS export")
    parser.add_argument("--subject", help="Subject label (defaults to value inferred from filename)")
    parser.add_argument("--session", help="Session label (defaults to value inferred from filename)")
    parser.add_argument("--task", help="Task label (defaults to value inferred from filename)")
    parser.add_argument("--run", help="Run label (defaults to value inferred from filename)")
    parser.add_argument(
        "--allowed",
        nargs="+",
        default=DEFAULT_ALLOWED,
        help="Allowed file extensions to search for in the source BIDS root",
    )
    parser.add_argument(
        "--format",
        default="parquet",
        choices=["parquet", "fif", "edf", "bdf", "brainvision", "eeglab", "hdf5", "h5", "zarr"],
        help="Export format for the new BIDS dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all recordings matching the filters instead of just the first match",
    )
    parser.add_argument("--modality", default="eeg", help="Modality directory to use for the export")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing output files")
    args = parser.parse_args()

    source_loader = BIDSLoader(args.bids_root)
    results = []
    if args.all:
        results = source_loader.read_bids_dataset(
            allowed_file_structures=args.allowed,
            subject=args.subject,
            session=args.session,
            task=args.task,
            run=args.run,
        )
    else:
        try:
            results = [
                source_loader.read_bids(
                    allowed_file_structures=args.allowed,
                    subject=args.subject,
                    session=args.session,
                    task=args.task,
                    run=args.run,
                )
            ]
        except FileNotFoundError:
            if args.run:
                results = [
                    source_loader.read_bids(
                        allowed_file_structures=args.allowed,
                        subject=args.subject,
                        session=args.session,
                        task=args.task,
                        run=None,
                    )
                ]
            else:
                raise

    target_loader = BIDSLoader(args.output_root)

    for result in results:
        inferred_subject = args.subject or _infer_token(result.source_path.stem, "sub")
        if inferred_subject is None:
            raise ValueError("Subject could not be inferred; pass --subject explicitly.")

        sidecars = result.metadata.get("sidecars", {}) if result.metadata else {}
        sidecar_payload = next(iter(sidecars.values())) if sidecars else None
        dataset_description = result.metadata.get("dataset_description") if result.metadata else None

        out_path = target_loader.to_bids(
            result.raw,
            subject=inferred_subject,
            session=args.session or _infer_token(result.source_path.stem, "ses"),
            task=args.task or _infer_token(result.source_path.stem, "task"),
            run=args.run or _infer_token(result.source_path.stem, "run"),
            modality=args.modality,
            format=args.format,
            events=result.events,
            channels=result.channels,
            sidecar=sidecar_payload,
            dataset_description=dataset_description,
            ancillary_files=result.ancillary_files,
            overwrite=args.overwrite,
        )

        print(f"Loaded:   {result.source_path}")
        print(f"Exported: {out_path}")


if __name__ == "__main__":
    main()
