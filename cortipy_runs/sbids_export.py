"""CLI entrypoint to export a single CortiPy dataset folder into SBIDS JSON-LD."""

from __future__ import annotations

import argparse
from pathlib import Path

from sbids import default_output_path, export_dataset
from test import run_validations


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path or name of the dataset folder (must contain params.json)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base directory to resolve dataset_dir when a bare name is provided (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="SBIDS dataset identifier (default: dataset-name upper-snake)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the SBIDS JSON-LD document (default: sbids_meta_<dataset>.jsonld)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON output (default: %(default)s)",
    )
    return parser


def _resolve_dataset_dir(dataset_dir: Path, base_dir: Path) -> Path:
    if dataset_dir.exists():
        return dataset_dir
    candidate = base_dir / dataset_dir
    if candidate.exists():
        return candidate
    raise SystemExit(f"Dataset directory not found: {dataset_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dataset_dir = _resolve_dataset_dir(args.dataset_dir, args.base_dir)
    dataset_name = dataset_dir.name
    dataset_id = args.dataset_id or dataset_name.replace(" ", "_").upper()
    output = args.output or dataset_dir.parent / default_output_path(dataset_name, dataset_id)

    export_dataset(
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        output=output,
        indent=args.indent,
    )
    stats = run_validations(dataset_dir=dataset_dir, sbids_path=output, quiet=True)
    print(f"Wrote {output} from {dataset_dir}.")
    print(f"Data matched (recordings={stats['recordings']}, channels={stats['channels']}).")
    print("JSON validated (jsonschema).")
    print("JSON-LD validated (pyld).")


if __name__ == "__main__":
    main()
