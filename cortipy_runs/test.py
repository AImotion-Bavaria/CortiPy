"""Validation helper to compare CortiPy params.json files to SBIDS JSON-LD output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sbids import (
    build_sbids_document_for_dataset,
    default_output_path,
    export_dataset,
    load_params,
)


try:
    import jsonschema
    from jsonschema import validate

    HAS_JSONSCHEMA = True
except Exception:  # pragma: no cover - import guard
    HAS_JSONSCHEMA = False

try:
    from pyld import jsonld

    HAS_PYLD = True
except Exception:  # pragma: no cover - import guard
    HAS_PYLD = False


def _require_validators() -> None:
    missing = []
    if not HAS_JSONSCHEMA:
        missing.append("jsonschema")
    if not HAS_PYLD:
        missing.append("pyld")
    if missing:
        raise SystemExit(
            f"Missing validation dependencies: {', '.join(missing)}. "
            "Install with: pip install jsonschema pyld"
        )


def _load_sbids(sbids_path: Path) -> dict:
    try:
        return json.loads(sbids_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {sbids_path}: {exc}") from exc


def _extract_dataset_info(doc: dict) -> tuple[str, str]:
    dataset_id = doc.get("@id", "").replace("urn:dataset:", "")
    dataset_name = None
    for node in doc.get("@graph", []):
        if node.get("@type") == "schema:Dataset":
            dataset_name = node.get("schema:name")
            break
    if not dataset_id or not dataset_name:
        raise SystemExit("SBIDS JSON-LD missing dataset id or name.")
    return dataset_id, dataset_name


def _build_node_map(doc: dict) -> dict[str, dict]:
    return {node.get("@id"): node for node in doc.get("@graph", []) if "@id" in node}


def _validate_structure_jsonschema(doc: dict) -> None:
    schema = {
        "type": "object",
        "required": ["@context", "@graph"],
        "properties": {
            "@context": {"type": "object"},
            "@graph": {
                "type": "array",
                "items": {"type": "object", "required": ["@id"], "properties": {"@id": {"type": "string"}}},
                "minItems": 1,
            },
        },
    }
    validate(instance=doc, schema=schema)


def _validate_jsonld(doc: dict) -> None:
    expanded = jsonld.expand(doc)
    if not isinstance(expanded, list):
        raise SystemExit("JSON-LD expansion did not return a list.")
    if len(expanded) == 0:
        raise SystemExit("JSON-LD expansion returned empty result.")


def _validate_channels(original_doc: dict, regenerated_doc: dict) -> None:
    """Ensure channel lists survived the migration (count + names)."""
    orig_nodes = _build_node_map(original_doc)
    regen_nodes = _build_node_map(regenerated_doc)
    for node_id, node in orig_nodes.items():
        if "schema:variableMeasured" not in node:
            continue
        expected = node.get("schema:variableMeasured", [])
        actual = regen_nodes.get(node_id, {}).get("schema:variableMeasured", [])
        if len(expected) != len(actual):
            raise SystemExit(
                f"Channel count mismatch for {node_id}: expected {len(expected)} != {len(actual)}"
            )
        for exp_chan, act_chan in zip(expected, actual):
            if exp_chan.get("columnName") != act_chan.get("columnName"):
                raise SystemExit(
                    f"Channel name mismatch in {node_id}: {exp_chan.get('columnName')} != {act_chan.get('columnName')}"
                )


def _compare_docs(expected: dict, actual: dict) -> None:
    expected_map = _build_node_map(expected)
    actual_map = _build_node_map(actual)
    if expected_map.keys() != actual_map.keys():
        missing = expected_map.keys() - actual_map.keys()
        extra = actual_map.keys() - expected_map.keys()
        raise SystemExit(f"Graph node mismatch. Missing: {missing}, Extra: {extra}")
    for node_id, exp_node in expected_map.items():
        act_node = actual_map[node_id]
        if exp_node != act_node:
            raise SystemExit(f"Node mismatch for {node_id}")


def _assert_inputs(dataset_dir: Path, sbids_path: Path) -> None:
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")
    params_path = dataset_dir / "params.json"
    if not params_path.exists():
        raise SystemExit(f"params.json not found under {dataset_dir}")
    if not sbids_path.exists():
        raise SystemExit(f"SBIDS file not found: {sbids_path}")


def run_validations(dataset_dir: Path, sbids_path: Path, quiet: bool = False) -> dict:
    _require_validators()
    _assert_inputs(dataset_dir, sbids_path)
    source_meta = load_params(dataset_dir)
    sbids_doc = _load_sbids(sbids_path)
    _validate_structure_jsonschema(sbids_doc)
    _validate_jsonld(sbids_doc)
    dataset_id, dataset_name = _extract_dataset_info(sbids_doc)
    regenerated_doc, _ = build_sbids_document_for_dataset(
        dataset_dir=dataset_dir, dataset_id=dataset_id, dataset_name=dataset_name
    )
    _compare_docs(regenerated_doc, sbids_doc)
    _validate_channels(regenerated_doc, sbids_doc)
    stats = {
        "recordings": 1,
        "channels": len(source_meta.get("Channels", [])),
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
    }
    if not quiet:
        print(f"Data matched: recordings={stats['recordings']}, channels={stats['channels']}")
        print("JSON validated (jsonschema).")
        print("JSON-LD validated (pyld expansion).")
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to a single dataset directory containing params.json",
    )
    parser.add_argument(
        "--sbids-path",
        type=Path,
        default=None,
        help="Path to sbids_meta_<dataset>.jsonld (default: current working directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")
    sbids_path = args.sbids_path or Path.cwd() / default_output_path(dataset_dir.name)
    if not sbids_path.exists():
        dataset_name = dataset_dir.name
        dataset_id = dataset_name.replace(" ", "_").upper()
        print(f"Exporting {dataset_dir} -> {sbids_path}")
        export_dataset(
            dataset_dir=dataset_dir,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            output=sbids_path,
            indent=2,
        )
    run_validations(dataset_dir=dataset_dir, sbids_path=sbids_path)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(exc, file=sys.stderr)
        raise
