"""Helpers for working with parameter dictionaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, MutableMapping


def load_params_from_file(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    # Heuristics: if JSON has a single top-level key, unwrap it.
    if isinstance(data, dict) and len(data) == 1:
        (key, value), = data.items()
        if isinstance(value, dict):
            value.setdefault("Method", key)
            return value
    return data


def merge_general_params(general: MutableMapping[str, Any], specific: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    merged = dict(general)
    merged.setdefault("Parameters", {})
    for key, value in specific.items():
        if key == "Parameters" and isinstance(value, dict):
            merged["Parameters"] = {**merged["Parameters"], **value}
        else:
            merged[key] = value
    return merged
