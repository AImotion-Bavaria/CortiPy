"""Configuration loading helpers for running cortipy without the MATLAB UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def load_config(path_or_obj: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    """Load a configuration from JSON/TOML or return mappings untouched."""
    if isinstance(path_or_obj, Mapping):
        return dict(path_or_obj)
    path = Path(path_or_obj)
    suffix = path.suffix.lower()
    if suffix in {".json", ".params"}:
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".toml", ".tml"}:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config format for {path}")


def normalize_params(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a config mapping into the Params structure expected by cortipy."""
    normalized: Dict[str, Any] = {}
    normalized["Method"] = config.get("method") or config.get("Method")
    if not normalized["Method"]:
        raise ValueError("Config must specify a 'method'.")
    normalized["Device"] = config.get("device") or config.get("Device") or "LSL"

    parameters = config.get("parameters") or config.get("Parameters") or {}
    normalized["Parameters"] = dict(parameters)

    channels = config.get("channels") or config.get("Channels")
    if channels is not None:
        normalized["Channels"] = list(channels)

    data = config.get("data") or config.get("Data")
    if data is not None:
        normalized["data"] = data

    if "DataFile" in config:
        normalized["DataFile"] = config["DataFile"]
    if "dataFile" in config and "DataFile" not in normalized:
        normalized["DataFile"] = config["dataFile"]

    evaluation = config.get("Evaluation") or config.get("evaluation")
    if evaluation is not None:
        normalized["Evaluation"] = evaluation

    timestamp = config.get("Timestamp") or config.get("timestamp")
    if timestamp is not None:
        normalized["Timestamp"] = timestamp

    metadata = config.get("metadata") or config.get("Metadata")
    if metadata is not None:
        normalized["Metadata"] = metadata

    if "dataPath" in config:
        normalized["DataPath"] = config["dataPath"]
    if "save" in config:
        normalized.setdefault("Save", config["save"])
    return normalized
