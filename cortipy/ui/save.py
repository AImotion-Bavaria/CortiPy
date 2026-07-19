"""Data persistence utilities replacing MATLAB `saveDatamain`/`mySave`."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


def _sanitize_stem(name: Any) -> str:
    """Filesystem-safe stem: keep alphanumerics, '-' and '_'; spaces become '_'."""
    text = str(name or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_-]", "", text)
    return text.strip("_-")


class SaveManager:
    """Persist Params/data locally and optionally forward to a custom callback."""

    def __init__(
        self,
        base_dir: str | Path | None = None,
        database_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.base_dir = Path(base_dir or Path.cwd() / "cortipy_runs")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.database_callback = database_callback
        self.last_target_dir: Optional[Path] = None

    def __call__(self, params: Dict[str, Any], target_dir: str | Path | None = None) -> Path:
        if target_dir is None:
            target_dir = self._default_target_dir(params)
        else:
            target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        self.last_target_dir = target_dir

        # Recording into a folder that already holds one (an explicit "active dataset
        # folder" reused across runs) must not overwrite it. Number the run instead, so
        # every recording is kept; the loader reads back the newest one.
        params_name, data_name = self._next_recording_names(target_dir)

        safe_params = dict(params)
        data = safe_params.pop("data", None)
        safe_params.setdefault("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if data is not None:
            data_file = target_dir / data_name
            np.savez_compressed(data_file, data=np.asarray(data))
            safe_params["DataFile"] = data_file.name

        (target_dir / params_name).write_text(
            json.dumps(safe_params, indent=2, default=_json_fallback),
            encoding="utf-8",
        )

        if self.database_callback is not None:
            self.database_callback(params)

        return target_dir

    @staticmethod
    def _next_recording_names(target_dir: Path) -> tuple[str, str]:
        """(params_name, data_name) for the next recording in ``target_dir``.

        The first recording is ``params.json`` / ``data.npz`` (unchanged). If those already
        exist, subsequent recordings become ``params_run-02.json`` / ``data_run-02.npz``,
        ``…run-03…`` and so on — nothing is overwritten. ``params.json`` counts as run 1.
        """
        runs = [1] if (target_dir / "params.json").exists() else []
        for path in target_dir.glob("params_run-*.json"):
            match = re.search(r"run-(\d+)", path.name)
            if match:
                runs.append(int(match.group(1)))
        if not runs:
            return "params.json", "data.npz"
        nxt = max(runs) + 1
        return f"params_run-{nxt:02d}.json", f"data_run-{nxt:02d}.npz"

    def _default_target_dir(self, params: Dict[str, Any]) -> Path:
        """Name the run folder after the operator's typed filename when there is one.

        The folder used to be ``{timestamp}_{method}`` every time, so the name never
        reflected the measurement the operator described. The filename typed on the session
        page (``Parameters.Filename``) now drives it; runs with the same name are numbered
        rather than overwritten, so no recording is lost. Without a filename we fall back to
        the old timestamped, method-named folder.
        """
        pblock = params.get("Parameters")
        filename = pblock.get("Filename") if isinstance(pblock, dict) else None
        stem = _sanitize_stem(filename) or _sanitize_stem(params.get("Filename"))
        if stem:
            base = stem
        else:
            method = params.get("Method", "Unknown")
            base = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{method}"

        candidate = self.base_dir / base
        suffix = 2
        while candidate.exists():
            candidate = self.base_dir / f"{base}_{suffix}"
            suffix += 1
        return candidate


def _json_fallback(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
