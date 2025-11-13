"""Data persistence utilities replacing MATLAB `saveDatamain`/`mySave`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


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

    def __call__(self, params: Dict[str, Any]) -> None:
        method = params.get("Method", "Unknown")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target_dir = self.base_dir / f"{timestamp}_{method}"
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_params = dict(params)
        data = safe_params.pop("data", None)
        safe_params.setdefault("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        data_file = None
        if data is not None:
            data_file = target_dir / "data.npz"
            np.savez_compressed(data_file, data=np.asarray(data))
            safe_params["DataFile"] = data_file.name

        (target_dir / "params.json").write_text(
            json.dumps(safe_params, indent=2, default=_json_fallback),
            encoding="utf-8",
        )

        if self.database_callback is not None:
            self.database_callback(params)


def _json_fallback(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
