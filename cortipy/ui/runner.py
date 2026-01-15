"""Programmatic entry points for cortipy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from cortipy.core.pipeline import MeasurementPipeline, PipelineHooks

from .config import load_config, normalize_params
from .save import SaveManager


Params = Dict[str, object]
ParamsProvider = Callable[[Optional[Params]], Optional[Params]]


@dataclass
class SessionOptions:
    save_dir: Path | str | None = None
    database_callback: Optional[Callable[[Params], None]] = None
    continue_prompt: bool = False


def run_from_config(config_path: str | Path, options: SessionOptions | None = None) -> Params:
    """Load a config file and execute the measurement pipeline once."""
    config = load_config(config_path)
    params = normalize_params(config)
    return run_session(params, options=options)


def run_session(params: Params, options: SessionOptions | None = None) -> Params:
    """Execute the measurement pipeline once with a prepared params dict."""
    opts = options or SessionOptions()
    saver = SaveManager(opts.save_dir, opts.database_callback)

    provider_called = {"done": False}

    def params_provider(_: Optional[Params]) -> Optional[Params]:
        if provider_called["done"]:
            return None
        provider_called["done"] = True
        return params

    hooks = PipelineHooks(
        params_provider=params_provider,
        save_callback=saver,
        should_continue=(lambda _: False) if not opts.continue_prompt else None,
    )

    pipeline = MeasurementPipeline(hooks=hooks)
    pipeline.run()
    return params
