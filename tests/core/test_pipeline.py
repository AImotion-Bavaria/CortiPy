"""Integration tests for MeasurementPipeline."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from cortipy.core.pipeline import MeasurementPipeline, PipelineHooks
from cortipy.modules.base import ModuleBase


class TrackingModule(ModuleBase):
    """Test double that records lifecycle invocations without touching hardware."""

    def __init__(self, name: str, method: str) -> None:
        super().__init__(name, [method])
        self.connected = 0
        self.collected = 0
        self.uses_device_called = 0

    def uses_device(self, context) -> bool:  # type: ignore[override]
        self.uses_device_called += 1
        return False

    def on_connected(self, context) -> None:  # type: ignore[override]
        self.connected += 1

    def collect_measurements(self, context) -> None:  # type: ignore[override]
        self.collected += 1
        payload = context.params.setdefault("Collected", deque())
        payload.append(self.name)
        context.append_data(np.ones((1, 1)))


def test_run_once_executes_all_modules(module_context_factory):
    """run_once should execute connect and collect for every module, then return the enriched params dictionary."""
    modules = [TrackingModule("AlphaA", "alpha"), TrackingModule("AlphaB", "alpha")]
    pipeline = MeasurementPipeline(modules=modules)
    params = {
        "Method": "alpha",
        "Device": "Dummy",
        "Parameters": {},
    }

    result = pipeline.run_once(params)

    assert list(result["Collected"]) == ["AlphaA", "AlphaB"]
    assert modules[0].connected == modules[0].collected == 1
    assert modules[1].connected == modules[1].collected == 1


def test_run_with_hooks_iterates_until_provider_returns_none(monkeypatch):
    """Main pipeline run should repeatedly pull params from the provider hook, save after each iteration, and release resources."""
    modules = [TrackingModule("Alpha", "alpha")]
    saves: list[dict] = []

    def params_provider(previous):
        if previous is None:
            return {
                "Method": "alpha",
                "Device": "Dummy",
                "Parameters": {},
            }
        return None

    hooks = PipelineHooks(
        params_provider=params_provider,
        should_continue=lambda params: False,
        save_callback=lambda params: saves.append(dict(params)),
    )
    pipeline = MeasurementPipeline(modules=modules, hooks=hooks)

    pipeline.run()

    assert len(saves) == 1
    assert saves[0]["Collected"].pop() == "Alpha"


def test_default_modules_returns_expected_order():
    """The default measurement pipeline should instantiate the canonical module ordering used in the MATLAB suite."""
    pipeline = MeasurementPipeline()
    names = [module.__class__.__name__ for module in pipeline.modules]
    assert names == [
        "AlphaModule",
        "VepModule",
        "SsvepModule",
        "AssrModule",
        "BeraModule",
        "BciModule",
        "P300Module",
    ]
