"""Shared pytest fixtures and utilities for cortipy tests."""

from __future__ import annotations

import importlib
from collections import deque
from dataclasses import dataclass
import inspect
import textwrap
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np
import pytest

from cortipy.core.context import ModuleContext
from cortipy.devices.base import DeviceInterface
from cortipy.evaluation.base import EvaluatorBase


class FakeDevice(DeviceInterface):
    """Deterministic device double that records lifecycle calls."""

    def __init__(
        self,
        prime_return: Any | None = None,
        acquire_returns: Iterable[Any] | None = None,
    ) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.prime_calls: list[Tuple[float, int]] = []
        self.acquire_calls: list[Tuple[float, int]] = []
        self.prime_return = prime_return
        self.acquire_returns = deque(acquire_returns or [])

    def connect(self) -> None:
        self.connect_calls += 1
        self.connected = True

    def prime(self, duration_seconds: float, aux_channels: int = 0) -> Any:
        self.prime_calls.append((duration_seconds, aux_channels))
        if self.prime_return is not None:
            return self.prime_return
        return self._make_chunk(duration_seconds, aux_channels)

    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> Any:
        self.acquire_calls.append((duration_seconds, aux_channels))
        if self.acquire_returns:
            return self.acquire_returns.popleft()
        return self._make_chunk(duration_seconds, aux_channels)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def _make_chunk(self, duration_seconds: float, aux_channels: int) -> np.ndarray:
        rows = max(1, int(round(duration_seconds * 10)))
        cols = max(1, int(aux_channels) or 1)
        return np.zeros((rows, cols))


class FakeEvaluator(EvaluatorBase):
    """Evaluator spy that records invocations."""

    def __init__(self) -> None:
        self.calls: list[ModuleContext] = []

    def evaluate(self, context: ModuleContext) -> None:  # type: ignore[override]
        self.calls.append(context)


@dataclass
class CallRecord:
    args: tuple[Any, ...]
    kwargs: Dict[str, Any]


class CallSpy:
    """Callable spy that stores ordered call records."""

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(CallRecord(args=args, kwargs=kwargs))


@pytest.fixture
def fake_device() -> FakeDevice:
    return FakeDevice()


@pytest.fixture
def fake_evaluator() -> FakeEvaluator:
    return FakeEvaluator()


@pytest.fixture
def params_factory() -> Callable[..., Dict[str, Any]]:
    """Factory returning fresh params dicts for each test."""

    def _factory(**overrides: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "Method": "Alpha",
            "Device": "Dummy",
            "Parameters": {},
        }
        for key, value in overrides.items():
            if key == "Parameters" and isinstance(value, dict):
                merged = dict(base["Parameters"])
                merged.update(value)
                base["Parameters"] = merged
            else:
                base[key] = value
        return base

    return _factory


@pytest.fixture
def module_context_factory(params_factory: Callable[..., Dict[str, Any]]) -> Callable[..., ModuleContext]:
    """Factory that produces ModuleContext instances with overridable params."""

    def _factory(**param_overrides: Any) -> ModuleContext:
        params = params_factory(**param_overrides)
        return ModuleContext(params=params)

    return _factory


@pytest.fixture
def spy_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], CallSpy]:
    """Patch helper: given a dotted path, replace the attribute with a CallSpy."""

    def _factory(target_path: str) -> CallSpy:
        module_path, attr = target_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        spy = CallSpy()
        monkeypatch.setattr(module, attr, spy)
        return spy

    return _factory


_EEG_DESCRIPTIONS: Dict[str, str] = {}
_SHOW_EEG_DESCRIPTIONS = False
_EEG_TARGETS: Dict[str, Tuple[str, ...]] = {
    "tests/core/test_context.py::test_append_data_initializes_and_stacks": ("ModuleContext.append_data",),
    "tests/core/test_context.py::test_append_data_ignores_none": ("ModuleContext.append_data",),
    "tests/core/test_context.py::test_reset_data_buffer_and_get_data": (
        "ModuleContext.reset_data_buffer",
        "ModuleContext.get_data",
    ),
    "tests/core/test_context.py::test_service_registry": (
        "ModuleContext.attach_service",
        "ModuleContext.get_service",
    ),
    "tests/core/test_context.py::test_flush_data_to_params_updates_field": ("ModuleContext.flush_data_to_params",),
    "tests/core/test_context.py::test_flush_data_to_params_noop_when_buffer_empty": ("ModuleContext.flush_data_to_params",),
    "tests/core/test_context.py::test_release_device_disconnects_and_clears": ("ModuleContext.release_device",),
    "tests/core/test_context.py::test_release_device_swallows_disconnect_errors": ("ModuleContext.release_device",),
    "tests/modules/test_alpha_module.py::test_collect_measurements_runs_live_loop": (
        "AlphaModule.collect_measurements",
        "AlphaModule._ensure_array",
        "AlphaModule._apply_reference",
    ),
    "tests/modules/test_alpha_module.py::test_ensure_array_columnizes_vectors": ("AlphaModule._ensure_array",),
    "tests/modules/test_alpha_module.py::test_apply_reference_actichamp_excludes_trigger": (
        "AlphaModule._apply_reference",
    ),
    "tests/modules/test_alpha_module.py::test_apply_reference_unicorn_limits_to_first_eight_channels": (
        "AlphaModule._apply_reference",
    ),
    "tests/modules/test_assr_module.py::test_collect_measurements_chunks_recording_time": (
        "AssrModule.collect_measurements",
    ),
    "tests/modules/test_assr_module.py::test_collect_measurements_sets_aux_zero_for_non_actichamp": (
        "AssrModule.collect_measurements",
    ),
    "tests/modules/test_base_module.py::test_should_run_is_case_insensitive": ("ModuleBase.should_run",),
    "tests/modules/test_base_module.py::test_connect_noop_when_method_not_supported": ("ModuleBase.connect",),
    "tests/modules/test_base_module.py::test_connect_executes_lifecycle_when_method_matches": ("ModuleBase.connect",),
    "tests/modules/test_base_module.py::test_collect_skips_when_method_not_supported": ("ModuleBase.collect",),
    "tests/modules/test_base_module.py::test_collect_triggers_measurement_and_evaluator": (
        "ModuleBase.collect",
        "ModuleBase.post_collect",
    ),
    "tests/modules/test_base_module.py::test_execute_runs_connect_and_collect": ("ModuleBase.execute",),
    "tests/modules/test_base_module.py::test_ensure_device_creates_and_attaches": ("ModuleBase.ensure_device",),
    "tests/modules/test_base_module.py::test_ensure_device_reuses_existing_device": ("ModuleBase.ensure_device",),
    "tests/modules/test_base_module.py::test_ensure_device_respects_uses_device_toggle": ("ModuleBase.ensure_device",),
    "tests/modules/test_base_module.py::test_require_device_raises_when_missing": ("ModuleBase.require_device",),
    "tests/modules/test_base_module.py::test_require_device_returns_existing_device": ("ModuleBase.require_device",),
    "tests/modules/test_bci_module.py::test_validate_parameters_normalizes_stimfreq": ("BciModule._validate_parameters",),
    "tests/modules/test_bci_module.py::test_validate_parameters_raises_on_missing_fields": (
        "BciModule._validate_parameters",
    ),
    "tests/modules/test_bci_module.py::test_select_recent_window_keeps_tail": ("BciModule._select_recent_window",),
    "tests/modules/test_bci_module.py::test_apply_reference_handles_unknown_device": ("BciModule._apply_reference",),
    "tests/modules/test_bci_module.py::test_resolve_detection_channels_excludes_reference_and_trigger": (
        "BciModule._resolve_detection_channels",
    ),
    "tests/modules/test_bci_module.py::test_aggregate_channels_handles_empty_selection": (
        "BciModule._aggregate_channels",
    ),
    "tests/modules/test_bci_module.py::test_resolve_frequency_range_enforces_bounds": (
        "BciModule._resolve_frequency_range",
    ),
    "tests/modules/test_bci_module.py::test_try_open_bluetooth_handles_missing_dependency": (
        "BciModule._try_open_bluetooth",
    ),
    "tests/modules/test_bci_module.py::test_try_open_bluetooth_success": (
        "BciModule._try_open_bluetooth",
    ),
    "tests/modules/test_bci_module.py::test_collect_measurements_populates_evaluation": (
        "BciModule.collect_measurements",
        "BciModule._apply_reference",
        "BciModule._select_recent_window",
    ),
    "tests/modules/test_bera_module.py::test_collect_measurements_requires_actichamp": (
        "BeraModule.collect_measurements",
    ),
    "tests/modules/test_bera_module.py::test_collect_measurements_requires_sampling_rate": (
        "BeraModule.collect_measurements",
    ),
    "tests/modules/test_bera_module.py::test_collect_measurements_streams_and_plots": (
        "BeraModule.collect_measurements",
        "BeraModule._compute_live_metrics",
    ),
    "tests/modules/test_bera_module.py::test_compute_live_metrics_pipeline": ("BeraModule._compute_live_metrics",),
    "tests/modules/test_bera_module.py::test_compute_live_metrics_returns_none_for_invalid_trigger": (
        "BeraModule._compute_live_metrics",
    ),
    "tests/modules/test_bera_module.py::test_apply_reference_masks_reference_and_trigger": (
        "BeraModule._apply_reference",
    ),
    "tests/modules/test_p300_module.py::test_collect_measurements_requires_sampling_rate": (
        "P300Module.collect_measurements",
    ),
    "tests/modules/test_p300_module.py::test_collect_measurements_streams_and_updates_plot": (
        "P300Module.collect_measurements",
        "P300Module._update_live_plot",
    ),
    "tests/modules/test_p300_module.py::test_apply_reference_actichamp_and_unicorn": ("P300Module._apply_reference",),
    "tests/modules/test_p300_module.py::test_update_live_plot_executes_pipeline": ("P300Module._update_live_plot",),
    "tests/modules/test_p300_module.py::test_update_live_plot_ignores_invalid_trigger": ("P300Module._update_live_plot",),
    "tests/modules/test_ssvep_module.py::test_collect_measurements_runs_fft": (
        "SsvepModule.collect_measurements",
        "SsvepModule._apply_reference",
    ),
    "tests/modules/test_ssvep_module.py::test_apply_reference_actichamp_and_unicorn": ("SsvepModule._apply_reference",),
    "tests/modules/test_ssvep_module.py::test_ensure_array_creates_column_vectors": ("SsvepModule._ensure_array",),
    "tests/modules/test_vep_module.py::test_collect_measurements_requires_sampling_rate": (
        "VepModule.collect_measurements",
    ),
    "tests/modules/test_vep_module.py::test_collect_measurements_runs_live_plot": (
        "VepModule.collect_measurements",
        "VepModule._update_live_plot",
    ),
    "tests/modules/test_vep_module.py::test_resolve_aux_channels_depends_on_device": (
        "VepModule._resolve_aux_channels",
    ),
    "tests/modules/test_vep_module.py::test_apply_reference_behaves_per_device": ("VepModule._apply_reference",),
    "tests/modules/test_vep_module.py::test_update_live_plot_executes_pipeline": ("VepModule._update_live_plot",),
    "tests/modules/test_vep_module.py::test_update_live_plot_exits_when_trigger_invalid": (
        "VepModule._update_live_plot",
    ),
    "tests/devices/test_factory.py::test_factory_creates_lsl_device_with_params": (
        "DeviceFactory.create",
        "LSLDevice.__init__",
    ),
    "tests/devices/test_factory.py::test_factory_creates_dummy_device_and_respects_noise": (
        "DeviceFactory.create",
        "DummyDevice.__init__",
    ),
    "tests/devices/test_factory.py::test_factory_creates_offline_device_with_inline_data": (
        "DeviceFactory.create",
        "OfflineDevice.acquire",
    ),
    "tests/devices/test_factory.py::test_factory_rejects_unknown_device": ("DeviceFactory.create",),
    "tests/evaluation/test_alpha_evaluator.py::test_alpha_evaluator_ignores_non_alpha_methods": (
        "AlphaEvaluator.evaluate",
    ),
    "tests/evaluation/test_alpha_evaluator.py::test_alpha_evaluator_requires_data": ("AlphaEvaluator.evaluate",),
    "tests/evaluation/test_alpha_evaluator.py::test_alpha_evaluator_populates_evaluation_payload": (
        "AlphaEvaluator.evaluate",
        "calc_psd_power_time",
    ),
    "tests/evaluation/test_assr_evaluator.py::test_assr_evaluator_ignores_non_assr_methods": (
        "AssrEvaluator.evaluate",
    ),
    "tests/evaluation/test_assr_evaluator.py::test_assr_evaluator_requires_data": ("AssrEvaluator.evaluate",),
    "tests/evaluation/test_assr_evaluator.py::test_assr_evaluator_populates_metrics": (
        "AssrEvaluator.evaluate",
        "AssrEvaluator._evaluate_channel",
    ),
    "tests/evaluation/test_vep_evaluator.py::test_vep_evaluator_ignores_other_methods": ("VepEvaluator.evaluate",),
    "tests/evaluation/test_vep_evaluator.py::test_vep_evaluator_requires_data": ("VepEvaluator.evaluate",),
    "tests/evaluation/test_vep_evaluator.py::test_vep_evaluator_populates_metrics": ("VepEvaluator.evaluate",),
    "tests/core/test_pipeline.py::test_run_once_executes_all_modules": ("MeasurementPipeline.run_once",),
    "tests/core/test_pipeline.py::test_run_with_hooks_iterates_until_provider_returns_none": ("MeasurementPipeline.run",),
    "tests/core/test_pipeline.py::test_default_modules_returns_expected_order": ("MeasurementPipeline.default_modules",),
    "tests/devices/test_devices.py::test_dummy_device_requires_connection": ("DummyDevice.acquire",),
    "tests/devices/test_devices.py::test_dummy_device_generates_expected_shape": ("DummyDevice.acquire",),
    "tests/devices/test_devices.py::test_offline_device_loads_from_npy": (
        "OfflineDevice.connect",
        "OfflineDevice.acquire",
    ),
    "tests/devices/test_devices.py::test_lsl_device_connects_and_acquires": (
        "LSLDevice.connect",
        "LSLDevice.acquire",
        "LSLDevice.disconnect",
    ),
    "tests/shared/test_signal.py::test_calc_fft_returns_expected_spectrum": ("calc_fft",),
    "tests/shared/test_signal.py::test_time_vector_supports_seconds_and_milliseconds": ("time_vector",),
}


def pytest_configure(config: pytest.Config) -> None:
    global _SHOW_EEG_DESCRIPTIONS
    config.addinivalue_line(
        "markers",
        "eeg(description): annotate a test with a clinician-friendly description for verbose output.",
    )
    _SHOW_EEG_DESCRIPTIONS = config.getoption("--verbose") > 0


def pytest_collection_modifyitems(items):
    for item in items:
        marker = item.get_closest_marker("eeg")
        desc = None
        if marker and marker.args:
            desc = str(marker.args[0])
        else:
            func = getattr(item, "function", None)
            if func is not None:
                desc = inspect.getdoc(func)
        if desc:
            clean = " ".join(str(desc).split())
            _EEG_DESCRIPTIONS[item.nodeid] = clean


def pytest_runtest_logreport(report):
    if report.when != "call" or not _SHOW_EEG_DESCRIPTIONS:
        return
    desc = _EEG_DESCRIPTIONS.get(report.nodeid)
    if not desc:
        return
    sep = "-" * 110
    wrapped = textwrap.fill(desc, width=100)
    wrapped = "\n".join(f"    {line}" for line in wrapped.splitlines())
    targets = _EEG_TARGETS.get(report.nodeid, ())
    targets_block = ""
    if targets:
        bullets = "\n".join(f"        • {target}" for target in targets)
        targets_block = f"\n    Targets:\n{bullets}"
    print(
        f"\n{sep}\n[EEG] {report.nodeid}\n{wrapped}{targets_block}\n"
        f"    Result: {report.outcome.upper()} in {report.duration:.3f}s\n{sep}\n"
    )
