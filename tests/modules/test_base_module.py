"""Unit tests for ModuleBase lifecycle/device flows."""

from __future__ import annotations

import pytest

from cortipy.core.context import ModuleContext
from cortipy.modules.base import ModuleBase


class InstrumentedModule(ModuleBase):
    def __init__(self, evaluator=None) -> None:
        super().__init__("Instrumented", ["alpha"], evaluator=evaluator)
        self.prepare_calls = 0
        self.on_connected_calls = 0
        self.collect_calls = 0
        self.created_devices = 0
        self.device_to_return = None
        self.uses_device_toggle = True
        self.on_connected_devices: list[object | None] = []

    def prepare_for_execution(self, context: ModuleContext) -> None:  # type: ignore[override]
        self.prepare_calls += 1
        super().prepare_for_execution(context)

    def on_connected(self, context: ModuleContext) -> None:  # type: ignore[override]
        self.on_connected_calls += 1
        self.on_connected_devices.append(context.device)

    def collect_measurements(self, context: ModuleContext) -> None:  # type: ignore[override]
        self.collect_calls += 1

    def create_device(self, context: ModuleContext):
        self.created_devices += 1
        if self.device_to_return is None:
            raise AssertionError("device_to_return must be set for this test")
        return self.device_to_return

    def uses_device(self, context: ModuleContext) -> bool:  # type: ignore[override]
        return self.uses_device_toggle


@pytest.fixture
def module(module_context_factory):
    inst = InstrumentedModule()
    inst.device_to_return = None
    return inst


def test_should_run_is_case_insensitive(module_context_factory):
    """Module gate should treat Method names case-insensitively like the MATLAB pipeline. This keeps protocol selection stable even when technicians capitalize entries differently."""
    context = module_context_factory(Method="ALPHA")
    module = InstrumentedModule()
    assert module.should_run(context) is True
    context.params["Method"] = "beta"
    assert module.should_run(context) is False


def test_connect_noop_when_method_not_supported(module_context_factory, fake_device):
    """Connect should exit quietly when the module is not scheduled for this Method. The behavior mirrors clinical workflows where only selected paradigms spin up hardware."""
    context = module_context_factory(Method="Beta")
    context.data_buffer = "existing"
    module = InstrumentedModule()
    module.device_to_return = fake_device

    module.connect(context)

    assert module.prepare_calls == 0
    assert module.on_connected_calls == 0
    assert context.device is None
    assert context.data_buffer == "existing"
    assert fake_device.connect_calls == 0


def test_connect_executes_lifecycle_when_method_matches(module_context_factory, fake_device):
    """Happy path connect resets buffers, ensures devices, and calls hooks once. This matches the expectation that preparation occurs exactly once before data capture."""
    context = module_context_factory(Method="Alpha")
    context.data_buffer = "stale"
    module = InstrumentedModule()
    module.device_to_return = fake_device

    module.connect(context)

    assert module.prepare_calls == 1
    assert context.data_buffer is None
    assert module.on_connected_calls == 1
    assert module.on_connected_devices == [fake_device]
    assert context.device is fake_device
    assert fake_device.connect_calls == 1


def test_collect_skips_when_method_not_supported(module_context_factory):
    """Collect should not run for unrelated paradigms to avoid stale state mutations. It guards against scenarios where a module would otherwise write into Params for the wrong study."""
    context = module_context_factory(Method="Other")
    module = InstrumentedModule()
    module.collect(context)
    assert module.collect_calls == 0


def test_collect_triggers_measurement_and_evaluator(module_context_factory, fake_evaluator):
    """Collect should run subclass measurement logic and invoke the evaluator hook. Clinicians rely on this to see auto-generated quality metrics immediately after acquisition."""
    context = module_context_factory(Method="Alpha")
    module = InstrumentedModule(evaluator=fake_evaluator)
    module.collect(context)
    assert module.collect_calls == 1
    assert fake_evaluator.calls == [context]


def test_execute_runs_connect_and_collect(module_context_factory, fake_device):
    """Execute should simply chain connect plus collect to mimic MATLAB orchestrator. That guarantees a single call can run the full measurement lifecycle end-to-end."""
    context = module_context_factory(Method="Alpha")
    module = InstrumentedModule()
    module.device_to_return = fake_device

    module.execute(context)

    assert module.prepare_calls == 1
    assert module.collect_calls == 1
    assert fake_device.connect_calls == 1


def test_ensure_device_creates_and_attaches(fake_device, module_context_factory):
    """ensure_device should instantiate hardware, connect, and cache it on the context. This ensures modules always share the same live device handle the operator sees."""
    context = module_context_factory()
    module = InstrumentedModule()
    module.device_to_return = fake_device

    module.ensure_device(context)

    assert module.created_devices == 1
    assert context.device is fake_device
    assert fake_device.connect_calls == 1


def test_ensure_device_reuses_existing_device(fake_device, module_context_factory):
    """ensure_device should respect an already connected device without reconnecting. It keeps the amplifier stable when multiple modules run back-to-back."""
    context = module_context_factory()
    context.device = fake_device
    module = InstrumentedModule()
    module.device_to_return = None

    module.ensure_device(context)

    assert module.created_devices == 0
    assert fake_device.connect_calls == 0


def test_ensure_device_respects_uses_device_toggle(module_context_factory, fake_device):
    """Modules that override uses_device False must skip all hardware setup steps. That enables purely software evaluations to run without touching the amplifier."""
    context = module_context_factory()
    module = InstrumentedModule()
    module.device_to_return = fake_device
    module.uses_device_toggle = False

    module.ensure_device(context)

    assert module.created_devices == 0
    assert context.device is None
    assert fake_device.connect_calls == 0


def test_require_device_raises_when_missing(module_context_factory):
    """require_device should protect modules from running without an attached interface. The loud failure prevents collecting bogus data when the cap is disconnected."""
    context = module_context_factory()
    module = InstrumentedModule()
    with pytest.raises(RuntimeError):
        module.require_device(context)


def test_require_device_returns_existing_device(module_context_factory, fake_device):
    """require_device returns the connected hardware handle for convenient reuse. Modules can confidently control the amplifier because the context acts as a single source of truth."""
    context = module_context_factory()
    context.device = fake_device
    module = InstrumentedModule()

    result = module.require_device(context)

    assert result is fake_device
