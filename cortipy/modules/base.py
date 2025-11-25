"""Base class shared by all measurement modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from cortipy.core.context import ModuleContext
from cortipy.devices.factory import DeviceFactory
from cortipy.evaluation.base import EvaluatorBase


class ModuleBase(ABC):
    """Python port of `ModuleBase`."""

    def __init__(
        self,
        name: str,
        supported_methods: Iterable[str],
        evaluator: EvaluatorBase | None = None,
    ) -> None:
        self.name = name
        self.supported_methods = tuple(str(m).lower() for m in supported_methods)
        self.evaluator = evaluator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def should_run(self, context: ModuleContext) -> bool:
        method = str(context.params.get("Method", "")).lower()
        return method in self.supported_methods

    def connect(self, context: ModuleContext) -> None:
        if not self.should_run(context):
            return
        self.prepare_for_execution(context)
        self.ensure_device(context)
        self.on_connected(context)

    def collect(self, context: ModuleContext) -> None:
        if not self.should_run(context):
            return
        self.collect_measurements(context)
        self.post_collect(context)

    def execute(self, context: ModuleContext) -> None:
        self.connect(context)
        self.collect(context)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def on_connected(self, context: ModuleContext) -> None:
        """Optional hook executed after the device is ready."""

    def prepare_for_execution(self, context: ModuleContext) -> None:
        context.reset_data_buffer()

    def uses_device(self, context: ModuleContext) -> bool:
        return True

    def ensure_device(self, context: ModuleContext) -> None:
        if not self.uses_device(context):
            return
        if context.device is None:
            device = self.create_device(context)
            device.connect()
            context.device = device

    def require_device(self, context: ModuleContext):
        if context.device is None:
            raise RuntimeError(f"{self.__class__.__name__} requires an attached device.")
        return context.device

    def create_device(self, context: ModuleContext):
        device = DeviceFactory.create(context.params)
        live_view = context.get_service("live_view")
        if live_view is not None:
            try:
                device = live_view.wrap_device(device, context.params)
            except Exception:
                pass
        return device

    def post_collect(self, context: ModuleContext) -> None:
        if self.evaluator is not None:
            self.evaluator.evaluate(context)

    # ------------------------------------------------------------------
    @abstractmethod
    def collect_measurements(self, context: ModuleContext) -> None:
        """Collect data for the current module and update ``context``."""
