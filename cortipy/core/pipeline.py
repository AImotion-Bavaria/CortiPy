"""High-level EEG measurement pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, MutableMapping, Optional, Sequence

from .context import ModuleContext


Params = MutableMapping[str, object]
ParamsProvider = Callable[[Optional[Params]], Optional[Params]]
ContinueDecider = Callable[[Params], bool]
SaveCallback = Callable[[Params], None]


@dataclass
class PipelineHooks:
    """Extensible callbacks used by :class:`MeasurementPipeline`."""

    params_provider: ParamsProvider
    should_continue: Optional[ContinueDecider] = None
    save_callback: Optional[SaveCallback] = None
    context_hook: Optional[Callable[["ModuleContext"], None]] = None


class MeasurementPipeline:
    """Python port of the MATLAB `MeasurementPipeline` class."""

    def __init__(
        self,
        modules: Optional[Sequence["ModuleBase"]] = None,
        hooks: Optional[PipelineHooks] = None,
    ) -> None:
        self.modules: List[ModuleBase] = list(modules) if modules is not None else self.default_modules()
        self.hooks = hooks

    def run(self) -> None:
        if self.hooks is None or self.hooks.params_provider is None:
            raise RuntimeError(
                "MeasurementPipeline requires a params_provider hook that yields parameter dictionaries."
            )

        params_provider = self.hooks.params_provider
        save_callback = self.hooks.save_callback or (lambda params: None)
        should_continue = self.hooks.should_continue or (lambda params: False)

        previous_params: Optional[Params] = None
        while True:
            params = params_provider(previous_params)
            if params is None:
                break

            context = ModuleContext(params)
            if self.hooks.context_hook is not None:
                self.hooks.context_hook(context)
            for module in self.modules:
                module.connect(context)
            for module in self.modules:
                module.collect(context)
                params = context.params

            context.release_device()
            context.flush_data_to_params()
            save_callback(params)

            previous_params = params
            if not should_continue(params):
                break

    def run_once(self, params: Params) -> Params:
        """Execute the pipeline a single time with explicit parameters."""
        context = ModuleContext(params)
        if self.hooks and self.hooks.context_hook is not None:
            self.hooks.context_hook(context)
        for module in self.modules:
            module.connect(context)
        for module in self.modules:
            module.collect(context)
            params = context.params
        context.release_device()
        context.flush_data_to_params()
        return params

    def default_modules(self) -> List["ModuleBase"]:
        from cortipy.modules.alpha import AlphaModule
        from cortipy.modules.assr import AssrModule
        from cortipy.modules.bera import BeraModule
        from cortipy.modules.bci import BciModule
        from cortipy.modules.p300 import P300Module
        from cortipy.modules.ssvep import SsvepModule
        from cortipy.modules.vep import VepModule

        return [
            AlphaModule(),
            VepModule(),
            SsvepModule(),
            AssrModule(),
            BeraModule(),
            BciModule(),
            P300Module(),
        ]


# Lazy import to avoid a circular dependency
from cortipy.modules.base import ModuleBase  # noqa: E402  isort:skip
