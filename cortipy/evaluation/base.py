"""Base class for evaluation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cortipy.core.context import ModuleContext


class EvaluatorBase(ABC):
    """Shared protocol for all evaluators."""

    @abstractmethod
    def evaluate(self, context: ModuleContext) -> None:
        """Mutate ``context.params`` with evaluation results."""

