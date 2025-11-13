"""cortipy – Python EEG acquisition and analysis toolkit."""

from __future__ import annotations

from cortipy.core.context import ModuleContext
from cortipy.core.pipeline import MeasurementPipeline

__all__ = [
    "MeasurementPipeline",
    "ModuleContext",
]

__version__ = "0.1.0"
