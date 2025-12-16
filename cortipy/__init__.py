"""cortipy – Python EEG acquisition and analysis toolkit.

This package is a ground-up Python port of the original MATLAB-based
EEG Analysis Tool.  It exposes the same measurement modules (Alpha,
VEP, SSVEP, BERA, ASSR, P300, BCI) together with their shared helpers,
device abstractions, and evaluation routines.
"""

from __future__ import annotations

from .core.pipeline import MeasurementPipeline
from .core.context import ModuleContext

__all__ = [
    "MeasurementPipeline",
    "ModuleContext",
]

__version__ = "v0.1.0"
