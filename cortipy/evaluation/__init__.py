"""Module evaluation helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from .alpha import AlphaEvaluator
from .assr import AssrEvaluator
from .base import EvaluatorBase
from .bera import BeraEvaluator
from .p300 import P300Evaluator
from .ssvep import SsvepEvaluator
from .vep import VepEvaluator

__all__ = [
    "EvaluatorBase",
    "AlphaEvaluator",
    "BeraEvaluator",
    "SsvepEvaluator",
    "VepEvaluator",
    "AssrEvaluator",
    "P300Evaluator",
]

def _configure_eval_logging() -> logging.Logger:
    """Attach a shared file handler for evaluation modules."""
    logger = logging.getLogger("cortipy.evaluation")
    logger.setLevel(logging.DEBUG)
    log_path = Path(__file__).resolve().parents[2] / "streamlit_app.log"

    existing = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == log_path
    ]
    if not existing:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


_configure_eval_logging()
