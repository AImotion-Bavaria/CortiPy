#!/usr/bin/env python
"""Run the ALPHA pipeline on a deterministic dummy dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortipy.core.context import ModuleContext
from cortipy.evaluation.alpha import AlphaEvaluator
from tests.regression.dummy_params import dummy_params_alpha


def _json_safe(value: Any) -> Any:
    """Convert numpy-heavy structures into JSON-serialisable data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    params = dummy_params_alpha()
    params['ReportAnalyzer'] = False
    ctx = ModuleContext(params)
    evaluator = AlphaEvaluator(show_plots=True)
    evaluator.evaluate(ctx)
    evaluation = ctx.params['Evaluation']
    summary = {k: _json_safe(v) for k, v in list(evaluation.items())[:5]}
    print('ALPHA evaluation completed. Key metrics:')
    print(json.dumps(summary, indent=2))
    print('Close plot windows to exit.')
    plt.show()


if __name__ == '__main__':
    main()
