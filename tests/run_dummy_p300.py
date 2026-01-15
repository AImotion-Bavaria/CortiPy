#!/usr/bin/env python
"""Run the P300 pipeline on a deterministic dummy dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortipy.core.context import ModuleContext  # noqa: E402
from cortipy.evaluation.p300 import P300Evaluator  # noqa: E402
from tests.regression.dummy_params import dummy_params_p300  # noqa: E402


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def main() -> None:
    params = dummy_params_p300()
    params['ReportAnalyzer'] = False
    ctx = ModuleContext(params)
    evaluator = P300Evaluator(show_plots=True)
    evaluator.evaluate(ctx)
    evaluation = ctx.params['Evaluation']
    summary = {k: v for k, v in list(evaluation.items())[:5]}
    print('P300 evaluation completed. Key metrics:')
    print(json.dumps(summary, indent=2, default=_json_default))
    print('Close plot windows to exit.')
    plt.show()


if __name__ == '__main__':
    main()
