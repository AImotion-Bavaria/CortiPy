#!/usr/bin/env python
"""Run the SSVEP pipeline on a deterministic dummy dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortipy.core.context import ModuleContext
from cortipy.evaluation.ssvep import SsvepEvaluator
from tests.regression.dummy_params import dummy_params_ssvep


def main() -> None:
    params = dummy_params_ssvep()
    params['ReportAnalyzer'] = False
    ctx = ModuleContext(params)
    evaluator = SsvepEvaluator(show_plots=True)
    evaluator.evaluate(ctx)
    evaluation = ctx.params['Evaluation']
    summary = {k: v for k, v in list(evaluation.items())[:5]}
    print('SSVEP evaluation completed. Key metrics:')
    print(json.dumps(summary, indent=2))
    print('Close plot windows to exit.')
    plt.show()


if __name__ == '__main__':
    main()
