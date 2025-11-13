"""Generate regression reference outputs for cortipy evaluators.

These references are intended to match the MATLAB parity fixtures. Run this
script whenever evaluator logic changes and verify the diffs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from cortipy.core.context import ModuleContext
from cortipy.evaluation.alpha import AlphaEvaluator
from cortipy.evaluation.assr import AssrEvaluator
from cortipy.evaluation.bera import BeraEvaluator
from cortipy.evaluation.p300 import P300Evaluator
from cortipy.evaluation.ssvep import SsvepEvaluator
from cortipy.evaluation.vep import VepEvaluator
from tests.regression.dummy_params import (
    dummy_params_alpha,
    dummy_params_assr,
    dummy_params_bera,
    dummy_params_p300,
    dummy_params_ssvep,
    dummy_params_vep,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    return obj


def evaluate(params: Dict[str, Any], evaluator) -> Dict[str, Any]:
    ctx = ModuleContext(params)
    evaluator.evaluate(ctx)
    return ctx.params["Evaluation"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("alpha", dummy_params_alpha, AlphaEvaluator()),
        ("vep", dummy_params_vep, VepEvaluator()),
        ("ssvep", dummy_params_ssvep, SsvepEvaluator()),
        ("bera", dummy_params_bera, BeraEvaluator()),
        ("p300", dummy_params_p300, P300Evaluator()),
        ("assr", dummy_params_assr, AssrEvaluator()),
    ]
    for name, factory, evaluator in specs:
        evaluation = evaluate(factory(), evaluator)
        serializable = to_serializable(evaluation)
        out_path = OUTPUT_DIR / f"{name}_evaluation.json"
        out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
