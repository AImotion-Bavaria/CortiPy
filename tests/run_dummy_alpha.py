#!/usr/bin/env python
"""Run the ALPHA pipeline using the bundled Dummy_Alpha MAT sample."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.io.matlab._mio5_params import mat_struct

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortipy.core.context import ModuleContext  # noqa: E402
from cortipy.evaluation.alpha import AlphaEvaluator  # noqa: E402
from tests.regression.dummy_params import dummy_params_alpha  # noqa: E402


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


_DEFAULT_DUMMY_ALPHA_PATH = Path(
    "/Users/jonasheinzmann/Downloads/EEG_Analysis_Tool-dev-merge_main_cortim/DummyData/Dummy_Alpha.mat"
)


def _mat_to_params(path: Path) -> dict:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "Params" not in mat:
        raise ValueError(f"MAT file {path} does not contain a 'Params' struct.")
    return _convert_mat_struct(mat["Params"])


def _convert_mat_struct(obj: Any) -> Any:
    if isinstance(obj, mat_struct):
        return {name: _convert_mat_struct(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [_convert_mat_struct(item) for item in obj.flat]
    return obj


def load_dummy_alpha_params() -> dict:
    path = Path(os.environ.get("CORTIPY_DUMMY_ALPHA_PATH", _DEFAULT_DUMMY_ALPHA_PATH))
    if path.exists():
        return _mat_to_params(path)

    print(
        f"Dummy Alpha MAT file not found at {path}. "
        "Falling back to generated dummy parameters."
    )
    return dummy_params_alpha()


def main() -> None:
    params = load_dummy_alpha_params()
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
