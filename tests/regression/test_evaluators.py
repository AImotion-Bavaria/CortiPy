import json
import os
from pathlib import Path

import numpy as np
import pytest

from cortipy.evaluation.alpha import AlphaEvaluator
from cortipy.evaluation.assr import AssrEvaluator
from cortipy.evaluation.bera import BeraEvaluator
from cortipy.evaluation.p300 import P300Evaluator
from cortipy.evaluation.ssvep import SsvepEvaluator
from cortipy.evaluation.vep import VepEvaluator
from cortipy.core.context import ModuleContext
from tests.regression import dummy_params

RUN_REGRESSION = bool(os.environ.get("CORTIPY_RUN_REGRESSION"))
if not RUN_REGRESSION:
    pytest.skip("Set CORTIPY_RUN_REGRESSION=1 to enable regression tests.", allow_module_level=True)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"

MODULES = [
    ("alpha", dummy_params.dummy_params_alpha, AlphaEvaluator()),
    ("vep", dummy_params.dummy_params_vep, VepEvaluator()),
    ("ssvep", dummy_params.dummy_params_ssvep, SsvepEvaluator()),
    ("bera", dummy_params.dummy_params_bera, BeraEvaluator()),
    ("p300", dummy_params.dummy_params_p300, P300Evaluator()),
    ("assr", dummy_params.dummy_params_assr, AssrEvaluator()),
]


def load_fixture(name: str):
    path = FIXTURE_DIR / f"{name}_evaluation.json"
    if not path.exists():
        pytest.skip(f"Fixture {path} missing. Run tests/run_parity_tests.m with writeFixtures=true")
    return json.loads(path.read_text())


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def assert_close(expected, actual, path="root", tol=1e-9):
    if isinstance(expected, dict):
        assert set(expected) == set(actual), f"{path}: field mismatch"
        for key in expected:
            assert_close(expected[key], actual[key], f"{path}.{key}", tol)
    elif isinstance(expected, list):
        assert len(expected) == len(actual), f"{path}: length mismatch"
        for idx, (exp_item, act_item) in enumerate(zip(expected, actual)):
            assert_close(exp_item, act_item, f"{path}[{idx}]", tol)
    else:
        if expected is None or actual is None:
            assert expected == actual, f"{path}: value mismatch"
            return
        exp_arr = np.asarray(expected)
        act_arr = np.asarray(actual)
        if exp_arr.dtype == object or act_arr.dtype == object:
            assert exp_arr.tolist() == act_arr.tolist(), f"{path}: mismatch"
        else:
            np.testing.assert_allclose(exp_arr, act_arr, atol=tol, rtol=tol, err_msg=path)


@pytest.mark.parametrize("name,factory,evaluator", MODULES)
def test_evaluator_against_fixture(name, factory, evaluator):
    expected = load_fixture(name)
    params = factory()
    ctx = ModuleContext(params)
    evaluator.evaluate(ctx)
    actual = to_serializable(ctx.params["Evaluation"])
    assert_close(expected, actual, path=name)
