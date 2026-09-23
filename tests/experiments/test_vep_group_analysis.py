from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import f as f_distribution

# The VEP group-analysis workflow moved into the CortiPy_VEP_Ear_Study submodule
# (analysis/workflow.py). Load it by path rather than as analysis.workflow: the
# package __init__ re-exports the plotting and study-level modules, which this
# file does not need. workflow.py itself only imports from cortipy.
_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / "CortiPy_VEP_Ear_Study"
    / "analysis"
    / "workflow.py"
)
if not _WORKFLOW_PATH.exists():
    pytest.skip(
        "CortiPy_VEP_Ear_Study submodule not checked out "
        "(git submodule update --init --recursive)",
        allow_module_level=True,
    )

_spec = importlib.util.spec_from_file_location("vep_study_workflow", _WORKFLOW_PATH)
_workflow = importlib.util.module_from_spec(_spec)
sys.modules["vep_study_workflow"] = _workflow
_spec.loader.exec_module(_workflow)

DEFAULT_EPOCH_SECONDS = _workflow.DEFAULT_EPOCH_SECONDS
apply_reference = _workflow.apply_reference
calculate_fsp = _workflow.calculate_fsp
calculate_fsp_by_channel = _workflow.calculate_fsp_by_channel
calculate_marker_metrics = _workflow.calculate_marker_metrics
find_trigger_positions = _workflow.find_trigger_positions
segment_measurement = _workflow.segment_measurement
to_jsonable = _workflow.to_jsonable


def test_calculate_fsp_matches_matlab_formula() -> None:
    fs = 1000.0
    sweeps = np.array(
        [
            np.linspace(0.0, 1.0, 300),
            np.linspace(0.0, 2.0, 300),
            np.linspace(0.0, 3.0, 300),
        ]
    )
    average = sweeps.mean(axis=0)
    result = calculate_fsp(sweeps, average, fs)
    signal_variance = np.var(average[49:150], ddof=1)
    sweep_variance = np.var(sweeps[:, 99], ddof=1)
    expected = sweeps.shape[0] * signal_variance / sweep_variance
    assert result["Fsp"] == expected
    assert result["p_sp"] == f_distribution.cdf(expected, 5, 2)


def test_calculate_fsp_by_channel() -> None:
    stack = np.ones((3, 300, 2), dtype=float)
    stack[:, :, 0] = np.arange(3)[:, None]
    average = stack.mean(axis=0)
    result = calculate_fsp_by_channel(stack, average, 1000.0)
    assert len(result) == 2
    assert set(result[0]) == {"Fsp", "p_sp", "Fmp", "p_mp"}


def test_apply_reference_uses_selected_one_based_channel() -> None:
    data = np.array([[10.0, 3.0], [8.0, 2.0]])
    np.testing.assert_array_equal(apply_reference(data, 2), [[7.0, 0.0], [6.0, 0.0]])


def test_trigger_locked_segmentation_excludes_trigger_from_eeg() -> None:
    trigger = np.zeros(1200)
    trigger[[10, 610]] = 1.0
    measurement = {
        "filename": "test.parquet",
        "fs": 1000.0,
        "data_uV": np.column_stack([np.arange(1200.0), np.ones(1200)]),
        "trigger_uV": trigger,
    }
    epochs, trigger_epochs = segment_measurement(measurement)
    epoch_samples = round(DEFAULT_EPOCH_SECONDS * measurement["fs"])
    assert find_trigger_positions(trigger).tolist() == [10, 610]
    assert epochs.shape == (2, epoch_samples, 2)
    assert trigger_epochs.shape == (2, epoch_samples)


def test_marker_metrics() -> None:
    result = calculate_marker_metrics(
        {"N75": {"time_ms": 75, "amplitude_uV": -2}, "P100": {"time_ms": 100, "amplitude_uV": 4}}
    )
    assert result["N75_P100_latency_difference_ms"] == 25
    assert result["N75_P100_amplitude_uV"] == 6


def test_to_jsonable_serializes_numpy_values() -> None:
    value = to_jsonable({"array": np.array([1.0, 2.0]), "scalar": np.float64(3.0)})
    json.dumps(value, allow_nan=False)
    assert value == {"array": [1.0, 2.0], "scalar": 3.0}
