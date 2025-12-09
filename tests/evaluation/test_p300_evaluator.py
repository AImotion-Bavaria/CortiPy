"""Unit tests for P300Evaluator."""

from __future__ import annotations

from cortipy.core.context import ModuleContext
from cortipy.evaluation.p300 import P300Evaluator
from tests.regression.dummy_params import dummy_params_p300


def test_p300_evaluator_segments_and_populates_metrics():
    """Ensure the evaluator keeps the trigger channel intact and returns averaged signals."""
    params = dummy_params_p300()
    params["ReportAnalyzer"] = False
    context = ModuleContext(params=params)

    evaluator = P300Evaluator(show_plots=False)
    evaluator.evaluate(context)

    evaluation = context.params.get("Evaluation")
    assert evaluation is not None
    avg = evaluation.get("average_signals", {}).get("voltage")
    assert avg is not None
    assert avg.shape[1] == 3  # three EEG channels in the dummy data
    assert evaluation.get("nTargets") == 4


def test_p300_evaluator_excludes_trigger_channel_when_not_last():
    """Trigger channel location should not leak into averaged ERP plots."""
    params = dummy_params_p300()
    params["ReportAnalyzer"] = False
    params["Parameters"]["TriggerChannel"] = 2  # move trigger into the middle
    data = params["data"]
    params["data"] = data[:, [0, 3, 1, 2]]  # reorder so trigger column now sits at index 1

    context = ModuleContext(params=params)
    evaluator = P300Evaluator(show_plots=False)
    evaluator.evaluate(context)

    avg = context.params["Evaluation"]["average_signals"]["voltage"]
    assert avg.shape[1] == 3
