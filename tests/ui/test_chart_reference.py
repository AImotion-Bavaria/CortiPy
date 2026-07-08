from __future__ import annotations

import numpy as np

from cortipy.ui_streamlit.charts import _chart_from_raw_data, referenced_eeg_view
from cortipy.ui_streamlit.reports import autoevaluate_if_needed


def test_referenced_eeg_view_applies_reference_without_mutating_raw() -> None:
    data = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
    params = {
        "Device": "ActiCHamp",
        "Parameters": {"NumberEEGChannels": 2, "ReferenceChannel": 2, "fs": 250},
    }

    referenced = referenced_eeg_view(params, data)

    np.testing.assert_array_equal(data, np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]]))
    np.testing.assert_array_equal(referenced, np.array([[-9.0, 0.0], [-18.0, 0.0]]))


def test_raw_chart_uses_referenced_display_values() -> None:
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    params = {
        "Device": "ActiCHamp",
        "Parameters": {"NumberEEGChannels": 2, "ReferenceChannel": 2, "fs": 250},
    }

    chart = _chart_from_raw_data("run", params, data)

    assert chart is not None
    assert chart.title == "EEG preview (reference applied)"
    np.testing.assert_array_equal(chart.series[0].y, np.array([-9.0, -18.0, -27.0]))
    np.testing.assert_array_equal(chart.series[1].y, np.array([0.0, 0.0, 0.0]))


def test_autoevaluate_does_not_store_evaluation_on_input_params(monkeypatch) -> None:
    class DummyEvaluator:
        def __init__(self, show_plots=False):
            self.show_plots = show_plots

        def evaluate(self, context):
            context.params.setdefault("Evaluation", {})["fft"] = {"freq": [0.0], "xdft": [1.0]}

    monkeypatch.setattr("cortipy.evaluation.ssvep.SsvepEvaluator", DummyEvaluator)
    params = {
        "Method": "SSVEP",
        "Device": "ActiCHamp",
        "Parameters": {"fs": 250, "NumberEEGChannels": 1, "ReferenceChannel": 1},
    }

    evaluation = autoevaluate_if_needed(params, np.ones((8, 1)))

    assert "fft" in evaluation
    assert "Evaluation" not in params
