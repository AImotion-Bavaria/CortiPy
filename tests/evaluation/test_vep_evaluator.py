"""Unit tests for VepEvaluator."""

from __future__ import annotations

import numpy as np
import pytest

from cortipy.evaluation.vep import VepEvaluator


def test_vep_evaluator_ignores_other_methods(module_context_factory):
    """VEP evaluator should return immediately when another paradigm produced the run to avoid corrupting their reports."""
    context = module_context_factory(Method="ASSR")
    evaluator = VepEvaluator()
    evaluator.evaluate(context)
    assert "Evaluation" not in context.params


def test_vep_evaluator_requires_data(module_context_factory):
    """If data is missing the evaluator must raise so clinicians know acquisition failed."""
    context = module_context_factory(Method="VEP", Parameters={"fs": 250})
    evaluator = VepEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(context)


def test_vep_evaluator_populates_metrics(module_context_factory, monkeypatch):
    """With valid data the evaluator should compute average signals, peaks, SNRs, and residual noise metrics."""
    params = {
        "Method": "VEP",
        "Device": "ActiCHamp",
        "Parameters": {"fs": 200, "ReferenceChannel": 1, "TriggerChannel": 2},
    }
    context = module_context_factory(**params)
    context.params["data"] = np.ones((10, 3))

    monkeypatch.setattr("cortipy.evaluation.vep._apply_reference", lambda data, device, block: (np.asarray(data), 0))
    monkeypatch.setattr("cortipy.evaluation.vep._filter_non_trigger_channels", lambda data, trig_idx, fs: data)
    monkeypatch.setattr("cortipy.evaluation.vep.trigger_adc", lambda data, fs, trig_idx, max_time, edge="b": data)
    segments = np.ones((5, 4, 2))
    monkeypatch.setattr("cortipy.evaluation.vep.seg_sig_fast", lambda *args, **kwargs: segments)
    monkeypatch.setattr("cortipy.evaluation.vep.vep_amplitude_latency", lambda avg, fs: {"P100": {"peak_times": np.array([100.0]), "peak_values": np.array([5.0])}, "N135": {"peak_times": np.array([135.0]), "peak_values": np.array([-3.0])}})
    monkeypatch.setattr("cortipy.evaluation.vep.t_test_grand_average", lambda *args, **kwargs: {"t": np.array([1.0]), "p": np.array([0.01]), "h": np.array([1])})
    monkeypatch.setattr("cortipy.evaluation.vep.vep_snr_general", lambda avg, fs: np.array([6.0]))
    monkeypatch.setattr("cortipy.evaluation.vep.snr_peak_metrics", lambda segments, avg, fs: (np.array([3.0]), np.array([0.2])))
    monkeypatch.setattr("cortipy.evaluation.vep.plot_vep", lambda *args, **kwargs: None)
    monkeypatch.setattr("cortipy.evaluation.vep.plot_vep_matrix", lambda *args, **kwargs: None)

    evaluator = VepEvaluator(show_plots=False)
    evaluator.evaluate(context)

    evaluation = context.params["Evaluation"]
    assert "average_signals" in evaluation
    assert "Amplitude" in evaluation
    assert "tRes" in evaluation
    assert "SNR_time" in evaluation
    assert "SNR_Peak" in evaluation
