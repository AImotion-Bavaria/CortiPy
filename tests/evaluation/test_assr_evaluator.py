"""Unit tests for AssrEvaluator."""

from __future__ import annotations

import numpy as np
import pytest

from cortipy.evaluation.assr import AssrEvaluator
from cortipy.shared.assr import SpectralResult


def test_assr_evaluator_ignores_non_assr_methods(module_context_factory):
    """ASSR evaluator should leave params untouched when another paradigm ran so reports stay isolated."""
    context = module_context_factory(Method="Alpha")
    evaluator = AssrEvaluator()

    evaluator.evaluate(context)

    assert "Evaluation" not in context.params


def test_assr_evaluator_requires_data(module_context_factory):
    """Missing EEG data should trigger a descriptive error, preventing clinicians from viewing empty ASSR metrics."""
    context = module_context_factory(Method="ASSR", Parameters={"fs": 250})
    evaluator = AssrEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(context)


def test_assr_evaluator_populates_metrics(module_context_factory, monkeypatch):
    """Given valid data and sampling parameters the evaluator should compute FFT/PSD/SNR/f-test metrics for ipsi/contra channels."""
    params = {
        "Method": "ASSR",
        "Device": "ActiCHamp",
        "Parameters": {
            "fs": 200,
            "ASSRModulationFrequency": 40.0,
            "ASSRCarrierFrequency": 1000.0,
            "ChannelIpsi": 1,
            "ChannelContra": 2,
            "ReferenceChannel": 1,
        },
    }
    context = module_context_factory(**params)
    context.params["data"] = np.ones((4, 3))

    monkeypatch.setattr("cortipy.evaluation.assr.calc_fft", lambda signal, fs: (np.array([1.0, 2.0]), np.array([0.0, 1.0])))

    def fake_psd(signal, fs):
        return SpectralResult(psd=np.ones(2), freq=np.array([0.0, 1.0]), dBpsd=np.zeros(2))

    monkeypatch.setattr("cortipy.evaluation.assr.assr_compute_psd", fake_psd)
    monkeypatch.setattr("cortipy.evaluation.assr.assr_calc_snr", lambda *args, **kwargs: 15.0)
    monkeypatch.setattr("cortipy.evaluation.assr.assr_f_test", lambda *args, **kwargs: (4.2, 3.0))
    monkeypatch.setattr("cortipy.evaluation.assr.plot_assr_spectrum", lambda *args, **kwargs: None)

    evaluator = AssrEvaluator(show_plots=False)
    evaluator.evaluate(context)

    evaluation = context.params["Evaluation"]
    assert "fft_ipsi" in evaluation
    assert "PSD_ipsi" in evaluation
    assert evaluation["SNR2_45Hz"] == 15.0
    assert "fft_contra" in evaluation
