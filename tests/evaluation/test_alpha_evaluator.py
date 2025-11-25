"""Unit tests for AlphaEvaluator."""

from __future__ import annotations

import numpy as np
import pytest

from cortipy.evaluation.alpha import AlphaEvaluator, calc_psd_power_time
from cortipy.shared.signal import hann_window


def test_alpha_evaluator_ignores_non_alpha_methods(module_context_factory):
    """Alpha evaluator should exit immediately when another paradigm ran so it never mutates unrelated measurement results."""
    context = module_context_factory(Method="SSVEP")
    evaluator = AlphaEvaluator()

    evaluator.evaluate(context)

    assert "Evaluation" not in context.params


def test_alpha_evaluator_requires_data(module_context_factory):
    """Attempting to evaluate Alpha without recorded data should raise a clear error for the clinician."""
    context = module_context_factory(Method="Alpha", Parameters={"fs": 250})
    evaluator = AlphaEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(context)


def test_calc_psd_power_time_handles_short_signal():
    """Recordings shorter than the analysis window should still compute PSD/SNR without SciPy errors."""
    fs = 250
    short_signal = np.ones(fs)
    long_window = hann_window(int(round(2.0 * fs)), periodic=True)

    band_power, power_density, time_axis, freq_axis, power_matrix = calc_psd_power_time(
        short_signal, long_window, overlap_sec=0.5, fs=fs
    )

    assert time_axis.size > 0
    assert band_power.size == time_axis.size
    assert power_matrix.shape[0] == freq_axis.size
    assert power_density.shape == power_matrix.shape


def test_alpha_evaluator_populates_evaluation_payload(module_context_factory, monkeypatch):
    """When data and sampling rate are present the evaluator should build alphaPower, PSD, and statistics entries for reports."""
    params = {
        "Method": "Alpha",
        "Device": "ActiCHamp",
        "Parameters": {
            "fs": 100,
            "ReferenceChannel": 1,
            "TriggerChannel": 2,
            "NumberEEGChannels": 2,
            "Trigger": "External",
        },
    }
    context = module_context_factory(**params)
    context.params["data"] = np.ones((4, 3))

    def fake_calc_psd(*_args, **_kwargs):
        return (
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([8.0, 12.0]),
            np.ones((2, 2)),
        )

    monkeypatch.setattr("cortipy.evaluation.alpha.calc_psd_power_time", fake_calc_psd)
    monkeypatch.setattr("cortipy.evaluation.alpha.plot_psd_time", lambda *args, **kwargs: None)
    monkeypatch.setattr("cortipy.evaluation.alpha.plot_spectrogram", lambda *args, **kwargs: None)
    monkeypatch.setattr("cortipy.evaluation.alpha.plot_alpha_matrix", lambda *args, **kwargs: None)
    monkeypatch.setattr("cortipy.evaluation.alpha.nearest_indices", lambda *args, **kwargs: np.array([0]))
    monkeypatch.setattr("cortipy.evaluation.alpha.split_alpha_segments", lambda bp, *_: (bp, bp / 2.0))
    monkeypatch.setattr("cortipy.evaluation.alpha.trigger_timestamp", lambda *args, **kwargs: np.array([0.0]))
    monkeypatch.setattr("cortipy.evaluation.alpha.resolve_channel_label", lambda _channels, idx: f"CH{idx+1}")

    def fake_alpha_snr_ram(num_channels, evaluation, **_kwargs):
        evaluation["alphaSNR"] = {"numChannels": num_channels}
        return evaluation

    monkeypatch.setattr("cortipy.evaluation.alpha.alpha_snr_ram", fake_alpha_snr_ram)

    evaluator = AlphaEvaluator(show_plots=False)
    evaluator.evaluate(context)

    evaluation = context.params["Evaluation"]
    assert "alphaPower" in evaluation
    assert "PSD" in evaluation
    assert "Stat" in evaluation
    stat_channels = evaluation["Stat"]["Channel"]
    assert evaluation["alphaSNR"]["numChannels"] == len(stat_channels)
