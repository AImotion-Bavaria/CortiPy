"""Unit tests for BeraModule."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from cortipy.modules.bera import BeraModule


def _make_context(module_context_factory, **parameter_overrides):
    params = {
        "Method": "BERA",
        "Device": "ActiCHamp",
        "Parameters": {
            "RecordingTime": 7.0,
            "fs": 2000.0,
            "NumberAUXChannels": 1,
            "ReferenceChannel": 1,
            "TriggerChannel": 3,
            "LivePlotCH": 1,
            "edge": "r",
        },
    }
    params["Parameters"].update(parameter_overrides)
    return module_context_factory(**params)


def test_collect_measurements_requires_actichamp(module_context_factory, fake_device):
    """BERA should refuse to run on unsupported hardware to avoid misleading auditory-response data. The guard keeps clinicians from thinking their ActiCHamp-only protocol has run on another system."""
    module = BeraModule()
    context = module_context_factory(Method="BERA", Device="UNICORN", Parameters={"fs": 1000})
    context.device = fake_device
    with pytest.raises(RuntimeError):
        module.collect_measurements(context)


def test_collect_measurements_requires_sampling_rate(module_context_factory, fake_device):
    """An explicit sampling rate is mandatory for BERA signal processing. Without it the latency windows and averaging maths would be meaningless, so the module must raise early."""
    module = BeraModule()
    context = module_context_factory(
        Method="BERA",
        Device="ActiCHamp",
        Parameters={"RecordingTime": 5.0, "fs": 0},
    )
    context.device = fake_device
    with pytest.raises(ValueError):
        module.collect_measurements(context)


def test_collect_measurements_streams_and_plots(module_context_factory, fake_device, spy_factory, monkeypatch):
    """Main BERA loop should acquire blocks, compute live metrics, and refresh clinician plots. The scenario mirrors the live view technicians expect while building up auditory responses."""
    module = BeraModule()
    context = _make_context(module_context_factory, RecordingTime=7.0)
    context.device = fake_device

    prime_data = np.zeros((4, 4))
    chunk_a = np.ones((2, 4))
    chunk_b = np.full((1, 4), 2.0)

    fake_device.prime_return = prime_data
    fake_device.acquire_returns = deque([chunk_a, chunk_b])

    start_spy = spy_factory("cortipy.modules.bera.info_start_live")
    end_spy = spy_factory("cortipy.modules.bera.info_end_live")
    plot_spy = spy_factory("cortipy.modules.bera.plot_live_avg_bera")

    metrics = [(np.array([0.1, 0.2]), 0.5, 0.8), None]
    call_count = {"value": 0}

    def fake_metrics(self, params, data, fs):
        idx = call_count["value"]
        call_count["value"] += 1
        return metrics[idx] if idx < len(metrics) else None

    monkeypatch.setattr(BeraModule, "_compute_live_metrics", fake_metrics)

    module.collect_measurements(context)

    expected = np.vstack([prime_data, chunk_a, chunk_b])
    np.testing.assert_array_equal(context.data_buffer, expected)
    np.testing.assert_array_equal(context.params["data"], expected)

    assert fake_device.prime_calls == [(module.first_second_duration, 1)]
    assert fake_device.acquire_calls == [(5.0, 1), (1.0, 1)]
    assert len(plot_spy.calls) == 1
    avg_arg = plot_spy.calls[0].args[0]
    np.testing.assert_array_equal(avg_arg, metrics[0][0])

    assert len(start_spy.calls) == 1
    assert len(end_spy.calls) == 1


def test_compute_live_metrics_pipeline(monkeypatch):
    """Live metric helper should execute referencing, filtering, triggering, segmentation, and KPI math. By stubbing each stage we confirm every transformation happens before KPIs are reported."""
    module = BeraModule()
    params = {
        "Parameters": {
            "ReferenceChannel": 1,
            "TriggerChannel": 3,
            "LivePlotCH": 2,
            "edge": "f",
            "AnaWindow": [0.002, 0.004],
        }
    }
    data = np.arange(30, dtype=float).reshape(10, 3)
    fs = 2000.0

    filter_calls = {}

    def fake_apply_reference(p, d):
        return d + 1.0

    def fake_filter(data_in, fs_in, exclude):
        filter_calls["exclude"] = exclude
        return data_in + 2.0

    def fake_trigger(data_in, fs_in, trig_idx, max_time, edge="r"):
        filter_calls["edge"] = edge
        return data_in + 3.0

    segments = np.arange(24, dtype=float).reshape(2, 4, 3)

    def fake_seg(data_in, fs_in, max_time, trig_idx):
        return segments

    def fake_prepro(data_in, *_args, **_kwargs):
        return data_in + 0.5

    def fake_residual(data_in, fs_in, window):
        return 0.25, None

    def fake_fsp(channel_segments, avg_signal, fs_in, sp_time, window):
        return None, None, 0.75, None

    monkeypatch.setattr(BeraModule, "_apply_reference", staticmethod(fake_apply_reference))
    monkeypatch.setattr("cortipy.modules.bera.filter_bera", fake_filter)
    monkeypatch.setattr("cortipy.modules.bera.trigger_adc", fake_trigger)
    monkeypatch.setattr("cortipy.modules.bera.seg_sig_fast", fake_seg)
    monkeypatch.setattr("cortipy.modules.bera.prepro", fake_prepro)
    monkeypatch.setattr("cortipy.modules.bera.residual_noise_eclipse", fake_residual)
    monkeypatch.setattr("cortipy.modules.bera.get_fsp_fmp", fake_fsp)

    result = module._compute_live_metrics(params, data, fs)

    assert result is not None
    avg_signal, rn, fmp = result
    np.testing.assert_array_equal(avg_signal, np.mean((segments[:, :, 1] + 0.5), axis=0))
    assert rn == 0.25
    assert fmp == 0.75
    assert filter_calls["exclude"] == {2, 0, data.shape[1] - 1}
    assert filter_calls["edge"] == "f"


def test_compute_live_metrics_returns_none_for_invalid_trigger():
    """Missing trigger channels should skip metric computation rather than crashing mid-session. This mimics a real-world loose cable scenario where the UI simply waits for good data."""
    module = BeraModule()
    params = {"Parameters": {"TriggerChannel": 10}}
    data = np.zeros((5, 2))
    result = module._compute_live_metrics(params, data, 1000.0)
    assert result is None


def test_apply_reference_masks_reference_and_trigger():
    """BERA referencing subtracts the reference lead while keeping trigger data untouched. It guarantees the stimulus marker remains pristine for alignment while EEG leads become common-reference."""
    module = BeraModule()
    params = {"Parameters": {"ReferenceChannel": 1, "TriggerChannel": 3}}
    data = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 9.0]])
    referenced = module._apply_reference(params, data)
    np.testing.assert_array_equal(referenced[:, 0], data[:, 0])
    np.testing.assert_array_equal(referenced[:, 1], data[:, 1] - data[:, 0])
    np.testing.assert_array_equal(referenced[:, 2], data[:, 2])
