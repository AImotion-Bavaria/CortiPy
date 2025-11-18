"""Unit tests for P300Module."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from cortipy.modules.p300 import P300Module


def test_collect_measurements_requires_sampling_rate(module_context_factory, fake_device):
    """P300 data collection must refuse to start without a defined sampling frequency. Without the correct fs value, the ERP windows would be misaligned, so the module halts early."""
    module = P300Module()
    context = module_context_factory(
        Method="P300",
        Device="ActiCHamp",
        Parameters={"fs": 0},
    )
    context.device = fake_device
    with pytest.raises(ValueError):
        module.collect_measurements(context)


def test_collect_measurements_streams_and_updates_plot(module_context_factory, fake_device, monkeypatch):
    """Live P300 acquisitions should buffer EEG and refresh ERP plots each block. This mirrors the clinician’s live view where each batch of stimuli immediately updates the averaged ERP curve."""
    module = P300Module()
    context = module_context_factory(
        Method="P300",
        Device="ActiCHamp",
        Parameters={"fs": 200.0, "RecordingTime": 5.0, "NumberAUXChannels": 1},
    )
    device = fake_device
    prime_data = np.zeros((4, 3))
    acquire_chunk = np.ones((2, 3))
    device.prime_return = prime_data
    device.acquire_returns = deque([acquire_chunk])
    context.device = device

    calls = []
    monkeypatch.setattr(P300Module, "_update_live_plot", lambda self, params, data, fs: calls.append(data.copy()))

    module.collect_measurements(context)

    expected = np.vstack([prime_data, acquire_chunk])
    np.testing.assert_array_equal(context.data_buffer, expected)
    np.testing.assert_array_equal(context.params["data"], expected)
    assert device.prime_calls == [(module.first_second_duration, 1)]
    assert device.acquire_calls == [(4.0, 1)]
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], expected)


def test_apply_reference_actichamp_and_unicorn():
    """P300 referencing logic differs for ActiCHamp versus UNICORN headsets and must stay correct. The test confirms both the reference subtraction and channel cropping rules used in practice."""
    module = P300Module()
    params_acti = {
        "Device": "ActiCHamp",
        "Parameters": {"ReferenceChannel": 1, "TriggerChannel": 3},
    }
    data = np.array([[10.0, 15.0, 20.0], [5.0, 6.0, 7.0]])
    referenced = module._apply_reference(params_acti, data)
    np.testing.assert_array_equal(referenced[:, 0], np.zeros(2))
    np.testing.assert_array_equal(referenced[:, 1], data[:, 1] - data[:, 0])

    params_unicorn = {
        "Device": "UNICORN",
        "Parameters": {},
    }
    unicorn_data = np.arange(40, dtype=float).reshape(5, 8)
    cropped = module._apply_reference(params_unicorn, unicorn_data)
    np.testing.assert_array_equal(cropped, unicorn_data[:, :8])


def test_update_live_plot_executes_pipeline(monkeypatch):
    """Plot update should reference, filter, trigger, segment, average, and visualize ERPs. Each step reflects the exact processing chain clinicians rely on when monitoring ERP quality in real time."""
    module = P300Module()
    params = {
        "Device": "ActiCHamp",
        "Parameters": {"ReferenceChannel": 1, "TriggerChannel": 3, "LivePlotCH": 1, "edge": "b"},
    }
    data = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, 5.0, 1.0],
            [2.0, 6.0, 0.0],
        ]
    )
    fs = 200.0

    monkeypatch.setattr("cortipy.modules.p300.filter_vep", lambda signal, cutoff, fs_in: (signal + 1.0).ravel())

    triggered_capture = {}

    def fake_trigger(d, fs_in, trig_idx, max_time, edge="f"):
        triggered_capture["edge"] = edge
        return d + 2.0

    segments = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    monkeypatch.setattr("cortipy.modules.p300.trigger_adc", fake_trigger)
    monkeypatch.setattr("cortipy.modules.p300.seg_sig_fast_p300", lambda *args, **kwargs: segments)

    plot_calls = []

    def fake_plot(avg_signal, params_in, max_time):
        plot_calls.append(avg_signal)

    monkeypatch.setattr("cortipy.modules.p300.plot_live_erp", fake_plot)

    module._update_live_plot(params, data, fs)

    assert triggered_capture["edge"] == "b"
    assert len(plot_calls) == 1
    avg = np.array([4.0, 7.0])  # mean of channel 0 data across segments
    avg -= np.mean(avg)
    np.testing.assert_array_equal(plot_calls[0], avg)


def test_update_live_plot_ignores_invalid_trigger(monkeypatch):
    """If trigger indexing is invalid, the P300 plotting routine should exit quietly. This matches the expectation that a missing stimulus cable simply pauses the chart instead of crashing the application."""
    module = P300Module()
    params = {"Parameters": {"TriggerChannel": 10}}
    data = np.zeros((5, 2))
    plot_spy = []
    monkeypatch.setattr("cortipy.modules.p300.plot_live_erp", lambda *args, **kwargs: plot_spy.append(True))
    module._update_live_plot(params, data, 200.0)
    assert not plot_spy
