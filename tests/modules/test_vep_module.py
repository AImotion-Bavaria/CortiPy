"""Unit tests for VepModule."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from cortipy.modules.vep import VepModule


def test_collect_measurements_requires_sampling_rate(module_context_factory, fake_device):
    """VEP module should gate execution until sampling frequency is defined. Without fs, the latency windows for visual responses would be nonsense, so the module halts immediately."""
    module = VepModule()
    context = module_context_factory(
        Method="VEP",
        Device="ActiCHamp",
        Parameters={"fs": 0},
    )
    context.device = fake_device
    with pytest.raises(ValueError):
        module.collect_measurements(context)


def test_collect_measurements_runs_live_plot(module_context_factory, fake_device, monkeypatch):
    """Live VEP sessions must stream data, honor AUX settings, and feed averaged plots. The test mimics a clinician’s short run and proves data reaches both the buffer and visualization pipeline."""
    module = VepModule()
    context = module_context_factory(
        Method="VEP",
        Device="ActiCHamp",
        Parameters={"fs": 200.0, "RecordingTime": 6.0, "NumberAUXChannels": 2},
    )
    device = fake_device
    prime = np.zeros((2, 4))
    chunk = np.ones((3, 4))
    device.prime_return = prime
    device.acquire_returns = deque([chunk])
    context.device = device

    calls = []
    monkeypatch.setattr(VepModule, "_update_live_plot", lambda self, params, data, fs: calls.append(data.copy()))

    module.collect_measurements(context)

    expected = np.vstack([prime, chunk])
    np.testing.assert_array_equal(context.data_buffer, expected)
    np.testing.assert_array_equal(context.params["data"], expected)
    assert device.prime_calls == [(module.first_second_duration, 2)]
    assert device.acquire_calls == [(module.time_step, 2)]
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], expected)


def test_resolve_aux_channels_depends_on_device(module_context_factory):
    """Only ActiCHamp rigs should expose auxiliary channels for VEP measurement. The helper confirms that AUX counts drop to zero on devices that lack those ports."""
    module = VepModule()
    params = {"Device": "ActiCHamp", "Parameters": {"NumberAUXChannels": 3}}
    assert module._resolve_aux_channels(params) == 3
    params_other = {"Device": "UNICORN", "Parameters": {"NumberAUXChannels": 5}}
    assert module._resolve_aux_channels(params_other) == 0


def test_apply_reference_behaves_per_device():
    """VEP referencing subtracts reference leads and gracefully crops UNICORN data. That keeps the live average focused on EEG channels while leaving the trigger untouched."""
    module = VepModule()
    params = {"Device": "ActiCHamp", "Parameters": {"ReferenceChannel": 1, "TriggerChannel": 3}}
    data = np.array([[5.0, 7.0, 9.0], [2.0, 4.0, 6.0]])
    referenced = module._apply_reference(params, data)
    np.testing.assert_array_equal(referenced[:, 0], np.zeros(2))
    np.testing.assert_array_equal(referenced[:, 1], data[:, 1] - data[:, 0])
    np.testing.assert_array_equal(referenced[:, 2], data[:, 2])

    params_unicorn = {"Device": "UNICORN", "Parameters": {}}
    unicorn = np.arange(40, dtype=float).reshape(5, 8)
    cropped = module._apply_reference(params_unicorn, unicorn)
    np.testing.assert_array_equal(cropped, unicorn[:, :8])


def test_update_live_plot_executes_pipeline(monkeypatch):
    """VEP plot update should reference, filter, trigger, segment, average, and plot the waveform. Each phase corresponds to what clinicians see while checking visual evoked responses on screen."""
    module = VepModule()
    params = {
        "Device": "ActiCHamp",
        "Parameters": {"ReferenceChannel": 1, "TriggerChannel": 3, "LivePlotCH": 1, "edge": "r"},
    }
    data = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, 6.0, 1.0],
            [2.0, 5.0, 0.0],
        ]
    )
    fs = 200.0

    monkeypatch.setattr("cortipy.modules.vep.filter_vep", lambda signal, cutoff, fs_in: (signal + 1.0).ravel())
    edge_capture = {}

    def fake_trigger(d, fs_in, trig_idx, max_time, edge="b"):
        edge_capture["value"] = edge
        return d + 2.0

    segments = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    monkeypatch.setattr("cortipy.modules.vep.trigger_adc", fake_trigger)
    monkeypatch.setattr("cortipy.modules.vep.seg_sig_fast", lambda *args, **kwargs: segments)

    plot_calls = []
    monkeypatch.setattr("cortipy.modules.vep.plot_live_avg_vep", lambda avg, params_in, max_time: plot_calls.append(avg))

    module._update_live_plot(params, data, fs)

    assert edge_capture["value"] == "r"
    assert len(plot_calls) == 1
    avg = np.array([4.0, 7.0])
    avg -= np.mean(avg)
    np.testing.assert_array_equal(plot_calls[0], avg)


def test_update_live_plot_exits_when_trigger_invalid(monkeypatch):
    """Invalid trigger configurations should short-circuit the live plot updater. Instead of throwing errors, the system simply skips plotting until a valid stimulus line returns."""
    module = VepModule()
    params = {"Parameters": {"TriggerChannel": 10}}
    data = np.zeros((5, 2))
    calls = []
    monkeypatch.setattr("cortipy.modules.vep.plot_live_avg_vep", lambda *args, **kwargs: calls.append(True))
    module._update_live_plot(params, data, 200.0)
    assert not calls
