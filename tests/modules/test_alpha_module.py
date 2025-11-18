"""Unit tests for AlphaModule."""

from __future__ import annotations

import numpy as np
import pytest

from cortipy.modules.alpha import AlphaModule


@pytest.fixture
def alpha_context(module_context_factory):
    def _factory(**param_overrides):
        params = {
            "Method": "Alpha",
            "Device": "ActiCHamp",
            "Parameters": {
                "RecordingTime": 3.0,
                "Trigger": "External",
                "TriggerTime": 1.0,
                "fs": 200,
                "ReferenceChannel": 1,
                "TriggerChannel": 3,
                "NumberAUXChannels": 2,
            },
        }
        params.update(param_overrides)
        return module_context_factory(**params)

    return _factory


def test_collect_measurements_runs_live_loop(alpha_context, spy_factory, monkeypatch, fake_device):
    """Alpha module should stream data, display live FFTs, and signal triggers for clinicians. The test simulates a short run to prove the buffer, spectrum plots, and beeps all align with the operator’s expectations."""
    module = AlphaModule()
    context = alpha_context()

    prime_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    acquire_chunk = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    device = fake_device
    device.prime_return = prime_data
    from collections import deque

    device.acquire_returns = deque([acquire_chunk])
    context.device = device

    beep_spy = spy_factory("cortipy.modules.alpha.beep")
    start_spy = spy_factory("cortipy.modules.alpha.info_start_live")
    end_spy = spy_factory("cortipy.modules.alpha.info_end_live")
    plot_spy = spy_factory("cortipy.modules.alpha.plot_fft_live")

    fft_calls: list[tuple[np.ndarray, float]] = []

    def fake_calc_fft(data: np.ndarray, fs: float):
        fft_calls.append((np.array(data), fs))
        spectrum = np.ones_like(data, dtype=float)
        freq = np.linspace(0, fs / 2.0, data.shape[0])
        return spectrum, freq

    monkeypatch.setattr("cortipy.modules.alpha.calc_fft", fake_calc_fft)

    module.collect_measurements(context)

    expected_buffer = np.vstack([prime_data, acquire_chunk])
    np.testing.assert_array_equal(context.data_buffer, expected_buffer)
    np.testing.assert_array_equal(context.params["data"], expected_buffer)

    assert device.prime_calls == [(module.first_second_duration, 2)]
    assert device.acquire_calls == [(2.0, 2)]

    assert len(fft_calls) == 1
    fft_data, fft_fs = fft_calls[0]
    assert fft_fs == 200
    np.testing.assert_array_equal(fft_data[:, 0], np.zeros(fft_data.shape[0]))
    np.testing.assert_array_equal(fft_data[:, 2], expected_buffer[:, 2])

    assert len(plot_spy.calls) == 1
    assert len(start_spy.calls) == 1
    assert len(end_spy.calls) == 1
    assert len(beep_spy.calls) == 1


def test_ensure_array_columnizes_vectors():
    """Single-channel buffers should be reshaped into column vectors to match MATLAB layouts. This preserves compatibility with routines that treat each column as a lead."""
    module = AlphaModule()
    arr = module._ensure_array(np.array([1.0, 2.0, 3.0]))
    assert arr.shape == (3, 1)


def test_apply_reference_actichamp_excludes_trigger():
    """ActiCHamp referencing must spare the trigger line while re-referencing EEG leads. It guarantees clinicians always see untouched stimulus markers alongside re-referenced EEG."""
    module = AlphaModule()
    params = {
        "Device": "ActiCHamp",
        "Parameters": {"ReferenceChannel": 1, "TriggerChannel": 3},
    }
    data = np.array([[10.0, 20.0, 30.0], [40.0, 60.0, 80.0]])
    referenced = module._apply_reference(params, data)

    np.testing.assert_array_equal(referenced[:, 0], np.zeros(2))
    np.testing.assert_array_equal(referenced[:, 1], data[:, 1] - data[:, 0])
    np.testing.assert_array_equal(referenced[:, 2], data[:, 2])


def test_apply_reference_unicorn_limits_to_first_eight_channels():
    """UNICORN recordings keep only the eight EEG channels expected by the alpha workflow. Anything beyond those channels is dropped so visualizations match the hardware’s true electrode count."""
    module = AlphaModule()
    params = {"Device": "UNICORN"}
    data = np.arange(48, dtype=float).reshape(6, 8)
    data = np.hstack([data, np.arange(12, dtype=float).reshape(6, 2)])
    result = module._apply_reference(params, data)
    np.testing.assert_array_equal(result, data[:, :8])
