"""Unit tests for SsvepModule."""

from __future__ import annotations

from collections import deque

import numpy as np

from cortipy.modules.ssvep import SsvepModule


def test_collect_measurements_runs_fft(module_context_factory, fake_device, spy_factory, monkeypatch):
    """SSVEP acquisition should stream data, reference channels, and keep FFT plots updating. The scenario mirrors the steady-state runs where each new block immediately refreshes the spectral display."""
    module = SsvepModule()
    context = module_context_factory(
        Method="SSVEP",
        Device="ActiCHamp",
        Parameters={"RecordingTime": 12.0, "fs": 100.0, "NumberAUXChannels": 1, "ReferenceChannel": 1},
    )
    device = fake_device
    prime = np.zeros((2, 3))
    chunk1 = np.ones((2, 3))
    chunk2 = np.full((2, 3), 2.0)
    chunk3 = np.full((1, 3), 3.0)
    device.prime_return = prime
    device.acquire_returns = deque([chunk1, chunk2, chunk3])
    context.device = device

    calc_calls = []

    def fake_calc_fft(data, fs):
        calc_calls.append((np.array(data), fs))
        freq = np.linspace(0, fs / 2.0, data.shape[0])
        spectrum = np.ones((data.shape[0], data.shape[1]))
        return spectrum, freq

    monkeypatch.setattr("cortipy.modules.ssvep.calc_fft", fake_calc_fft)
    plot_spy = spy_factory("cortipy.modules.ssvep.plot_fft_live")

    module.collect_measurements(context)

    expected = np.vstack([prime, chunk1, chunk2, chunk3])
    np.testing.assert_array_equal(context.data_buffer, expected)
    np.testing.assert_array_equal(context.params["data"], expected)

    assert fake_device.prime_calls == [(module.first_second_duration, 1)]
    assert fake_device.acquire_calls == [(module.time_step, 1), (module.time_step, 1), (1.0, 1)]
    assert len(calc_calls) == 3
    for data_ref, fs in calc_calls:
        np.testing.assert_array_equal(data_ref[:, 0], np.zeros(data_ref.shape[0]))
        assert fs == 100.0
    assert len(plot_spy.calls) == 3


def test_apply_reference_actichamp_and_unicorn():
    """SSVEP referencing handles traditional leads differently from UNICORN records. The test ensures both pipelines produce the channel layouts expected by frequency decoders."""
    module = SsvepModule()
    params = {
        "Device": "ActiCHamp",
        "Parameters": {"ReferenceChannel": 1},
    }
    data = np.array([[5.0, 7.0], [2.0, 4.0]])
    referenced = module._apply_reference(params, data)
    np.testing.assert_array_equal(referenced[:, 0], np.zeros(2))
    np.testing.assert_array_equal(referenced[:, 1], data[:, 1] - data[:, 0])

    params_unicorn = {"Device": "UNICORN", "Parameters": {}}
    unicorn_data = np.arange(30, dtype=float).reshape(5, 6)
    truncated = module._apply_reference(params_unicorn, unicorn_data)
    np.testing.assert_array_equal(truncated, unicorn_data[:, :8])


def test_ensure_array_creates_column_vectors():
    """SSVEP helper should convert 1-D waveforms into column vectors for consistency. This keeps downstream math identical regardless of whether the amplifier returns mono or multi-channel data."""
    module = SsvepModule()
    arr = module._ensure_array(np.array([1.0, 2.0, 3.0]))
    assert arr.shape == (3, 1)
