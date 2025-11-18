"""Unit tests for BciModule."""

from __future__ import annotations

import builtins
from collections import deque
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cortipy.modules.bci import BciModule


def _params(**overrides):
    params = {
        "Method": "BCI",
        "Device": "ActiCHamp",
        "Parameters": {
            "fs": 10.0,
            "RecordingTime": 3.0,
            "NumberEEGChannels": 3,
            "ReferenceChannel": 1,
            "TriggerChannel": 3,
            "StimFreq": [12.0, 15.0],
            "NumberAUXChannels": 1,
        },
    }
    params["Parameters"].update(overrides)
    return params


def test_validate_parameters_normalizes_stimfreq():
    """BCI configuration should normalize stimulus frequency lists regardless of user input formats. Whether the operator enters a scalar, string, or list, the pipeline must see a clean float array."""
    module = BciModule()
    params = {"Parameters": {"fs": 1, "RecordingTime": 2, "NumberEEGChannels": 2, "ReferenceChannel": 1, "StimFreq": 15}}
    result = module._validate_parameters(params)
    assert result["Parameters"]["StimFreq"] == [15.0]

    params["Parameters"]["StimFreq"] = "10 12.5"
    result = module._validate_parameters(params)
    assert result["Parameters"]["StimFreq"] == [10.0, 12.5]


def test_validate_parameters_raises_on_missing_fields():
    """BCI setup must complain when key acquisition parameters are absent. This prevents clinicians from running a session without the fundamental sampling or channel settings."""
    module = BciModule()
    with pytest.raises(ValueError):
        module._validate_parameters({"Parameters": {"fs": 1}})


def test_select_recent_window_keeps_tail():
    """Recent-window helper should operate like a rolling buffer feeding spectral analysis. Only the most recent samples stay in the detection window, mirroring how online SSVEP decoders behave."""
    module = BciModule()
    data = np.arange(20).reshape(10, 2)
    window = module._select_recent_window(data, 4)
    np.testing.assert_array_equal(window, data[-4:])


def test_apply_reference_handles_unknown_device(monkeypatch):
    """Unknown headset models should fall back to raw EEG while warning the operator once. That keeps the system usable even with experimental headsets, while clearly flagging reduced support."""
    module = BciModule()
    params = {
        "Device": "Unknown",
        "Parameters": {"NumberEEGChannels": 2},
    }
    printed = {"count": 0}

    def fake_print(msg):
        printed["count"] += 1

    monkeypatch.setattr("builtins.print", fake_print)
    data = np.arange(6, dtype=float).reshape(3, 2)
    referenced = module._apply_reference(params, data)
    np.testing.assert_array_equal(referenced, data)
    assert printed["count"] == 1


def test_resolve_detection_channels_excludes_reference_and_trigger():
    """Detection channel selection must drop reference and trigger indices before averaging. Leaving them in would water down the SSVEP signal, so the test verifies they are removed deterministically."""
    module = BciModule()
    params_block = {"NumberEEGChannels": 4, "ReferenceChannel": 2, "TriggerChannel": 4}
    channels = module._resolve_detection_channels(params_block, total_columns=4)
    np.testing.assert_array_equal(channels, np.array([0, 2]))


def test_aggregate_channels_handles_empty_selection():
    """If no detection leads survive filtering, the module should average all channels. That fallback approximates the MATLAB behavior when operators deselect every electrode."""
    module = BciModule()
    data = np.arange(12, dtype=float).reshape(4, 3)
    result = module._aggregate_channels(data, np.array([], dtype=int))
    np.testing.assert_allclose(result, data.mean(axis=1))


def test_resolve_frequency_range_enforces_bounds():
    """Frequency bounds should auto-correct if the operator supplies an inverted range. The analyzer always needs a positive passband, even when the UI inputs are reversed."""
    module = BciModule()
    params_block = {"LowestFrequency": 5.0, "HighestFrequency": 4.0}
    low, high = module._resolve_frequency_range(params_block, np.array([12.0]))
    assert low == 5.0
    assert high > low


def test_try_open_bluetooth_handles_missing_dependency(monkeypatch, capsys):
    """Bluetooth output should gracefully degrade when the optional dependency is missing. Instead of crashing, the module explains that wireless output is unavailable and continues running."""
    module = BciModule()
    params_block = {"BluetoothDevice": "target"}

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bluetooth":
            raise ImportError("forced")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    conn, cleanup = module._try_open_bluetooth(params_block)
    captured = capsys.readouterr()
    assert conn is None and cleanup is None
    assert "Bluetooth support unavailable" in captured.out


def test_try_open_bluetooth_success(monkeypatch):
    """When the Bluetooth stack is available the module should connect to the requested target and expose a cleanup hook that closes the socket."""
    module = BciModule()
    params_block = {"BluetoothDevice": "00:11:22:33:44:55", "BluetoothChannel": 7}
    calls = {}

    class FakeSocket:
        def __init__(self, proto):
            calls["proto"] = proto
            self.connected_to = None
            self.closed = False

        def connect(self, target):
            self.connected_to = target

        def close(self):
            self.closed = True

    fake_bt = SimpleNamespace(BluetoothSocket=FakeSocket, RFCOMM=3)
    monkeypatch.setitem(sys.modules, "bluetooth", fake_bt)

    sock, cleanup = module._try_open_bluetooth(params_block)

    assert sock is not None
    assert sock.connected_to == ("00:11:22:33:44:55", 7)
    assert calls["proto"] == 3
    assert cleanup is not None
    cleanup()
    assert sock.closed is True


def test_collect_measurements_populates_evaluation(module_context_factory, fake_device, monkeypatch, spy_factory):
    """BCI acquisition should produce FFT/T²/CCA metrics, Bluetooth dispatches, and evaluation payloads. The test simulates a multi-second session to ensure spectra, predictions, and history arrays are all populated for clinicians."""
    module = BciModule()
    context = module_context_factory(**_params())
    device = fake_device
    prime_data = np.zeros((4, 3))
    acquire_chunks = [np.ones((2, 3)), np.full((2, 3), 2.0)]
    device.prime_return = prime_data
    device.acquire_returns = deque(acquire_chunks)
    context.device = device

    fft_calls = []

    def fake_calc_fft(data, fs):
        fft_calls.append((np.array(data), fs))
        freq = np.linspace(0, fs / 2.0, data.shape[0])
        spectrum = np.ones(freq.shape)
        return spectrum, freq

    freq_idx = np.array([0, 1])
    predictions = deque([np.array([0.6, 0.4]), np.array([0.3, 0.7])])
    t2_queue = deque([(np.array([1.0]), np.array([0.1])), (np.array([1.2]), np.array([0.05]))])
    cca_queue = deque([(np.array([0.8, 0.9]), np.array([12.0, 15.0])), (np.array([0.85, 0.95]), np.array([12.0, 15.0]))])

    monkeypatch.setattr("cortipy.modules.bci.calc_fft", fake_calc_fft)
    monkeypatch.setattr("cortipy.modules.bci.get_frequency_indices", lambda freq, stim: freq_idx)
    monkeypatch.setattr("cortipy.modules.bci.classify_fft", lambda *args: predictions.popleft())
    monkeypatch.setattr("cortipy.modules.bci.compute_t2circ", lambda *args: t2_queue.popleft())
    monkeypatch.setattr("cortipy.modules.bci.cca_correlations", lambda *args: cca_queue.popleft())
    monkeypatch.setattr("cortipy.modules.bci.describe_prediction", lambda *args: "predicted")
    send_spy = spy_factory("cortipy.modules.bci.send_prediction_bt")
    spectrum_calls: list[tuple[np.ndarray, np.ndarray, tuple[float, float]]] = []

    def fake_plot_spectrum(self, freq, spectrum, freq_range, params):
        spectrum_calls.append((freq, spectrum, freq_range))

    monkeypatch.setattr(BciModule, "_plot_spectrum", fake_plot_spectrum)

    module.collect_measurements(context)

    expected_rows = np.vstack([prime_data, *acquire_chunks])
    np.testing.assert_array_equal(context.data_buffer, expected_rows)
    assert context.data_buffer.shape[1] == 3

    assert len(fft_calls) == 2
    assert len(spectrum_calls) == 2
    assert spectrum_calls[0][2] == (9.0, 20.0)
    assert len(send_spy.calls) == 2

    evaluation = context.params["Evaluation"]["BCI"]
    np.testing.assert_array_equal(evaluation["StimFreq"], np.array([12.0, 15.0]))
    assert evaluation["PredictionHistory"].shape == (2, 2)
    np.testing.assert_array_equal(evaluation["PredictionTimes"], np.array([2.0, 3.0]))
    assert evaluation["T2circ"].shape == (2, 1)
    assert evaluation["CCA"]["rho"].shape == (2, 2)
