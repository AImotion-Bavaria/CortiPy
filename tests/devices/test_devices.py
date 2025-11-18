"""Unit tests for device implementations."""

from __future__ import annotations

import numpy as np
import pytest

from cortipy.devices.dummy import DummyDevice
from cortipy.devices.lsl import LSLDevice
from cortipy.devices.offline import OfflineDevice


def test_dummy_device_requires_connection():
    """DummyDevice should guard against acquisition before connect to mimic real hardware expectations."""
    device = DummyDevice(channel_count=4, sampling_rate=100, noise=0.0)
    with pytest.raises(RuntimeError):
        device.acquire(1.0)


def test_dummy_device_generates_expected_shape():
    """Once connected the DummyDevice should emit samples for EEG plus requested AUX channels."""
    device = DummyDevice(channel_count=4, sampling_rate=100, noise=0.0)
    device.connect()
    chunk = device.acquire(0.1, aux_channels=2)
    assert chunk.shape == (10, 6)


def test_offline_device_loads_from_npy(tmp_path):
    """OfflineDevice should read numpy files from disk and stream them sequentially."""
    array = np.arange(12, dtype=float).reshape(6, 2)
    npy_path = tmp_path / "recording.npy"
    np.save(npy_path, array)
    device = OfflineDevice(data_path=npy_path, sampling_rate=100)
    device.connect()
    np.testing.assert_array_equal(device.acquire(0.02), array[:2])
    np.testing.assert_array_equal(device.acquire(0.04), array[2:6])


def test_lsl_device_connects_and_acquires(monkeypatch):
    """LSLDevice should resolve the desired stream, update sampling/channel counts, and pull chunks from the inlet."""
    pulled = []

    class FakeInfo:
        def nominal_srate(self):
            return 256.0

        def channel_count(self):
            return 3

    class FakeStream:
        def info(self):
            return FakeInfo()

    class FakeStreamInlet:
        def __init__(self, stream, max_buflen=None):
            self.stream = stream
            self.max_buflen = max_buflen
            self.closed = False

        def info(self):
            return self.stream.info()

        def pull_chunk(self, timeout, max_samples):
            chunk = np.ones((max_samples, 3)).tolist()
            pulled.append(len(chunk))
            return chunk, None

        def close_stream(self):
            self.closed = True

    monkeypatch.setattr("cortipy.devices.lsl.resolve_stream", lambda name, value, timeout=None: [FakeStream()])
    monkeypatch.setattr("cortipy.devices.lsl.StreamInlet", FakeStreamInlet)

    device = LSLDevice(stream_name="EEG", channel_count=3, sampling_rate=None)
    device.connect()
    data = device.acquire(0.01)
    assert data.shape[1] == 3
    assert device.sampling_rate == 256.0
    assert pulled
    device.disconnect()
