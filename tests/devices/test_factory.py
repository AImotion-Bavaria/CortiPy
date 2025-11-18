"""Unit tests for DeviceFactory outputs."""

from __future__ import annotations

import numpy as np
import pytest

from cortipy.devices.dummy import DummyDevice
from cortipy.devices.factory import DeviceFactory
from cortipy.devices.lsl import LSLDevice
from cortipy.devices.offline import OfflineDevice


def test_factory_creates_lsl_device_with_params():
    """DeviceFactory should build an LSLDevice reflecting the provided stream, channel count, and sampling rate."""
    params = {
        "Device": "lsl",
        "StreamName": "NeuroEEG",
        "Parameters": {
            "NumberEEGChannels": 16,
            "fs": 500,
        },
    }

    device = DeviceFactory.create(params)

    assert isinstance(device, LSLDevice)
    assert device.stream_name == "NeuroEEG"
    assert device.channel_count == 16
    assert device.sampling_rate == 500


def test_factory_creates_dummy_device_and_respects_noise():
    """When clinicians select the Dummy device the factory should honor the requested channel count, sampling rate, and noise."""
    params = {
        "Device": "Dummy",
        "Parameters": {
            "NumberEEGChannels": 4,
            "fs": 200,
            "Noise": 0.5,
        },
    }

    device = DeviceFactory.create(params)

    assert isinstance(device, DummyDevice)
    assert device.channel_count == 4
    assert device.sampling_rate == 200
    assert device.noise == 0.5


def test_factory_creates_offline_device_with_inline_data(tmp_path):
    """Offline selections should yield OfflineDevice instances that consume prerecorded numpy arrays."""
    data = np.arange(20, dtype=float).reshape(10, 2)
    params = {
        "Device": "Offline",
        "data": data,
        "Parameters": {"fs": 100},
    }

    device = DeviceFactory.create(params)

    assert isinstance(device, OfflineDevice)
    device.connect()
    device.sampling_rate = 100
    np.testing.assert_array_equal(device.acquire(0.05), data[:5])


def test_factory_rejects_unknown_device():
    """Supplying an unknown device name should raise a clear ValueError so operators can fix their configuration."""
    with pytest.raises(ValueError):
        DeviceFactory.create({"Device": "UnknownDevice"})
