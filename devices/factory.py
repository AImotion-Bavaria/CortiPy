"""Device factory."""

from __future__ import annotations

from typing import Any, MutableMapping

from .base import DeviceInterface
from .dummy import DummyDevice
from .lsl import LSLDevice
from .offline import OfflineDevice


class DeviceFactory:
    """Instantiate device adapters based on the Params dictionary."""

    @staticmethod
    def create(params: MutableMapping[str, Any]) -> DeviceInterface:
        device_name = str(params.get("Device", "LSL")).lower()
        device_params = params.get("Parameters", {})

        if device_name in {"unicorn", "actichamp", "lsl"}:
            stream_name = params.get("StreamName") or device_params.get("StreamName")
            if stream_name is None:
                stream_name = "EEG"
            channel_count = device_params.get("NumberEEGChannels", 8)
            fs = device_params.get("fs", 250)
            return LSLDevice(
                stream_name=str(stream_name),
                channel_count=int(channel_count),
                sampling_rate=float(fs),
            )

        if device_name in {"dummy", "sim", "simulation"}:
            channel_count = int(device_params.get("NumberEEGChannels", 8))
            fs = float(device_params.get("fs", 250))
            noise = float(device_params.get("Noise", 0.1))
            return DummyDevice(channel_count=channel_count, sampling_rate=fs, noise=noise)

        if device_name in {"offline", "file"}:
            data = params.get("data")
            data_path = params.get("DataPath") or device_params.get("DataPath")
            sampling_rate = device_params.get("fs")
            return OfflineDevice(data=data, data_path=data_path, sampling_rate=sampling_rate)

        raise ValueError(f"Unknown device '{params.get('Device')}'.")
