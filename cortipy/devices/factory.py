"""Device factory."""

from __future__ import annotations

from typing import Any, MutableMapping

from .base import DeviceInterface
from .dummy import DummyDevice
from .lsl import LSLDevice
from .offline import OfflineDevice
from .unicorn import UnicornDevice


class DeviceFactory:
    """Instantiate device adapters based on the Params dictionary."""

    @staticmethod
    def create(params: MutableMapping[str, Any]) -> DeviceInterface:
        device_name = str(params.get("Device", "LSL")).lower()
        device_params = params.get("Parameters", {})

        if device_name == "unicorn":
            port = (
                params.get("UnicornPort")
                or params.get("UnicornAddress")
                or device_params.get("UNICORNPort")
                or device_params.get("UNICORNAddress")
                or device_params.get("UnicornPort")
                or device_params.get("UnicornAddress")
            )
            if not port:
                raise ValueError("UNICORN device requires 'UnicornPort' (serial/Bluetooth COM port).")
            device_label = (
                params.get("UnicornDeviceName")
                or device_params.get("UNICORNDeviceName")
                or device_params.get("UnicornDeviceName")
                or str(port)
            )
            fs = float(device_params.get("fs", 250))
            timeout = float(device_params.get("UnicornTimeout") or params.get("UnicornTimeout") or 5.0)
            return UnicornDevice(
                port=str(port),
                device_name=str(device_label),
                sampling_rate=fs,
                timeout=timeout,
            )

        if device_name in {"actichamp", "lsl"}:
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
