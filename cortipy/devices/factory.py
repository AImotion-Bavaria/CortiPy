"""Device factory."""

from __future__ import annotations

import re
from typing import Any, MutableMapping

from .actichamp_device import ActiChampDevice
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
            port = normalize_unicorn_port(port)
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

        if device_name == "actichamp":
            fs = float(device_params.get("fs", params.get("fs", 0) or 0))
            if fs <= 0:
                raise ValueError("ActiChamp device requires 'fs' sampling rate in Params.Parameters.")

            channel_count = max(32, int(device_params.get("NumberEEGChannels", 32)))
            aux_channels = int(device_params.get("NumberAUXChannels", 0))
            trigger_channel = int(device_params.get("TriggerChannel", 0) or 0)
            if str(params.get("Method", "")).lower() == "vep":
                trigger_channel = 33
                device_params["TriggerChannel"] = trigger_channel
            if trigger_channel > channel_count:
                aux_channels = max(aux_channels, trigger_channel - channel_count)
            install_dir = (
                params.get("ActiChampPath")
                or device_params.get("ActiChampPath")
                or device_params.get("actichampPath")
            )
            include_triggers = bool(device_params.get("IncludeTriggers", True))
            use_active = bool(
                params.get("UseActiveElectrodes")
                or device_params.get("UseActiveElectrodes")
                or False
            )
            timeout = float(device_params.get("ActiChampTimeout") or params.get("ActiChampTimeout") or 10.0)

            return ActiChampDevice(
                sampling_rate=fs,
                channel_count=channel_count,
                aux_channels=aux_channels,
                install_dir=install_dir,
                timeout=timeout,
                include_triggers=include_triggers,
                use_active_electrodes=use_active,
            )

        if device_name == "lsl":
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
            channel_count = int(device_params.get("NumberEEGChannels") or params.get("NumberEEGChannels") or 8)
            channel_count = max(1, channel_count)
            fs = float(device_params.get("fs", params.get("fs", 250)))
            noise = float(device_params.get("Noise", 0.1))
            return DummyDevice(channel_count=channel_count, sampling_rate=fs, noise=noise)

        if device_name in {"offline", "file"}:
            data = params.get("data")
            data_path = params.get("DataPath") or device_params.get("DataPath")
            sampling_rate = device_params.get("fs")
            return OfflineDevice(data=data, data_path=data_path, sampling_rate=sampling_rate)

        raise ValueError(f"Unknown device '{params.get('Device')}'.")


def normalize_unicorn_port(port: Any) -> str:
    """Return a serial port token suitable for pyserial."""
    text = str(port or "").strip()
    match = re.match(r"^(COM\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text
