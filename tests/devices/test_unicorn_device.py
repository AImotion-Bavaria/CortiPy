from __future__ import annotations

from cortipy.devices.factory import DeviceFactory, normalize_unicorn_port
from cortipy.devices.unicorn import UnicornDevice


def test_normalize_unicorn_port_strips_windows_description() -> None:
    assert normalize_unicorn_port("COM7 - Standard Serial over Bluetooth link") == "COM7"
    assert normalize_unicorn_port("com12 saved") == "COM12"
    assert normalize_unicorn_port("/dev/tty.Unicorn-DevB") == "/dev/tty.Unicorn-DevB"


def test_device_factory_uses_normalized_unicorn_port() -> None:
    device = DeviceFactory.create(
        {
            "Device": "UNICORN",
            "Parameters": {
                "UNICORNPort": "COM8 - g.tec UNICORN",
                "fs": 250,
            },
        }
    )

    assert isinstance(device, UnicornDevice)
    assert device.port == "COM8"


def test_unicorn_connect_retries_start_ack(monkeypatch) -> None:
    serial_instances = []

    class FakeSerial:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.is_open = True
            self.writes = []
            self.reads = [b"bad", UnicornDevice.START_RESPONSE]
            serial_instances.append(self)

        def reset_input_buffer(self):
            pass

        def reset_output_buffer(self):
            pass

        def write(self, data):
            self.writes.append(data)

        def flush(self):
            pass

        def read(self, size):
            del size
            return self.reads.pop(0) if self.reads else UnicornDevice.START_RESPONSE

        def close(self):
            self.is_open = False

    monkeypatch.setattr("cortipy.devices.unicorn.serial.Serial", FakeSerial)
    monkeypatch.setattr("cortipy.devices.unicorn.time.sleep", lambda _seconds: None)

    device = UnicornDevice("COM7", timeout=0.5)
    device.connect()

    fake = serial_instances[0]
    assert fake.kwargs["port"] == "COM7"
    assert fake.kwargs["baudrate"] == UnicornDevice.BAUD_RATE
    assert fake.writes == [UnicornDevice.START_ACQ, UnicornDevice.START_ACQ]
    assert device._serial is fake
