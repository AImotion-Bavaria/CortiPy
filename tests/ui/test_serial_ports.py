"""Port discovery, and reading real device names out of Windows.

Windows names every Bluetooth SPP port after the driver, not the device:

    Standardmäßige serielle über Bluetooth-Verbindung (COM12)  —  Microsoft

So every paired headset looks identical and nothing says "UNICORN". The device's real name
lives on the port's parent node in the PnP tree. These tests pin that resolution down,
since the Windows path cannot be exercised from CI on Linux/macOS.
"""

from __future__ import annotations

import json

import pytest

from cortipy.ui_streamlit.serial_ports import (
    describe_port,
    looks_like_unicorn,
    parse_windows_pnp_json,
    serial_port_options,
    windows_bluetooth_names,
)


class FakePort:
    """Stands in for pyserial's ListPortInfo."""

    def __init__(self, device, description="", manufacturer="", serial_number="", vid=None, pid=None):
        self.device = device
        self.description = description
        self.manufacturer = manufacturer
        self.serial_number = serial_number
        self.vid = vid
        self.pid = pid


class TestWindowsPnpParsing:
    # What the PowerShell query actually returns on a German Windows 11 box.
    GERMAN = json.dumps(
        [
            {"port": "Standardmäßige serielle über Bluetooth-Verbindung (COM12)", "device": "UN-2021.05.05"},
            {"port": "Standardmäßige serielle über Bluetooth-Verbindung (COM13)", "device": "UN-2019.03.11"},
            {"port": "USB Serial Port (COM3)", "device": "USB Serial Converter"},
        ]
    )

    def test_maps_each_com_port_to_its_bluetooth_device(self):
        assert parse_windows_pnp_json(self.GERMAN) == {
            "COM12": "UN-2021.05.05",
            "COM13": "UN-2019.03.11",
            "COM3": "USB Serial Converter",
        }

    def test_two_headsets_are_told_apart(self):
        names = parse_windows_pnp_json(self.GERMAN)
        # The whole point: identical port names, different devices.
        assert names["COM12"] != names["COM13"]

    def test_accepts_the_bare_object_powershell_emits_for_a_single_port(self):
        # ConvertTo-Json returns an object, not a list, when there is exactly one result.
        single = json.dumps({"port": "Standard Serial over Bluetooth link (COM7)", "device": "UN-2020.01.01"})
        assert parse_windows_pnp_json(single) == {"COM7": "UN-2020.01.01"}

    @pytest.mark.parametrize("payload", ["", "   ", "not json", "null", "[]"])
    def test_junk_yields_no_names_rather_than_raising(self, payload):
        # A port list is a convenience; failing to build one must never break the page.
        assert parse_windows_pnp_json(payload) == {}

    def test_ports_without_a_parent_device_are_skipped(self):
        payload = json.dumps([{"port": "Serial (COM9)", "device": None}])
        assert parse_windows_pnp_json(payload) == {}

    def test_entries_without_a_com_number_are_skipped(self):
        payload = json.dumps([{"port": "Some device with no port", "device": "UN-2021.05.05"}])
        assert parse_windows_pnp_json(payload) == {}

    def test_returns_nothing_off_windows(self, monkeypatch):
        monkeypatch.setattr("cortipy.ui_streamlit.serial_ports.os.name", "posix")
        assert windows_bluetooth_names() == {}


class TestPortLabelling:
    def test_windows_bluetooth_port_is_labelled_with_the_real_device(self):
        port = FakePort(
            "COM12",
            description="Standardmäßige serielle über Bluetooth-Verbindung",
            manufacturer="Microsoft",
        )
        device, label, is_unicorn = describe_port(port, {"COM12": "UN-2021.05.05"})
        assert device == "COM12"
        assert "UN-2021.05.05" in label
        assert is_unicorn is True

    def test_the_microsoft_manufacturer_is_dropped(self):
        # Every Windows Bluetooth SPP port claims "Microsoft"; it identifies nothing.
        port = FakePort("COM12", description="Standard Serial over Bluetooth link", manufacturer="Microsoft")
        _device, label, _ = describe_port(port, {"COM12": "UN-2021.05.05"})
        assert "Microsoft" not in label

    def test_without_the_parent_name_a_windows_port_is_indistinguishable(self):
        # This is the bug: no name resolution -> nothing marks it as a UNICORN.
        port = FakePort("COM12", description="Standardmäßige serielle über Bluetooth-Verbindung",
                        manufacturer="Microsoft")
        _device, _label, is_unicorn = describe_port(port, {})
        assert is_unicorn is False

    def test_a_mac_port_still_works(self):
        port = FakePort("/dev/cu.UN-2021", description="")
        device, label, is_unicorn = describe_port(port, {})
        assert device == "/dev/cu.UN-2021"
        assert is_unicorn is True
        assert label.startswith("UNICORN")

    def test_a_plain_usb_port_is_not_flagged(self):
        port = FakePort("COM3", description="USB Serial Port", manufacturer="FTDI", vid=0x0403, pid=0x6001)
        _device, label, is_unicorn = describe_port(port, {})
        assert is_unicorn is False
        assert "0403:6001" in label


class TestOrdering:
    def test_likely_unicorns_come_first(self, monkeypatch):
        ports = [
            FakePort("COM3", description="USB Serial Port", manufacturer="FTDI"),
            FakePort("COM13", description="Standard Serial over Bluetooth link", manufacturer="Microsoft"),
            FakePort("COM12", description="Standard Serial over Bluetooth link", manufacturer="Microsoft"),
        ]
        monkeypatch.setattr("cortipy.ui_streamlit.serial_ports.list_ports.comports", lambda: ports)
        options = serial_port_options({"COM12": "UN-2021.05.05", "COM13": "UN-2019.03.11"})
        assert [p for p, _ in options] == ["COM12", "COM13", "COM3"]
        assert all("UNICORN" in label for _p, label in options[:2])

    def test_every_paired_headset_is_listed(self, monkeypatch):
        ports = [
            FakePort("COM12", description="Standard Serial over Bluetooth link", manufacturer="Microsoft"),
            FakePort("COM13", description="Standard Serial over Bluetooth link", manufacturer="Microsoft"),
        ]
        monkeypatch.setattr("cortipy.ui_streamlit.serial_ports.list_ports.comports", lambda: ports)
        options = serial_port_options({"COM12": "UN-2021.05.05", "COM13": "UN-2019.03.11"})
        labels = [label for _p, label in options]
        assert len(options) == 2
        assert any("UN-2021.05.05" in x for x in labels)
        assert any("UN-2019.03.11" in x for x in labels)


@pytest.mark.parametrize("name", ["UN-2021.05.05", "Unicorn Hybrid", "g.tec UNICORN", "un-2019"])
def test_unicorn_names_are_recognised(name):
    assert looks_like_unicorn(name) is True


@pytest.mark.parametrize("name", ["USB Serial Port", "Microsoft", "Bluetooth-Verbindung"])
def test_other_names_are_not(name):
    assert looks_like_unicorn(name) is False


class TestPlaceholderDetails:
    def test_pyserials_n_a_placeholder_is_not_shown(self):
        # macOS reports description="n/a", which rendered as "/dev/cu.BT-583 - n/a".
        port = FakePort("/dev/cu.BT-583", description="n/a")
        _device, label, _ = describe_port(port, {})
        assert label == "/dev/cu.BT-583"

    def test_a_real_description_is_still_shown(self):
        port = FakePort("/dev/cu.usbserial", description="FT232R USB UART")
        _device, label, _ = describe_port(port, {})
        assert "FT232R USB UART" in label
