"""Serial/Bluetooth port discovery, with real device names on Windows.

Windows names every Bluetooth SPP port after the *driver*, not the device:

    Standardmäßige serielle über Bluetooth-Verbindung (COM12)   —  Microsoft
    Standard Serial over Bluetooth link (COM13)                 —  Microsoft

Every paired headset therefore looks identical, and nothing in the port's own description
says "UNICORN". The real name ("UN-2021.05.05") lives on the port's *parent* node in the
PnP tree — the Bluetooth device the port belongs to. This module walks that link so the
dropdown can tell two headsets apart, which is the whole point of listing them.

Kept free of Streamlit so the parsing can be tested without a browser.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

try:
    from serial.tools import list_ports

    SERIAL_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - depends on the environment
    list_ports = None
    # Swallowing this silently left the dropdown with nothing but "Manual entry" and no
    # explanation, which reads exactly like "the UI only offers one device".
    SERIAL_IMPORT_ERROR = str(exc)

LOGGER = logging.getLogger(__name__)

# A paired UNICORN carries one of these in its name. The Bluetooth name of a g.tec Unicorn
# is "UN-<date>", which is why the bare "un-" is here.
UNICORN_PORT_HINTS = ("unicorn", "un-", "g.tec", "gtec")

# Ask Windows for each serial port and the friendly name of the device it hangs off.
_WINDOWS_PNP_QUERY = r"""
$ErrorActionPreference = 'SilentlyContinue'
Get-PnpDevice -Class Ports -PresentOnly | ForEach-Object {
    $parentId = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName 'DEVPKEY_Device_Parent').Data
    $parentName = $null
    if ($parentId) { $parentName = (Get-PnpDevice -InstanceId $parentId).FriendlyName }
    [pscustomobject]@{ port = $_.FriendlyName; device = $parentName }
} | ConvertTo-Json -Compress
"""

_COM_IN_NAME = re.compile(r"\((COM\d+)\)", re.IGNORECASE)

# pyserial fills these in when it knows nothing; they say less than an empty string.
_PLACEHOLDER_DETAILS = {"n/a", "unknown", "none", "-"}


def looks_like_unicorn(*fields: Optional[str]) -> bool:
    haystack = " ".join(f for f in fields if f).lower()
    return any(hint in haystack for hint in UNICORN_PORT_HINTS)


def parse_windows_pnp_json(payload: str) -> Dict[str, str]:
    """COM port -> Bluetooth device name, from the PowerShell query above.

    ConvertTo-Json emits a bare object when there is exactly one port and a list otherwise,
    so both shapes have to be accepted. Anything malformed yields {} rather than raising:
    a port list is a convenience, and failing to build it must never break the page.
    """
    text = (payload or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        LOGGER.info("Could not parse the Windows PnP port query output")
        return {}

    entries = data if isinstance(data, list) else [data]
    names: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        port_label = str(entry.get("port") or "")
        device_name = str(entry.get("device") or "").strip()
        match = _COM_IN_NAME.search(port_label)
        if not match or not device_name:
            continue
        names[match.group(1).upper()] = device_name
    return names


def windows_bluetooth_names(timeout: float = 8.0) -> Dict[str, str]:
    """COM port -> paired Bluetooth device name. Empty off Windows, or on any failure."""
    if os.name != "nt":
        return {}
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_PNP_QUERY],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - Windows only
        LOGGER.info("Could not query Windows for Bluetooth device names: %s", exc)
        return {}
    if proc.returncode != 0:  # pragma: no cover - Windows only
        LOGGER.info("Windows PnP query exited %s: %s", proc.returncode, (proc.stderr or "")[:200])
        return {}
    return parse_windows_pnp_json(proc.stdout)


def _meaningful(value: str) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _PLACEHOLDER_DETAILS else text


def describe_port(info, bluetooth_names: Optional[Dict[str, str]] = None) -> Tuple[str, str, bool]:
    """Turn one pyserial port into (device, label, looks_like_unicorn)."""
    bluetooth_names = bluetooth_names or {}
    device = getattr(info, "device", None) or getattr(info, "name", None) or ""
    description = _meaningful(getattr(info, "description", "") or getattr(info, "product", ""))
    manufacturer = _meaningful(getattr(info, "manufacturer", ""))
    serial_no = _meaningful(getattr(info, "serial_number", "") or getattr(info, "serial", ""))
    vid = getattr(info, "vid", None)
    pid = getattr(info, "pid", None)
    usb_id = f"{vid:04X}:{pid:04X}" if isinstance(vid, int) and isinstance(pid, int) else ""

    # On Windows this is the only place the headset's actual name appears.
    bt_name = bluetooth_names.get(str(device).upper(), "")

    parts: List[str] = []
    if bt_name:
        parts.append(bt_name)
    if description and description != device and description != bt_name:
        parts.append(description)
    if manufacturer and manufacturer.lower() != "microsoft":
        # Every Windows Bluetooth SPP port claims "Microsoft"; it identifies nothing.
        parts.append(manufacturer)
    if usb_id:
        parts.append(usb_id)
    if serial_no:
        parts.append(f"SN {serial_no}")

    details = ", ".join(parts)
    label = f"{device} - {details}" if details else str(device)
    is_unicorn = looks_like_unicorn(device, description, manufacturer, serial_no, bt_name)
    if is_unicorn:
        label = f"UNICORN · {label}"
    return str(device), label, is_unicorn


def serial_port_options(bluetooth_names: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    """Every serial port the OS reports, likely UNICORNs first.

    Each paired headset is its own virtual COM/cu port, so listing all of them is what
    surfaces multiple simultaneous connections.
    """
    if list_ports is None:
        return []
    if bluetooth_names is None:
        bluetooth_names = windows_bluetooth_names()

    seen = set()
    found: List[Tuple[str, str, bool]] = []
    for info in list_ports.comports():
        device, label, is_unicorn = describe_port(info, bluetooth_names)
        if not device or device in seen:
            continue
        seen.add(device)
        found.append((device, label, is_unicorn))

    # Likely UNICORNs first, then everything else, each group alphabetical.
    found.sort(key=lambda item: (not item[2], item[0]))
    return [(device, label) for device, label, _ in found]


if __name__ == "__main__":  # pragma: no cover - operator diagnostic
    # Run on the measurement machine to see exactly what the port dropdown will show:
    #     python -m cortipy.ui_streamlit.serial_ports
    print(f"platform: {os.name}")
    if list_ports is None:
        print(f"pyserial unavailable: {SERIAL_IMPORT_ERROR}")
        raise SystemExit(1)

    bt = windows_bluetooth_names()
    print(f"bluetooth device names resolved: {bt or '(none — expected off Windows)'}")

    options = serial_port_options(bt)
    if not options:
        print("no serial ports found — pair the headset first")
        raise SystemExit(0)
    print(f"\n{len(options)} port(s):")
    for device, label in options:
        print(f"  {device:24} {label}")
