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

# Ask Windows for each serial port, the friendly name of the device it hangs off (its PnP
# parent), and the bus-reported device description. Force UTF-8 so German driver names decode
# on any console codepage. We take whichever of parent/bus name is not a generic driver label.
_WINDOWS_PNP_QUERY = r"""
$ErrorActionPreference = 'SilentlyContinue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$out = Get-PnpDevice -Class Ports -PresentOnly | ForEach-Object {
    $inst = $_.InstanceId
    $parentId = (Get-PnpDeviceProperty -InstanceId $inst -KeyName 'DEVPKEY_Device_Parent').Data
    $parentName = $null
    if ($parentId) { $parentName = (Get-PnpDevice -InstanceId $parentId).FriendlyName }
    $busName = (Get-PnpDeviceProperty -InstanceId $inst -KeyName 'DEVPKEY_Device_BusReportedDeviceDesc').Data
    [pscustomobject]@{ port = $_.FriendlyName; device = $parentName; busName = $busName }
}
$out | ConvertTo-Json -Compress
"""

_POWERSHELL_CANDIDATES = ("powershell", "pwsh")

_COM_IN_NAME = re.compile(r"\((COM\d+)\)", re.IGNORECASE)

# Generic Bluetooth-SPP driver labels that name the *driver*, not the paired device. When the
# PnP parent carries one of these too, it is useless — fall back to the bus-reported name.
_GENERIC_NAME = re.compile(
    r"standard.*bluetooth|bluetooth.*(link|verbindung)|serielle?\s+über\s+bluetooth"
    r"|standardm|rfcomm|^microsoft$",
    re.IGNORECASE,
)

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
        match = _COM_IN_NAME.search(port_label)
        if not match:
            continue
        # Prefer the PnP parent's name; if that is a generic driver label (or missing), fall
        # back to the bus-reported device description before giving up on this port.
        device_name = _first_real_name(entry.get("device"), entry.get("busName"))
        if not device_name:
            continue
        names[match.group(1).upper()] = device_name
    return names


def _first_real_name(*candidates: Optional[str]) -> str:
    """First candidate that is a real device name, not empty and not a generic driver label."""
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and not _GENERIC_NAME.search(text):
            return text
    return ""


def _run_powershell(script: str, timeout: float):
    """Run a PowerShell script decoded as UTF-8. Returns (stdout, stderr, returncode, exe).

    Tries Windows PowerShell then PowerShell 7. On total failure returns (None, error, None, None).
    """
    last_error = "no PowerShell interpreter found"
    for exe in _POWERSHELL_CANDIDATES:
        try:
            proc = subprocess.run(
                [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except FileNotFoundError:
            last_error = f"'{exe}' not found on PATH"
            continue
        except Exception as exc:  # pragma: no cover - Windows only
            return None, str(exc), None, exe
        return proc.stdout, proc.stderr, proc.returncode, exe
    return None, last_error, None, None


def windows_bluetooth_names(timeout: float = 8.0) -> Dict[str, str]:
    """COM port -> paired Bluetooth device name. Empty off Windows, or on any failure."""
    if os.name != "nt":
        return {}
    stdout, stderr, returncode, exe = _run_powershell(_WINDOWS_PNP_QUERY, timeout)
    if stdout is None:
        LOGGER.info("Could not query Windows for Bluetooth device names: %s", stderr)
        return {}
    if returncode not in (0, None) and not stdout.strip():
        LOGGER.info("Windows PnP query (%s) exited %s: %s", exe, returncode, (stderr or "")[:200])
        return {}
    # ConvertTo-Json may still have produced usable output even on a nonzero exit, so parse it.
    return parse_windows_pnp_json(stdout)


def diagnose(timeout: float = 8.0) -> Dict[str, object]:
    """Collect everything needed to debug Windows port-name resolution.

    Returns the raw PowerShell stdout/stderr/return code, the parsed COM->name map, and the
    ports pyserial reports. Meant to be printed on the measurement machine and shared.
    """
    report: Dict[str, object] = {"platform": os.name, "pyserial_error": SERIAL_IMPORT_ERROR}
    if os.name == "nt":
        stdout, stderr, returncode, exe = _run_powershell(_WINDOWS_PNP_QUERY, timeout)
        report["powershell_exe"] = exe
        report["powershell_returncode"] = returncode
        report["powershell_stdout"] = stdout
        report["powershell_stderr"] = stderr
        report["parsed_names"] = parse_windows_pnp_json(stdout or "")
    ports = []
    if list_ports is not None:
        for info in list_ports.comports():
            ports.append({
                "device": getattr(info, "device", None),
                "description": getattr(info, "description", None),
                "manufacturer": getattr(info, "manufacturer", None),
                "hwid": getattr(info, "hwid", None),
            })
    report["pyserial_ports"] = ports
    return report


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

    # Likely UNICORNs first, then everything else. Sort COM ports numerically (COM2 before
    # COM10, not lexically), falling back to the name for non-COM devices (/dev/…).
    def _port_key(item):
        device = item[0]
        match = re.search(r"(\d+)$", device)
        return (not item[2], 0 if match else 1, int(match.group(1)) if match else 0, device.lower())

    found.sort(key=_port_key)
    return [(device, label) for device, label, _ in found]


if __name__ == "__main__":  # pragma: no cover - operator diagnostic
    # Run on the measurement machine to see exactly what the port dropdown will show, and —
    # if names are missing — WHY. Paste the whole output when reporting a problem:
    #     python -m cortipy.ui_streamlit.serial_ports
    report = diagnose()
    print(f"platform: {report['platform']}")
    if report.get("pyserial_error"):
        print(f"pyserial unavailable: {report['pyserial_error']}")
        raise SystemExit(1)

    if report["platform"] == "nt":
        print(f"\n--- Windows PnP name query ---")
        print(f"powershell: {report.get('powershell_exe')}  (exit {report.get('powershell_returncode')})")
        stderr = (report.get("powershell_stderr") or "").strip()
        if stderr:
            print(f"stderr: {stderr[:500]}")
        stdout = (report.get("powershell_stdout") or "").strip()
        print(f"raw stdout: {stdout[:1500] or '(empty)'}")
        print(f"parsed COM -> name: {report.get('parsed_names') or '(none)'}")

    print(f"\n--- pyserial ports ---")
    for port in report.get("pyserial_ports", []):
        print(f"  {port['device']}: desc={port['description']!r} mfr={port['manufacturer']!r} hwid={port['hwid']!r}")

    bt = report.get("parsed_names") if report["platform"] == "nt" else {}
    options = serial_port_options(bt or {})
    print(f"\n--- dropdown will show ({len(options)}) ---")
    for device, label in options:
        print(f"  {device:24} {label}")
    if not options:
        print("  (no serial ports — pair the headset first)")
