"""Static device / montage / field configuration for the Streamlit UI.

Extracted from apps/streamlit_app.py (modularization). Pure data + one leaf helper;
no Streamlit or app dependencies.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

SUPPORTED_EXTRA_DEVICES = ["LSL", "Offline", "Dummy"]
VIEW_OPTIONS = [
    "Session configuration",
    "Workflow",
    "Electrodes",
    "Live preview",
    "Preview",
    "Charts",
    "Saved sessions",
]
DEVICE_CONFIG_SCHEMA: Dict[str, List[Dict[str, Any]]] = {
    "UNICORN": [
        {
            "name": "UNICORNPort",
            "label": "UNICORN Port / Address",
            "kind": "text",
            "placeholder": "COM7 or /dev/tty.Unicorn-DevB",
            "help": "Enter the virtual COM port (USB/Bluetooth serial) exposed by the UNICORN.",
            "default": "",
            "aliases": ["UnicornPort", "UNICORNAddress", "UnicornAddress"],
        },
        {
            "name": "UNICORNDeviceName",
            "label": "Device Name (optional)",
            "kind": "text",
            "placeholder": "EEG-Headset-01",
            "help": "Friendly name stored alongside the recording (appears in logs).",
            "default": "",
            "aliases": ["UnicornDeviceName"],
        },
        {
            "name": "UnicornTimeout",
            "label": "Connection timeout (s)",
            "kind": "number",
            "default": 5.0,
            "min": 0.5,
            "max": 30.0,
            "step": 0.5,
            "help": "Maximum time to wait for the UNICORN stream handshake.",
            "aliases": ["UNICORNTimeout"],
        },
    ]
}
DEVICE_FIELD_ALIASES: Dict[str, List[str]] = {}
for _fields in DEVICE_CONFIG_SCHEMA.values():
    for _field in _fields:
        DEVICE_FIELD_ALIASES[_field["name"]] = _field.get("aliases", [])


def device_default_values(device: str) -> Dict[str, Any]:
    return {field["name"]: field.get("default") for field in DEVICE_CONFIG_SCHEMA.get(device, [])}

DEVICE_DEFAULT_CHANNELS = {
    "ActiCHamp": 32,
    "UNICORN": 8,
    "BIOPACK": 16,
    "LSL": 8,
    "Offline": 8,
    "Dummy": 8,
}
DEVICE_EXTRA_LABELS = {
    "ActiCHamp": ["GND"],
    "UNICORN": ["GND", "Ref"],
}

# Sampling rates actually offered per device. Others fall back to the general schema list.
# Requesting a rate the hardware can't do gets clamped by the producer, which makes a recording
# run longer than RecordingTime (see the acquire() wall-time cap) — so constrain the choices here.
DEVICE_FS_OPTIONS = {
    "ActiCHamp": ["250", "500", "1000", "2000", "5000", "10000", "25000", "50000", "100000"],
    "UNICORN": ["250"],
}

INT_FIELD_NAMES = {
    "NumberEEGChannels",
    "NumberAUXChannels",
    "ReferenceChannel",
    "TriggerChannel",
    "LivePlotCH",
    "ChannelIpsi",
    "ChannelContra",
    "TestSubjectNo",
    "RecordingTime",
    "TriggerTime",
    "RepeatMeasCount",
}
INT_FIELD_PREFIXES = ("Number",)
INT_FIELD_SUFFIXES = ("Channel", "Channels", "Trials", "Count", "No")

TEN_TWENTY_32 = [
    "Fp1", "Fpz", "Fp2", "AF3", "AFz", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1",
    "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3", "PO4", "Oz",
]
UNICORN_8 = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
BIOPACK_16 = TEN_TWENTY_32[:16]

DEVICE_POSITION_DEFAULTS = {
    "ActiCHamp": TEN_TWENTY_32,
    "UNICORN": UNICORN_8,
    "BIOPACK": BIOPACK_16,
    "LSL": UNICORN_8,
    "Offline": UNICORN_8,
    "Dummy": UNICORN_8,
}


def normalize_position_label(label: str) -> str:
    return re.sub(r"\s+", "", str(label or "").strip()).upper()


STANDARD_POSITION_ORDER: List[str] = []
for _positions in DEVICE_POSITION_DEFAULTS.values():
    for _pos in _positions:
        _normalized = normalize_position_label(_pos)
        if _normalized and _normalized not in STANDARD_POSITION_ORDER:
            STANDARD_POSITION_ORDER.append(_normalized)
for _extras in DEVICE_EXTRA_LABELS.values():
    for _label in _extras:
        _normalized = normalize_position_label(_label)
        if _normalized and _normalized not in STANDARD_POSITION_ORDER:
            STANDARD_POSITION_ORDER.append(_normalized)
if not STANDARD_POSITION_ORDER:
    STANDARD_POSITION_ORDER = [f"CH{idx+1}" for idx in range(32)]
POSITION_ANGLE_LOOKUP = {label: idx for idx, label in enumerate(STANDARD_POSITION_ORDER)}
