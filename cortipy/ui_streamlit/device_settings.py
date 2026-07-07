"""Device-specific settings panels for the Streamlit UI."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

try:
    from serial.tools import list_ports
except Exception:  # pragma: no cover
    list_ports = None

from cortipy.ui_streamlit.constants import DEVICE_CONFIG_SCHEMA, device_default_values
from cortipy.ui_streamlit.fields import coerce_number

MANUAL_UNICORN_PORT_OPTION = "__manual_unicorn_port__"


def serial_port_options() -> List[tuple[str, str]]:
    if list_ports is None:
        return []
    seen = set()
    options: List[tuple[str, str]] = []
    for info in list_ports.comports():
        device = getattr(info, "device", None) or getattr(info, "name", None)
        if not device or device in seen:
            continue
        description = getattr(info, "description", "") or getattr(info, "product", "")
        manufacturer = getattr(info, "manufacturer", "")
        serial_no = getattr(info, "serial_number", "") or getattr(info, "serial", "")
        vid = getattr(info, "vid", None)
        pid = getattr(info, "pid", None)
        usb_id = f"{vid:04X}:{pid:04X}" if isinstance(vid, int) and isinstance(pid, int) else ""
        details_parts = []
        if description and description != device:
            details_parts.append(description)
        if manufacturer:
            details_parts.append(manufacturer)
        if usb_id:
            details_parts.append(usb_id)
        if serial_no:
            details_parts.append(f"SN {serial_no}")
        details = ", ".join(details_parts)
        label = f"{device} - {details}" if details else device
        options.append((device, label))
        seen.add(device)
    return sorted(options, key=lambda item: item[0])


def render_unicorn_port_input(target, field: Dict[str, Any], current: Any, key: str) -> str:
    port_entries = serial_port_options()
    labels = {port: label for port, label in port_entries}
    options: List[str] = [port for port, _ in port_entries]

    current_str = str(current) if current not in (None, "") else ""
    if current_str and current_str not in options:
        options.append(current_str)
        labels[current_str] = f"{current_str} (saved)"

    options.append(MANUAL_UNICORN_PORT_OPTION)
    default_choice = current_str if current_str in options else options[0]

    selection = target.selectbox(
        field["label"],
        options=options,
        index=options.index(default_choice) if options else 0,
        format_func=lambda value: "Manual entry" if value == MANUAL_UNICORN_PORT_OPTION else labels.get(value, value),
        help=field.get("help"),
        key=key,
    )

    if selection == MANUAL_UNICORN_PORT_OPTION:
        manual_value = target.text_input(
            "Custom UNICORN port",
            value=current_str if current_str and current_str not in labels else "",
            help="Enter the UNICORN serial/Bluetooth port manually.",
            key=f"{key}_manual",
        )
        return manual_value.strip()
    return selection


def render_device_config(device: str) -> Dict[str, Any]:
    schema = DEVICE_CONFIG_SCHEMA.get(device)
    device_forms = st.session_state.setdefault("device_forms", {})
    form_state = device_forms.setdefault(device, device_default_values(device))

    with st.expander(f"{device} device settings", expanded=True):
        if not schema:
            st.info(
                f"No device-specific settings are available for {device}. "
                "Use the general session fields and electrode editor for this device."
            )
            return dict(form_state)

        cols = st.columns(2)
        for idx, field in enumerate(schema):
            target = cols[idx % 2]
            key = f"device_{device}_{field['name']}"
            current = form_state.get(field["name"], field.get("default"))
            if device == "UNICORN" and field["name"] == "UNICORNPort":
                value = render_unicorn_port_input(target, field, current, key)
            elif field["kind"] == "number":
                fallback = field.get("default", 0.0)
                numeric = coerce_number(current)
                value_default = float(numeric if numeric is not None else fallback or 0.0)
                kwargs: Dict[str, Any] = {
                    "value": value_default,
                    "step": float(field.get("step", 0.5)),
                    "help": field.get("help"),
                    "key": key,
                }
                if field.get("min") is not None:
                    kwargs["min_value"] = float(field["min"])
                if field.get("max") is not None:
                    kwargs["max_value"] = float(field["max"])
                value = target.number_input(field["label"], **kwargs)
            else:
                value = target.text_input(
                    field["label"],
                    value=str(current or ""),
                    help=field.get("help"),
                    placeholder=field.get("placeholder"),
                    key=key,
                )
                value = value.strip()
            form_state[field["name"]] = value
    return dict(form_state)
