"""Device-specific settings panels for the Streamlit UI."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import streamlit as st

from cortipy.ui_streamlit.constants import DEVICE_CONFIG_SCHEMA, device_default_values
from cortipy.ui_streamlit.fields import coerce_number
from cortipy.ui_streamlit.serial_ports import (
    SERIAL_IMPORT_ERROR as _SERIAL_IMPORT_ERROR,
    list_ports,
    serial_port_options,
    windows_bluetooth_names,
)

LOGGER = logging.getLogger(__name__)

MANUAL_UNICORN_PORT_OPTION = "__manual_unicorn_port__"

# Relative column widths per device field. The port carries long device paths; a timeout is
# four characters. Anything unlisted gets a sensible middle.
DEVICE_FIELD_WIDTHS: Dict[str, float] = {
    "UNICORNPort": 2.6,
    "UNICORNDeviceName": 1.8,
    "UnicornTimeout": 1.1,
}

# Placeholder occupying the narrow column that holds the port's Rescan button.
_RESCAN_SLOT = object()

_PORT_CACHE_KEY = "_serial_port_options"


def cached_serial_port_options(force: bool = False) -> List[tuple]:
    """Port list, cached across reruns.

    Resolving Bluetooth device names on Windows shells out to PowerShell, which costs a
    noticeable fraction of a second — far too slow to repeat on every Streamlit rerun.
    "Rescan" is what re-enumerates, which is also when a newly paired headset appears.
    """
    if force:
        st.session_state.pop(_PORT_CACHE_KEY, None)
    cached = st.session_state.get(_PORT_CACHE_KEY)
    if cached is not None:
        return cached
    options = serial_port_options(windows_bluetooth_names())
    st.session_state[_PORT_CACHE_KEY] = options
    return options


def render_unicorn_port_input(target, field: Dict[str, Any], current: Any, key: str, rescan_target=None) -> str:
    """Port picker plus its rescan button, on one line.

    The "N port(s) found" note folds into the field's help tooltip instead of taking its own
    line, and the rescan button sits beside the selectbox rather than above it — the panel
    used to be three stacked rows tall for what is really a single row of fields.
    """
    # Streamlit reruns keep the old option list; rescanning has to be explicit or a headset
    # paired after the app started never appears.
    rescan_on = rescan_target if rescan_target is not None else target
    rescan = rescan_on.button(
        "Rescan", key=f"{key}_rescan", width="stretch",
        help="Re-enumerate connected serial/Bluetooth devices. Press this after pairing a headset.",
    )
    port_entries = cached_serial_port_options(force=rescan)
    labels = {port: label for port, label in port_entries}
    options: List[str] = [port for port, _ in port_entries]

    help_text = field.get("help") or ""
    if list_ports is None:
        target.error(
            "pyserial is not importable, so no ports can be listed "
            f"({_SERIAL_IMPORT_ERROR}). Install it with `pip install pyserial`, "
            "or choose Manual entry."
        )
    elif not port_entries:
        target.warning(
            "No serial ports found. Pair the UNICORN first — each headset appears as its "
            "own port, and all of them are listed here."
        )
    else:
        unicorn_like = [p for p, lab in port_entries if lab.startswith("UNICORN")]
        found = f"{len(port_entries)} port(s) found" + (
            f", {len(unicorn_like)} look like UNICORN devices." if unicorn_like else "."
        )
        help_text = f"{help_text}\n\n{found}" if help_text else found

    current_str = str(current) if current not in (None, "") else ""
    if current_str and current_str not in options:
        options.append(current_str)
        labels[current_str] = f"{current_str} (saved, not currently present)"

    options.append(MANUAL_UNICORN_PORT_OPTION)
    default_choice = current_str if current_str in options else options[0]

    selection = target.selectbox(
        field["label"],
        options=options,
        index=options.index(default_choice) if options else 0,
        format_func=lambda value: "Manual entry" if value == MANUAL_UNICORN_PORT_OPTION else labels.get(value, value),
        help=help_text or None,
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

        # Everything on one bottom-aligned row, each field only as wide as it needs. A
        # 2-column grid put three fields over two rows and left half the panel empty, and
        # the rescan button stacked above the port instead of sitting next to it.
        layout: List[Any] = []
        for field in schema:
            layout.append((DEVICE_FIELD_WIDTHS.get(field["name"], 1.4), field))
            if device == "UNICORN" and field["name"] == "UNICORNPort":
                layout.append((0.7, _RESCAN_SLOT))

        cols = st.columns([width for width, _ in layout], vertical_alignment="bottom")
        slots = {id(item): col for col, (_, item) in zip(cols, layout)}
        rescan_col = slots.get(id(_RESCAN_SLOT))

        for _width, field in layout:
            if field is _RESCAN_SLOT:
                continue
            target = slots[id(field)]
            key = f"device_{device}_{field['name']}"
            current = form_state.get(field["name"], field.get("default"))
            if device == "UNICORN" and field["name"] == "UNICORNPort":
                value = render_unicorn_port_input(target, field, current, key, rescan_target=rescan_col)
            elif field["kind"] == "number":
                fallback = field.get("default", 0.0)
                numeric = coerce_number(current)
                value_default = float(numeric if numeric is not None else fallback or 0.0)
                # Key-only (no value=) so the stepper does not snap back on a slow rerun.
                if not isinstance(st.session_state.get(key), (int, float)) or isinstance(st.session_state.get(key), bool):
                    st.session_state[key] = value_default
                kwargs: Dict[str, Any] = {
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
