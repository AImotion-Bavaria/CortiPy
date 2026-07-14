"""Electrode-table utilities for the Streamlit UI.

Extracted from apps/streamlit_app.py (modularization). Pure impedance/channel
helpers plus the editor-revision bump. The full ``render_channel_editor`` stays
in the app for now (it is deeply tied to Streamlit state, plotting and devices).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from cortipy.ui_streamlit.constants import DEVICE_EXTRA_LABELS


def actichamp_channel_count(rows: List[Dict[str, Any]]) -> int:
    extras = set(DEVICE_EXTRA_LABELS.get("ActiCHamp", []))
    return sum(1 for row in rows if row.get("Channel") not in extras)


def bump_channel_editor_revision(device: str) -> None:
    revisions = st.session_state.setdefault("_channel_editor_revision", {})
    revisions[device] = int(revisions.get(device, 0)) + 1


def _normalize(label: Any) -> str:
    """Compare channel labels ignoring case and spacing ("Ref" == "REF", "Ch 1" == "ch1")."""
    return "".join(ch for ch in str(label or "").lower() if ch.isalnum())


def map_impedances_to_channels(rows: List[Dict[str, Any]], values: List[float]) -> List[Dict[str, Any]]:
    """Write measured impedances (ohms) onto the electrode rows, in kOhm.

    The amplifier reports ``|GND|REF|CH1|CH2|...`` (AmplifierSDK.h). Matching is
    case-insensitive: the row label is "Ref" while the SDK calls it "REF", so an exact
    comparison silently left the reference electrode's impedance unset.
    """
    if len(values) < 3:
        return rows

    labels = ["GND", "REF"] + [f"Ch {idx}" for idx in range(1, len(values) - 1)]
    mapping = {_normalize(label): values[idx] for idx, label in enumerate(labels) if idx < len(values)}

    for row in rows:
        value = mapping.get(_normalize(row.get("Channel")))
        if value is None or value < 0:
            continue
        row["Impedance"] = round(float(value) / 1000.0, 1)

    return rows


def has_measured_impedance(values: List[float]) -> bool:
    return any(value > 0 for value in values)


def impedance_range_kohm(values: List[float]) -> Optional[tuple[float, float]]:
    measured = [float(value) / 1000.0 for value in values if value > 0]
    if not measured:
        return None
    return min(measured), max(measured)
