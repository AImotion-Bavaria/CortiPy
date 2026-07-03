"""Standard 10-20 electrode scalp coordinates and lookup helpers.

Extracted from apps/streamlit_app.py as the first step of splitting that module up.
Pure data + leaf helper — no Streamlit or app dependencies.
"""
from __future__ import annotations

from typing import Optional

TEN_TWENTY_COORDS = {
    "Fp1": (-0.5, 0.9),
    "Fpz": (0.0, 0.95),
    "Fp2": (0.5, 0.9),
    "F7": (-0.95, 0.6),
    "F3": (-0.5, 0.6),
    "Fz": (0.0, 0.65),
    "F4": (0.5, 0.6),
    "F8": (0.95, 0.6),
    "FC5": (-0.7, 0.35),
    "FC3": (-0.4, 0.35),
    "FC1": (-0.2, 0.35),
    "FCz": (0.0, 0.35),
    "FC2": (0.2, 0.35),
    "FC4": (0.4, 0.35),
    "FC6": (0.7, 0.35),
    "T7": (-1.05, 0.0),
    "C5": (-0.7, 0.05),
    "C3": (-0.5, 0.0),
    "C1": (-0.2, 0.0),
    "Cz": (0.0, 0.0),
    "C2": (0.2, 0.0),
    "C4": (0.5, 0.0),
    "C6": (0.7, 0.05),
    "T8": (1.05, 0.0),
    "TP7": (-0.95, -0.2),
    "CP5": (-0.7, -0.25),
    "CP3": (-0.4, -0.25),
    "CP1": (-0.2, -0.25),
    "CPz": (0.0, -0.25),
    "CP2": (0.2, -0.25),
    "CP4": (0.4, -0.25),
    "CP6": (0.7, -0.25),
    "TP8": (0.95, -0.2),
    "P7": (-0.95, -0.55),
    "P5": (-0.65, -0.55),
    "P3": (-0.5, -0.55),
    "P1": (-0.2, -0.55),
    "Pz": (0.0, -0.6),
    "P2": (0.2, -0.55),
    "P4": (0.5, -0.55),
    "P6": (0.65, -0.55),
    "P8": (0.95, -0.55),
    "PO7": (-0.7, -0.75),
    "PO3": (-0.35, -0.75),
    "PO4": (0.35, -0.75),
    "PO8": (0.7, -0.75),
    "O1": (-0.3, -0.95),
    "Oz": (0.0, -1.0),
    "O2": (0.3, -0.95),
}


def channel_default_coords(label: str) -> tuple[Optional[float], Optional[float]]:
    """Return (x, y) scalp coordinates for a 10-20 label, or (None, None) if unknown."""
    clean = label.replace(" ", "")
    coords = TEN_TWENTY_COORDS.get(clean)
    if coords is None:
        return None, None
    return coords
