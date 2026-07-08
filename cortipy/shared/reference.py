"""Shared EEG reference helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def eeg_channel_count(params_block: Mapping[str, Any], total_columns: int) -> int:
    """Return the number of leading columns that represent EEG channels."""
    count = _safe_int(params_block.get("NumberEEGChannels"), total_columns)
    if count <= 0:
        count = total_columns
    return min(count, total_columns)


def reference_channel_index(params_block: Mapping[str, Any], eeg_count: int) -> int:
    """Return the 0-based reference index, constrained to the EEG channel range."""
    if eeg_count <= 0:
        return 0
    index = _safe_int(params_block.get("ReferenceChannel"), 1) - 1
    return index if 0 <= index < eeg_count else 0


def apply_eeg_reference(data: np.ndarray, params_block: Mapping[str, Any]) -> np.ndarray:
    """Apply all-EEG-channel minus selected-reference-channel referencing."""
    referenced = np.asarray(data, dtype=float).copy()
    if referenced.ndim != 2 or referenced.shape[1] == 0:
        return referenced

    eeg_count = eeg_channel_count(params_block, referenced.shape[1])
    if eeg_count <= 0:
        return referenced

    ref_idx = reference_channel_index(params_block, eeg_count)
    referenced[:, :eeg_count] = referenced[:, :eeg_count] - referenced[:, [ref_idx]]
    return referenced
