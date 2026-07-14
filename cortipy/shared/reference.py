"""Shared EEG reference helpers."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

LOGGER = logging.getLogger("cortipy.shared.reference")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def eeg_channel_count(params_block: Mapping[str, Any], total_columns: int) -> int:
    """Return the number of leading columns that represent EEG channels."""
    explicit = _safe_int(params_block.get("NumberEEGChannels"), 0)
    if explicit > 0:
        return min(explicit, total_columns)

    count = total_columns
    trigger_idx = _safe_int(params_block.get("TriggerChannel"), 0) - 1
    if 0 <= trigger_idx < total_columns:
        count = min(count, trigger_idx)

    aux_count = _safe_int(params_block.get("NumberAUXChannels"), 0)
    if aux_count > 0:
        count = min(count, max(0, total_columns - aux_count))

    if count <= 0:
        count = total_columns
    return min(count, total_columns)


def reference_channel_index(params_block: Mapping[str, Any], eeg_count: int) -> int:
    """Return the 0-based reference index, constrained to the EEG channel range."""
    if eeg_count <= 0:
        return 0
    index = _safe_int(params_block.get("ReferenceChannel"), 1) - 1
    if 0 <= index < eeg_count:
        return index
    # Referencing against a channel that was never acquired would silently subtract
    # channel 1 instead — say so, because the resulting signal is not what was asked for.
    LOGGER.warning(
        "ReferenceChannel %r is outside the %d acquired EEG channels; using channel 1",
        params_block.get("ReferenceChannel"),
        eeg_count,
    )
    return 0


def apply_eeg_reference(
    data: np.ndarray,
    params_block: Mapping[str, Any],
    *,
    zero_reference: bool = True,
) -> np.ndarray:
    """Apply all-EEG-channel minus selected-reference-channel referencing."""
    referenced = np.asarray(data, dtype=float).copy()
    if referenced.ndim != 2 or referenced.shape[1] == 0:
        return referenced
    if "ReferenceChannel" not in params_block:
        return referenced

    eeg_count = eeg_channel_count(params_block, referenced.shape[1])
    if eeg_count <= 0:
        return referenced

    ref_idx = reference_channel_index(params_block, eeg_count)
    trigger_idx = _safe_int(params_block.get("TriggerChannel"), 0) - 1
    eeg_indices = [idx for idx in range(eeg_count) if idx != trigger_idx]
    if not eeg_indices:
        return referenced

    reference_values = referenced[:, [ref_idx]]
    target_indices = eeg_indices if zero_reference else [idx for idx in eeg_indices if idx != ref_idx]
    if target_indices:
        referenced[:, target_indices] = referenced[:, target_indices] - reference_values
    return referenced
