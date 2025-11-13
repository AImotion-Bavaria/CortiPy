"""Signal segmentation helpers."""

from __future__ import annotations

import numpy as np


def seg_sig_fast(data: np.ndarray, fs: float, max_time: float, trigger_channel: int) -> np.ndarray:
    """Port of the MATLAB `seg_sig_fast` helper."""
    if data is None:
        raise ValueError("`data` must not be None")
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("`data` must be a 2-D array of samples x channels.")
    n_channels = array.shape[1]
    trig_idx = int(trigger_channel)
    if trig_idx < 0 or trig_idx >= n_channels:
        raise IndexError("`trigger_channel` is out of bounds.")
    max_samples = max(1, int(round(float(max_time) * float(fs))))

    segments: list[np.ndarray] = []
    current: list[np.ndarray] = []
    capturing = False
    prev_value = float(array[0, trig_idx])

    for row in array:
        value = float(row[trig_idx])
        if value == 0 and prev_value != 0:
            if capturing and current:
                segments.append(np.vstack(current))
            capturing = True
            current = []
        prev_value = value

        if capturing and len(current) < max_samples:
            current.append(row.copy())

    if capturing and current:
        segments.append(np.vstack(current))

    if len(segments) <= 1:
        return np.zeros((0, max_samples, n_channels))

    trimmed = segments[:-1]  # drop final (possibly incomplete) sweep
    padded = np.zeros((len(trimmed), max_samples, n_channels))
    for idx, segment in enumerate(trimmed):
        length = min(segment.shape[0], max_samples)
        padded[idx, :length, :] = segment[:length, :]
    return padded


def seg_sig_fast_p300(data: np.ndarray, fs: float, max_time: float, trigger_channel: int) -> np.ndarray:
    """Variant of `seg_sig_fast` that discards the initial sweep."""
    segments = seg_sig_fast(data, fs, max_time, trigger_channel)
    if segments.shape[0] <= 1:
        return np.zeros_like(segments)
    return segments[1:, :, :]
