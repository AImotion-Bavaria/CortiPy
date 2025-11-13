"""Trigger processing helpers."""

from __future__ import annotations

import numpy as np


def trigger_adc(
    data: np.ndarray,
    fs: float,
    trigger_channel: int,
    max_time: float,
    edge: str = "b",
    threshold: float = 0.02,
) -> np.ndarray:
    """Emulate the MATLAB `triggerADC` helper."""
    if data is None:
        raise ValueError("`data` must not be None.")
    array = np.asarray(data, dtype=float).copy()
    if array.ndim != 2:
        raise ValueError("`data` must be a 2-D array (samples x channels).")
    if trigger_channel < 0 or trigger_channel >= array.shape[1]:
        raise IndexError("trigger_channel out of bounds.")

    channel = array[:, trigger_channel]
    unique_vals = np.unique(np.round(channel, decimals=6))
    analog_signal = unique_vals.size != 2

    if analog_signal:
        diff_signal = np.diff(channel)
        if diff_signal.size == 0:
            return array
        if edge == "r":
            temp = np.square(np.maximum(diff_signal, 0.0))
        elif edge == "f":
            temp = np.square(np.minimum(diff_signal, 0.0))
        else:  # both
            temp = np.square(np.abs(diff_signal))
        max_val = float(np.max(temp))
        temp = temp / max_val if max_val > 0 else temp
        temp_bool = temp > threshold
    else:
        temp_bool = channel[1:] > 0.5 if channel.size > 1 else np.array([], dtype=bool)

    padded_prev = np.concatenate(([False], temp_bool))
    padded_next = np.concatenate((temp_bool, [False]))
    indices = np.flatnonzero(padded_prev & ~padded_next)

    times = indices / float(fs)
    if times.size > 1:
        diffs = np.diff(times)
        keep = np.ones_like(times, dtype=bool)
        keep[1:] = diffs >= max_time
        times = times[keep]

    idx_times = np.clip(np.round(times * fs).astype(int), 0, array.shape[0] - 1)
    trigger_signal = np.zeros(array.shape[0], dtype=float)
    trigger_signal[idx_times] = 1.0
    array[:, trigger_channel] = trigger_signal
    return array
