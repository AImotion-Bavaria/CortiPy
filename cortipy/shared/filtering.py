"""Filtering utilities for EEG processing."""

from __future__ import annotations

import numpy as np
from scipy import signal


def filter_vep(data, highpass_cutoff: float, fs: float) -> np.ndarray:
    """Apply a zero-phase high-pass filter to VEP data."""
    if data is None:
        raise ValueError("`data` must not be None.")
    array = np.asarray(data, dtype=float)
    squeeze = False
    if array.ndim == 1:
        array = array[:, np.newaxis]
        squeeze = True
    if array.size == 0:
        return np.asarray(data, dtype=float)

    b_hp, a_hp = signal.butter(2, highpass_cutoff, btype="highpass", fs=fs)

    filtered = signal.filtfilt(b_hp, a_hp, array, axis=0, padlen=min(3 * max(len(a_hp), len(b_hp)), array.shape[0] - 1))

    if squeeze:
        return filtered[:, 0]
    return filtered
