"""Signal-processing helpers translated from MATLAB."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from scipy.signal import get_window


def hann_window(length: int, periodic: bool = True) -> np.ndarray:
    """Return a Hann window with MATLAB-compatible periodic behavior."""
    return get_window("hann", length, fftbins=periodic)


def calc_fft(data: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return the single-sided FFT magnitude spectrum."""
    data = np.asarray(data)
    n = data.shape[0]
    xdft = np.fft.rfft(data, axis=0)
    xdft = np.abs(xdft) / n
    freq = np.linspace(0.0, fs / 2.0, xdft.shape[0])
    return xdft, freq


def time_vector(data_or_length: Union[int, np.ndarray], fs: float, unit: str = "s") -> np.ndarray:
    """Create a time vector for the given sample count or data array."""
    if np.isscalar(data_or_length):
        length = int(data_or_length)
    else:
        length = int(np.asarray(data_or_length).shape[0])
    t = np.arange(length, dtype=float) / float(fs)
    if unit.lower() in {"ms", "millisecond", "milliseconds"}:
        t *= 1000.0
    return t
