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
    if n <= 0:
        return np.asarray([]), np.asarray([])
    xdft = np.abs(np.fft.rfft(data, axis=0)) / n
    if xdft.shape[0] > 1:
        if n % 2 == 0:
            xdft[1:-1] *= 2.0
        else:
            xdft[1:] *= 2.0
    freq = np.fft.rfftfreq(n, d=1.0 / float(fs))
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
