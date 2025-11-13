"""BERA-specific helpers (filtering, preprocessing, metrics)."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import signal, special


def filter_bera(
    data: np.ndarray,
    fs: float,
    exclude_channels: Iterable[int],
    low_cutoff: float = 100.0,
    high_cutoff: float = 3000.0,
    order: int = 4,
) -> np.ndarray:
    """Band-pass filter all channels except those listed in exclude_channels."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("BERA filtering expects a 2-D array (samples x channels).")
    result = np.array(array, copy=True)
    if array.size == 0:
        return result

    nyquist = fs / 2.0
    low = np.clip(low_cutoff / nyquist, 1e-6, 0.9999)
    high = np.clip(high_cutoff / nyquist, 1e-6, 0.9999)

    b_high, a_high = signal.butter(order, low, btype="highpass")
    b_low, a_low = signal.butter(order, high, btype="lowpass")

    exclude = set(int(idx) for idx in exclude_channels)
    for idx in range(array.shape[1]):
        if idx in exclude:
            continue
        column = array[:, idx]
        column = signal.filtfilt(b_high, a_high, column, axis=0)
        column = signal.filtfilt(b_low, a_low, column, axis=0)
        result[:, idx] = column
    return result


def prepro(signal_in: np.ndarray, voltage: float) -> np.ndarray:
    """Convert to nV using MATLAB's `prepro` behaviour."""
    data = np.asarray(signal_in, dtype=float)
    if voltage == 1e-3:
        return data * 1e6
    if voltage == 1e-6:
        return (data / 100.0) * 1e3
    raise ValueError("Unsupported voltage level; expected 1e-3 or 1e-6.")


def avg_seg_avg(segments: np.ndarray, n_splits: int) -> np.ndarray:
    """Average all sweeps and split into blocks similar to MATLAB avg_seg_avg."""
    arr = np.asarray(segments, dtype=float)
    if arr.ndim != 3:
        raise ValueError("Segments must be a 3-D array (epochs x samples x channels).")
    means = [arr.mean(axis=0, keepdims=True)]
    if n_splits <= 0 or arr.shape[0] < n_splits:
        return np.concatenate(means, axis=0)
    block_size = arr.shape[0] // n_splits
    if block_size == 0:
        return np.concatenate(means, axis=0)
    for idx in range(n_splits):
        start = idx * block_size
        end = start + block_size
        block = arr[start:end]
        if block.size == 0:
            continue
        means.append(block.mean(axis=0, keepdims=True))
    return np.concatenate(means, axis=0)


def block_weighted(sweeps: np.ndarray, fs: float, sp_time: float, block_size: int = 250) -> np.ndarray:
    """Weighted block averaging via Bayesian inference."""
    array = np.asarray(sweeps, dtype=float)
    num_blocks = array.shape[0] // block_size
    if num_blocks == 0:
        return array.mean(axis=0)
    sp_idx = int(round(sp_time * fs))
    sp_idx = np.clip(sp_idx, 0, array.shape[1] - 1)

    numerator = np.zeros(array.shape[1])
    denominator = 0.0
    for block in range(num_blocks):
        start = block * block_size
        end = start + block_size
        segment = array[start:end]
        variance = np.var(segment[:, sp_idx])
        if variance <= 0:
            continue
        numerator += segment.mean(axis=0) / variance
        denominator += 1.0 / variance
    if denominator == 0:
        return array.mean(axis=0)
    return numerator / denominator


def residual_noise(sweeps: np.ndarray, fs: float, noise_window: Tuple[float, float]) -> float:
    array = np.asarray(sweeps, dtype=float)
    start, end = _window_to_slice(noise_window, fs, array.shape[1])
    window = array[:, start:end]
    var_per_sample = np.var(window, axis=0)
    rms = np.sqrt(var_per_sample)
    return float(np.mean(rms) * 3.07)


def residual_noise_eclipse(sweeps: np.ndarray, fs: float, analysis_window: Tuple[float, float]) -> Tuple[float, np.ndarray]:
    array = np.asarray(sweeps, dtype=float)
    start, end = _window_to_slice(analysis_window, fs, array.shape[1])
    odd = array[0::2, start:end]
    even = array[1::2, start:end]
    if even.size == 0:
        even = odd
    avg_odd = odd.mean(axis=0)
    avg_even = even.mean(axis=0)
    diff_wave = (avg_odd - avg_even) / 2.0
    return float(np.std(diff_wave)), diff_wave


def f_test_elberling_don(var_x: float, var_y: float, n: int) -> Tuple[float, float]:
    df1 = 5
    df2 = max(n - 1, 1)
    var_y = var_y if var_y > 0 else 1e-12
    f_value = n * (var_x / var_y)
    p_value = special.betainc(
        df1 / 2.0,
        df2 / 2.0,
        (df1 * f_value) / (df1 * f_value + df2),
    )
    p_value = float(min(max(p_value, 0.0), 1.0))
    return p_value, float(f_value)


def get_fsp_fmp(
    sweeps: np.ndarray,
    avg_signal: np.ndarray,
    fs: float,
    sp_time: float,
    analysis_window: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    sweeps_arr = np.asarray(sweeps, dtype=float)
    avg_arr = np.asarray(avg_signal, dtype=float).reshape(-1)
    n_epochs, n_samples = sweeps_arr.shape
    sp_idx = np.clip(int(round(sp_time * fs)), 0, n_samples - 1)
    idx_10ms = np.clip(int(round(0.010 * fs)), 1, n_samples)
    win_start, win_end = _window_to_slice(analysis_window, fs, n_samples)

    var_avg = np.var(avg_arr[:idx_10ms])
    var_sweeps = np.var(sweeps_arr[:, sp_idx])
    p_sp, fsp = f_test_elberling_don(var_avg, var_sweeps, n_epochs)

    var_avg_multi = np.var(avg_arr[win_start:win_end])
    var_sweeps_multi = np.mean(np.var(sweeps_arr[:, win_start:win_end], axis=0))
    p_mp, fmp = f_test_elberling_don(var_avg_multi, var_sweeps_multi, n_epochs)
    return fsp, p_sp, fmp, p_mp


def wave_amplitude(
    wave_low: float, wave_high: float, data: np.ndarray, time_ms: np.ndarray
) -> Tuple[float, Optional[int], Optional[int]]:
    mask = (time_ms >= wave_low) & (time_ms <= wave_high)
    idx_range = np.flatnonzero(mask)
    if idx_range.size == 0:
        return float("nan"), None, None
    idx_max = int(idx_range[np.argmax(data[idx_range])])

    min_mask = (time_ms >= time_ms[idx_max] + 2.0) & (time_ms <= time_ms[idx_max] + 3.0)
    idx_min_range = np.flatnonzero(min_mask)
    if idx_min_range.size == 0:
        return float("nan"), idx_max, None
    idx_min = int(idx_min_range[np.argmin(data[idx_min_range])])
    amplitude = float(data[idx_max] - data[idx_min])
    return amplitude, idx_max, idx_min


def _window_to_slice(window: Tuple[float, float], fs: float, max_samples: int) -> Tuple[int, int]:
    start = max(0, int(round(window[0] * fs)))
    end = min(max_samples, int(round(window[1] * fs)))
    if end <= start:
        end = min(max_samples, start + 1)
    return start, end
