"""ASSR-specific spectral helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SpectralResult:
    psd: np.ndarray
    freq: np.ndarray
    dBpsd: np.ndarray


def compute_psd(signal: np.ndarray, fs: float) -> SpectralResult:
    """Replicates the MATLAB `clacPSD` helper."""
    data = np.asarray(signal, dtype=float)
    n = data.shape[0]
    fft_vals = np.fft.rfft(data, axis=0)
    psd = (1.0 / (fs * n)) * np.abs(fft_vals) ** 2
    if psd.shape[0] > 2:
        psd[1:-1] *= 2
    freq = np.linspace(0, fs / 2.0, psd.shape[0])
    dBpsd = 10.0 * np.log10(np.maximum(psd, np.finfo(float).tiny))
    return SpectralResult(psd=psd, freq=freq, dBpsd=dBpsd)


def calc_snr(
    spectrum: np.ndarray,
    freq: np.ndarray,
    stim_freq: float,
    noise_min: float,
    noise_max: float,
    f_signal_band: float,
) -> float:
    """Port of `CalcSNR.m`."""
    amplitudes = np.abs(np.asarray(spectrum, dtype=float))
    noise_ranges = [
        (noise_min, stim_freq - f_signal_band),
        (stim_freq + f_signal_band, noise_max),
    ]
    signal_range = (stim_freq - f_signal_band, stim_freq + f_signal_band)

    def mean_band(band: Tuple[float, float]) -> float:
        low, high = band
        if low >= high:
            return float("nan")
        idx_low = int(np.argmin(np.abs(freq - low)))
        idx_high = int(np.argmin(np.abs(freq - high)))
        if idx_high <= idx_low:
            idx_high = idx_low + 1
        return float(np.mean(amplitudes[idx_low:idx_high]))

    noise_values = [mean_band(band) for band in noise_ranges]
    noise = np.nanmean(noise_values)
    signal = mean_band(signal_range)
    if noise is None or noise <= 0 or signal is None or not np.isfinite(signal):
        return float("nan")
    return float(20.0 * np.log10(signal / noise))


def assr_f_test(
    spectrum: np.ndarray,
    freq: np.ndarray,
    stim_freq: float,
    noise_min: float,
    noise_max: float,
    f_signal_band: float,
    critical_value: float = 3.0,
) -> Tuple[float, float]:
    """Port of `ftestASSR.m`."""
    amplitudes = np.abs(np.asarray(spectrum, dtype=float))
    noise_ranges = [
        (noise_min, stim_freq - f_signal_band),
        (stim_freq + f_signal_band, noise_max),
    ]
    signal_range = (stim_freq - f_signal_band, stim_freq + f_signal_band)

    def band_mean(band: Tuple[float, float]) -> float:
        low, high = band
        if low >= high:
            return float("nan")
        idx_low = int(np.argmin(np.abs(freq - low)))
        idx_high = int(np.argmin(np.abs(freq - high)))
        if idx_high <= idx_low:
            idx_high = idx_low + 1
        return float(np.mean(amplitudes[idx_low:idx_high]))

    noise = np.nanmean([band_mean(band) for band in noise_ranges])
    signal = band_mean(signal_range)
    if noise is None or noise == 0 or not np.isfinite(noise):
        return float("nan"), critical_value
    f_value = signal / noise
    return float(f_value), float(critical_value)
