"""Shared helpers for SSVEP/BCI processing."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy import linalg, stats


def get_frequency_indices(freq: np.ndarray, targets: Sequence[float]) -> np.ndarray:
    freq = np.asarray(freq, dtype=float)
    indices = []
    for target in targets:
        if freq.size == 0:
            indices.append(0)
            continue
        idx = int(np.argmin(np.abs(freq - target)))
        indices.append(idx)
    return np.asarray(indices, dtype=int)


def classify_fft(
    freq: np.ndarray,
    spectrum: np.ndarray,
    ssvep_indices: np.ndarray,
    low_freq: float,
    high_freq: float,
    dominance_ratio: float = 1.4,
) -> np.ndarray:
    """Port of MATLAB `classification.m`."""
    freq = np.asarray(freq, dtype=float)
    psd = np.abs(np.asarray(spectrum, dtype=float))
    low_idx = int(np.argmin(np.abs(freq - low_freq)))
    high_idx = int(np.argmin(np.abs(freq - high_freq)))
    if high_idx <= low_idx:
        high_idx = low_idx + 1
    segment = psd[low_idx:high_idx]
    if segment.size < 2:
        return np.zeros_like(ssvep_indices, dtype=int)
    sorted_idx = np.argsort(segment)
    top = segment[sorted_idx[-1]]
    second = segment[sorted_idx[-2]]
    peak_idx = sorted_idx[-1] + low_idx
    predictions = np.zeros(len(ssvep_indices), dtype=int)
    for i, idx in enumerate(ssvep_indices):
        if top > second * dominance_ratio and (idx - 1) <= peak_idx <= (idx + 1):
            predictions[i] = 1
    return predictions


def plot_psd_ssvep(ax, freq: np.ndarray, spectrum: np.ndarray, low_freq: float, high_freq: float) -> None:
    import matplotlib.pyplot as plt  # local import to avoid mandatory dependency at module import

    freq = np.asarray(freq, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    low_idx = int(np.argmin(np.abs(freq - low_freq)))
    high_idx = int(np.argmin(np.abs(freq - high_freq)))
    if high_idx <= low_idx:
        high_idx = low_idx + 1
    ax.plot(freq[low_idx:high_idx], spectrum[low_idx:high_idx], "+-", color="tab:blue")
    ax.set_title("Periodogram Using FFT")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (uV)")
    ax.grid(True, alpha=0.3)


def send_prediction_bt(pred: Sequence[int], bt_connection) -> None:
    if bt_connection is None:
        return
    command = "S"
    if len(pred) >= 1 and pred[0]:
        command = "R"
    elif len(pred) >= 2 and pred[1]:
        command = "L"
    elif len(pred) >= 3 and pred[2]:
        command = "F"
    try:
        bt_connection.write(command.encode("utf-8"))
    except Exception:
        pass


def describe_prediction(pred: Sequence[int], stim_freqs: Sequence[float]) -> str:
    msgs = []
    for flag, freq in zip(pred, stim_freqs):
        if flag:
            msgs.append(f"SSVEP {freq:.1f} Hz detected")
    return "; ".join(msgs) if msgs else "No stimulus detected"


def compute_t2circ(data: np.ndarray, fs: float, stim_freqs: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute T²circ statistics for each channel."""
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if not stim_freqs:
        return np.array([]), np.array([])
    primary = float(stim_freqs[0])
    if primary <= 0 or fs <= 0:
        return np.array([]), np.array([])
    trial_duration = 2.0 / primary
    trial_samples = max(1, int(round(fs * trial_duration)))
    num_trials = data.shape[0] // trial_samples
    if num_trials < 2:
        return np.full(data.shape[1], np.nan), np.full(data.shape[1], np.nan)
    trimmed = data[: num_trials * trial_samples]
    trials = trimmed.reshape(num_trials, trial_samples, data.shape[1])
    fft_trials = np.fft.rfft(trials, axis=1)
    freq = np.fft.rfftfreq(trial_samples, d=1.0 / fs)
    idx = int(np.argmin(np.abs(freq - primary)))
    if idx >= fft_trials.shape[1]:
        idx = fft_trials.shape[1] - 1

    t2_values = np.zeros(data.shape[1])
    p_values = np.zeros(data.shape[1])
    df1 = 2
    df2 = max(2 * num_trials - 2, 1)
    for ch in range(data.shape[1]):
        z = fft_trials[:, idx, ch]
        z_hat = np.mean(z)
        numerator = (num_trials - 1) * np.abs(z_hat) ** 2
        denominator = np.sum(np.abs(z - z_hat) ** 2)
        if denominator <= 0:
            t2 = np.nan
        else:
            t2 = float(numerator / denominator)
        t2_values[ch] = t2
        if np.isnan(t2):
            p = np.nan
        else:
            p = stats.f.cdf(max(t2, 0), df1, df2)
        p_values[ch] = p
    return t2_values, p_values


def cca_correlations(data: np.ndarray, fs: float, stim_freqs: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return CCA correlation for each reference frequency."""
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.shape[0] < 2:
        return np.array([]), np.array([])
    t = np.arange(data.shape[0]) / fs
    freq_list = list(stim_freqs) + [stim_freqs[0] + 1, 50] if stim_freqs else [50]
    rho = np.zeros(len(freq_list))
    for i, freq in enumerate(freq_list):
        ref = np.column_stack(
            [
                np.cos(2 * np.pi * freq * t),
                np.sin(2 * np.pi * freq * t),
                np.cos(4 * np.pi * freq * t),
                np.sin(4 * np.pi * freq * t),
            ]
        )
        rho[i] = _max_canonical_correlation(data, ref)
    return rho, np.asarray(freq_list, dtype=float)


def _max_canonical_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - np.mean(X, axis=0, keepdims=True)
    Yc = Y - np.mean(Y, axis=0, keepdims=True)
    n = Xc.shape[0]
    if n <= 1:
        return float("nan")
    Sxx = (Xc.T @ Xc) / (n - 1)
    Syy = (Yc.T @ Yc) / (n - 1)
    Sxy = (Xc.T @ Yc) / (n - 1)
    eps = 1e-9
    Sxx += eps * np.eye(Sxx.shape[0])
    Syy += eps * np.eye(Syy.shape[0])
    try:
        Rx = linalg.cholesky(Sxx, lower=True)
    except linalg.LinAlgError:
        Rx = linalg.cholesky(Sxx + eps * np.eye(Sxx.shape[0]), lower=True)
    try:
        Ry = linalg.cholesky(Syy, lower=True)
    except linalg.LinAlgError:
        Ry = linalg.cholesky(Syy + eps * np.eye(Syy.shape[0]), lower=True)
    invRx = linalg.solve_triangular(Rx, np.eye(Rx.shape[0]), lower=True)
    invRy = linalg.solve_triangular(Ry, np.eye(Ry.shape[0]), lower=True)
    M = invRx @ Sxy @ invRy.T
    sing_vals = linalg.svd(M, compute_uv=False)
    if sing_vals.size == 0:
        return float("nan")
    return float(min(max(sing_vals[0], 0.0), 1.0))


def ssvep_snr(
    spectrum: np.ndarray,
    freq: np.ndarray,
    stim_freqs: Sequence[float],
    noise_min: float,
    noise_max: float,
    band_width: float,
) -> np.ndarray:
    """Vectorised SNR computation mirroring MATLAB's CalcSNR."""
    freq = np.asarray(freq, dtype=float)
    amps = np.abs(np.asarray(spectrum, dtype=float))
    if amps.ndim == 1:
        amps = amps[:, np.newaxis]
    snr_matrix = np.zeros((len(stim_freqs), amps.shape[1]))
    for idx, stim in enumerate(np.atleast_1d(stim_freqs).astype(float)):
        noise_ranges = [
            (noise_min, stim - band_width),
            (stim + band_width, noise_max),
        ]
        signal_range = (stim - band_width, stim + band_width)
        noise_values = []
        for low, high in noise_ranges:
            if high <= low:
                continue
            i_low = int(np.argmin(np.abs(freq - low)))
            i_high = int(np.argmin(np.abs(freq - high)))
            if i_high <= i_low:
                i_high = i_low + 1
            noise_values.append(np.mean(amps[i_low:i_high, :], axis=0))
        if noise_values:
            noise = np.mean(noise_values, axis=0)
        else:
            noise = np.full(amps.shape[1], np.nan)
        s_low = int(np.argmin(np.abs(freq - signal_range[0])))
        s_high = int(np.argmin(np.abs(freq - signal_range[1])))
        if s_high <= s_low:
            s_high = s_low + 1
        signal = np.mean(amps[s_low:s_high, :], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = 20.0 * np.log10(signal / noise)
        snr_matrix[idx, :] = snr
    return snr_matrix.squeeze()


def ssvep_f_test(psd: np.ndarray, freq: np.ndarray, stim_freq: float, alpha: float = 0.05) -> dict:
    """Reimplementation of the legacy `F_Test` helper."""
    freq = np.asarray(freq, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if psd.ndim == 1:
        psd = psd[:, np.newaxis]
    idx_target = int(np.argmin(np.abs(freq - stim_freq)))
    noise_low = stim_freq - np.ceil(stim_freq / 2.0)
    noise_high = stim_freq + np.ceil(stim_freq / 2.0)
    noise_mask = (freq >= noise_low) & (freq <= noise_high) & (np.abs(freq - stim_freq) > 1e-9)
    noise_indices = np.where(noise_mask)[0]
    if not noise_indices.size:
        return {}
    p_peak = psd[idx_target, :]
    p_noise = np.mean(psd[noise_indices, :], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_values = p_peak / p_noise
    f_values = np.where(np.isfinite(f_values), f_values, np.nan)
    f_crit = stats.f.ppf(1 - alpha, dfn=1, dfd=len(noise_indices))
    significant = f_values > f_crit
    return {
        "channel": np.arange(psd.shape[1]),
        "P_peak": p_peak,
        "P_noise": p_noise,
        "F_value": f_values,
        "F_crit": np.full(psd.shape[1], f_crit),
        "significant": significant,
    }
