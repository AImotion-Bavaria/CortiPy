"""Unit tests for shared signal helpers."""

from __future__ import annotations

import numpy as np

from cortipy.shared.signal import calc_fft, time_vector


def test_calc_fft_returns_expected_spectrum():
    """FFT helper should return a single-sided magnitude spectrum matching numpy semantics for a known sine wave."""
    fs = 100.0
    t = np.arange(0, 1.0, 1 / fs)
    data = np.sin(2 * np.pi * 10 * t)
    spectrum, freq = calc_fft(data, fs)
    assert np.isclose(freq[-1], fs / 2.0)
    assert np.argmax(spectrum) == 10
    assert np.isclose(float(np.max(spectrum)), 1.0)


def test_calc_fft_labels_50hz_bin_with_odd_sample_count():
    fs = 250.0
    n_samples = 375
    t = np.arange(n_samples) / fs
    data = 2.0 * np.sin(2 * np.pi * 50.0 * t)

    spectrum, freq = calc_fft(data, fs)
    peak_idx = int(np.argmax(spectrum))

    assert np.isclose(freq[peak_idx], 50.0)
    assert np.isclose(spectrum[peak_idx], 2.0)


def test_time_vector_supports_seconds_and_milliseconds():
    """time_vector should produce evenly spaced stamps with correct unit scaling."""
    vec_seconds = time_vector(4, 200)
    assert np.allclose(vec_seconds, np.array([0.0, 0.005, 0.01, 0.015]))
    vec_ms = time_vector(np.zeros((4, 1)), 100, unit="ms")
    assert np.allclose(vec_ms, np.array([0.0, 10.0, 20.0, 30.0]))
