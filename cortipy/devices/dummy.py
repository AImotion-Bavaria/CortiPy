"""Simple deterministic dummy device for testing cortipy pipelines."""

from __future__ import annotations

import numpy as np

from .base import DeviceInterface


class DummyDevice(DeviceInterface):
    """Generates synthetic EEG-like data (multiple oscillatory bands + noise)."""

    def __init__(
        self,
        channel_count: int = 8,
        sampling_rate: float = 250.0,
        noise: float = 3.0,
        seed: int | None = 42,
    ) -> None:
        self.channel_count = max(1, int(channel_count))
        self.sampling_rate = float(sampling_rate)
        self.noise = float(noise)
        self.seed = seed
        self._connected = False
        self._rng = np.random.default_rng(seed)
        self._phases = self._rng.uniform(0, 2 * np.pi, size=self.channel_count + 8)
        self._sample_index = 0
        # Approximate microvolt amplitudes for canonical bands.
        self._bands = [
            (1.5, 18.0),  # delta
            (6.0, 12.0),  # theta
            (10.0, 15.0),  # alpha
            (18.0, 8.0),  # beta
            (35.0, 4.0),  # gamma
        ]
        self._drift = (0.3, 6.0)  # slow drift

    def connect(self) -> None:
        self._connected = True
        self._sample_index = 0

    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("DummyDevice must be connected before acquiring.")
        total_channels = self.channel_count + int(aux_channels)
        samples = max(1, int(round(duration_seconds * self.sampling_rate)))
        t = (self._sample_index + np.arange(samples)) / self.sampling_rate
        self._sample_index += samples

        def synth_for_channel(ch_idx: int) -> np.ndarray:
            phase = self._phases[ch_idx % len(self._phases)]
            sig = np.zeros(samples, dtype=float)
            for freq, amp in self._bands:
                sig += amp * np.sin(2 * np.pi * freq * t + phase)
            drift_freq, drift_amp = self._drift
            sig += drift_amp * np.sin(2 * np.pi * drift_freq * t + phase)
            # Add modest pink-ish noise by filtering white noise with cumulative sum factor.
            white = self._rng.standard_normal(samples)
            pinkish = np.convolve(white, np.ones(3) / 3, mode="same")
            sig += self.noise * pinkish
            # Random per-channel gain (within 15%) for variety.
            gain = 1.0 + 0.15 * (self._rng.random() - 0.5)
            return gain * sig

        data = np.column_stack([synth_for_channel(ch) for ch in range(total_channels)])
        return data

    def prime(self, duration_seconds: float, aux_channels: int = 0) -> np.ndarray:
        """Warm-up call mirrors acquire to feed live-preview windows."""
        return self.acquire(duration_seconds, aux_channels)

    def disconnect(self) -> None:
        self._connected = False
