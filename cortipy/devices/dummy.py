"""Simple deterministic dummy device for testing cortipy pipelines."""

from __future__ import annotations

import numpy as np

from .base import DeviceInterface


class DummyDevice(DeviceInterface):
    """Generates random Gaussian noise at the configured sampling rate."""

    def __init__(self, channel_count: int = 8, sampling_rate: float = 250.0, noise: float = 0.1) -> None:
        self.channel_count = channel_count
        self.sampling_rate = sampling_rate
        self.noise = noise
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("DummyDevice must be connected before acquiring.")
        total_channels = self.channel_count + int(aux_channels)
        samples = max(1, int(round(duration_seconds * self.sampling_rate)))
        rng = np.random.default_rng()
        return self.noise * rng.standard_normal((samples, total_channels))

    def disconnect(self) -> None:
        self._connected = False
