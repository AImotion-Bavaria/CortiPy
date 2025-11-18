"""Offline data playback device."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat

from .base import DeviceInterface


class OfflineDevice(DeviceInterface):
    """Feeds prerecorded data chunks to the pipeline."""

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        data_path: Optional[str | Path] = None,
        sampling_rate: Optional[float] = None,
    ) -> None:
        if data is None and data_path is None:
            raise ValueError("OfflineDevice requires either `data` or `data_path`.")
        self._data = np.asarray(data) if data is not None else None
        self._data_path = Path(data_path) if data_path is not None else None
        self.sampling_rate = sampling_rate
        self._cursor = 0

    def connect(self) -> None:
        if self._data is None and self._data_path is not None:
            self._data = self._load_file(self._data_path)
        self._cursor = 0

    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("OfflineDevice is not connected or data failed to load.")
        if self.sampling_rate is None:
            raise RuntimeError("OfflineDevice requires a sampling_rate for timed acquisition.")

        samples = max(1, int(round(duration_seconds * self.sampling_rate)))
        start = self._cursor
        end = min(start + samples, self._data.shape[0])
        chunk = self._data[start:end]
        self._cursor = end
        return chunk

    def disconnect(self) -> None:
        self._cursor = 0

    def _load_file(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path)
        if suffix == ".npz":
            data = np.load(path)
            first_key = next(iter(data.files))
            return data[first_key]
        if suffix in {".csv", ".txt"}:
            return np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
        if suffix == ".mat":
            mat = loadmat(path)
            for key, value in mat.items():
                if key.startswith("__"):
                    continue
                if isinstance(value, np.ndarray):
                    return np.asarray(value)
            raise RuntimeError(f"No numeric array found in MAT file {path}.")
        raise RuntimeError(f"Unsupported offline data format: {path.suffix}")
