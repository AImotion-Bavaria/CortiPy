"""Live data acquisition via Lab Streaming Layer."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from .base import DeviceInterface

# These globals are populated lazily to avoid importing pylsl (and its native
# liblsl dependency) unless an LSL device is actually used.
StreamInlet = None
resolve_stream = None

if TYPE_CHECKING:  # pragma: no cover
    from pylsl import StreamInlet as PylslStreamInlet  # noqa: F401


def _ensure_pylsl_loaded() -> None:
    """Load pylsl only when needed so non-LSL flows don't trigger native loads."""
    global StreamInlet, resolve_stream
    if StreamInlet is not None and resolve_stream is not None:
        return
    try:
        from pylsl import StreamInlet as _StreamInlet, resolve_stream as _resolve_stream
    except Exception as exc:  # pragma: no cover - exercised when pylsl unavailable
        raise RuntimeError(
            "pylsl is not available or failed to load. "
            "Install pylsl>=1.16 to use LSLDevice."
        ) from exc
    StreamInlet = _StreamInlet
    resolve_stream = _resolve_stream


class LSLDevice(DeviceInterface):
    """Simple Lab Streaming Layer (LSL) consumer."""

    def __init__(
        self,
        stream_name: str,
        channel_count: int,
        sampling_rate: float | None,
        timeout: float = 5.0,
        chunk_size: int = 1024,
    ) -> None:
        self.stream_name = stream_name
        self.channel_count = channel_count
        self.sampling_rate = sampling_rate
        self.timeout = timeout
        self.chunk_size = chunk_size
        self._inlet: Optional["PylslStreamInlet"] = None

    def connect(self) -> None:
        _ensure_pylsl_loaded()
        streams = resolve_stream("name", self.stream_name, timeout=self.timeout)
        if not streams:
            raise RuntimeError(f"Unable to resolve LSL stream named '{self.stream_name}'.")
        self._inlet = StreamInlet(streams[0], max_buflen=self.chunk_size * 2)
        if self.sampling_rate is None:
            self.sampling_rate = float(self._inlet.info().nominal_srate())
        if self.channel_count is None:
            self.channel_count = int(self._inlet.info().channel_count())

    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> np.ndarray:
        if self._inlet is None:
            raise RuntimeError("LSLDevice is not connected.")
        if self.sampling_rate is None:
            raise RuntimeError("Sampling rate is unknown; connect before acquiring.")

        samples_needed = max(1, int(round(duration_seconds * self.sampling_rate)))
        collected = []
        while sum(len(block) for block in collected) < samples_needed:
            chunk, _ = self._inlet.pull_chunk(timeout=self.timeout, max_samples=self.chunk_size)
            if not chunk:
                continue
            collected.append(np.asarray(chunk))
        data = np.vstack(collected)
        if data.shape[1] > self.channel_count:
            data = data[:, : self.channel_count]
        return data[:samples_needed]

    def disconnect(self) -> None:
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:
                pass
        self._inlet = None
