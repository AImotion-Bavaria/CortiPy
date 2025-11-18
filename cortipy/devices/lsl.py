"""Live data acquisition via Lab Streaming Layer."""

from __future__ import annotations

from typing import Optional

import numpy as np
try:
    from pylsl import StreamInlet, resolve_stream
except ImportError:  # pragma: no cover
    StreamInlet = None
    resolve_stream = None

from .base import DeviceInterface


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
        self._inlet: Optional[StreamInlet] = None

    def connect(self) -> None:
        if StreamInlet is None or resolve_stream is None:
            raise RuntimeError(
                "pylsl is not available or missing resolve_stream. "
                "Install pylsl>=1.16 to use LSLDevice."
            )
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
