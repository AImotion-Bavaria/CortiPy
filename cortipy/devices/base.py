"""Device abstraction layer.

The class hierarchy mirrors the MATLAB `DeviceInterface` contract used by the
legacy pipeline: concrete adapters must expose ``connect()``, ``acquire()``,
and ``disconnect()`` while ``prime()`` provides an optional warm-up shot that
defaults to ``acquire()``.  Keeping the interface identical ensures the same
device names (UNICORN, ActiCHamp, generic LSL streams, offline playback, etc.)
can be wired into the Python modules without touching their acquisition loops.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Optional, Protocol, Type, TypeVar, runtime_checkable


DeviceChunk = Any
DeviceT = TypeVar("DeviceT", bound="DeviceInterface")
ExcType = Optional[Type[BaseException]]
ExcValue = Optional[BaseException]
Traceback = Optional[TracebackType]


@runtime_checkable
class DeviceLike(Protocol):
    """Structural typing hook for anything that behaves like a device."""

    def connect(self) -> None:
        ...

    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> DeviceChunk:
        ...

    def disconnect(self) -> None:
        ...

    def prime(self, duration_seconds: float, aux_channels: int = 0) -> DeviceChunk:
        ...


class DeviceInterface(DeviceLike, ABC):
    """Contract that hardware adapters must implement."""

    def __enter__(self: DeviceT) -> DeviceT:
        """Allow ``with Device() as dev`` usage mirroring MATLAB sessions."""
        self.connect()
        return self

    def __exit__(self, exc_type: ExcType, exc: ExcValue, tb: Traceback) -> bool:
        self.disconnect()
        return False  # never swallow exceptions

    @abstractmethod
    def connect(self) -> None:
        """Establish the underlying hardware/stream connection."""

    @abstractmethod
    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> DeviceChunk:
        """Capture ``duration_seconds`` worth of samples (+ optional AUX)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the device and free any buffers/resources."""

    def prime(self, duration_seconds: float, aux_channels: int = 0) -> DeviceChunk:
        """Optional warm-up acquisition that defaults to ``acquire``."""
        return self.acquire(duration_seconds, aux_channels)

    def close(self) -> None:
        """Compatibility shim for contexts expecting a ``close`` method."""
        self.disconnect()
