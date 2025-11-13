"""Device abstraction layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol


class DeviceInterface(ABC):
    """Contract that hardware adapters must implement."""

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> Any:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    def prime(self, duration_seconds: float, aux_channels: int = 0) -> Any:
        """Optional warm-up acquisition that defaults to `acquire`."""
        return self.acquire(duration_seconds, aux_channels)
