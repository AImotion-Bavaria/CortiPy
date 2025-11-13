"""Shared execution context that is passed between cortipy modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, MutableMapping, Optional

import numpy as np


ParamsDict = MutableMapping[str, Any]


@dataclass
class ModuleContext:
    """Mutable state container shared between measurement modules."""

    params: ParamsDict
    device: Optional["DeviceInterface"] = None
    data_buffer: Optional[np.ndarray] = None
    services: Dict[str, Any] = field(default_factory=dict)

    def reset_data_buffer(self) -> None:
        """Clear the transient acquisition buffer."""
        self.data_buffer = None

    def append_data(self, chunk: np.ndarray) -> None:
        """Append a numpy chunk to the buffer, creating it if needed."""
        if chunk is None:
            return
        if self.data_buffer is None:
            self.data_buffer = np.asarray(chunk)
        else:
            self.data_buffer = np.vstack([self.data_buffer, np.asarray(chunk)])

    def get_data(self) -> Optional[np.ndarray]:
        return self.data_buffer

    def attach_service(self, name: str, service: Any) -> None:
        self.services[name] = service

    def get_service(self, name: str, default: Any = None) -> Any:
        return self.services.get(name, default)

    def flush_data_to_params(self, field: str = "data") -> None:
        if self.data_buffer is None:
            return
        self.params[field] = self.data_buffer

    def release_device(self) -> None:
        """Disconnect and clear the active device, if any."""
        if self.device is not None:
            try:
                self.device.disconnect()
            except Exception:
                pass
        self.device = None


# Imported lazily to avoid circular dependencies
from cortipy.devices.base import DeviceInterface  # noqa: E402  isort:skip
