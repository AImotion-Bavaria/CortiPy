"""Brain Products ActiCHamp shared-memory device adapter.

This adapter mirrors the MATLAB helpers that ship with the vendor SDK:

- ``EEG_SharedMemoryProducer.exe`` streams samples into a named shared memory
  block (``EEG_SharedMemory``) and waits for a sampling rate via
  ``targetSamplingRate``.
- Consumers read samples from the ring buffer and advance the shared
  ``readIndex`` just like the `ReadHybridMemory` MEX function.

The Python implementation keeps that layout intact so the existing DLLs and
math remain untouched. Only Windows is supported because the SDK relies on
Win32 shared-memory primitives and vendor DLLs.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import DeviceInterface


BUFFER_SIZE = 200000
MAX_CHANNELS = 43
SHM_NAME = "EEG_SharedMemory"
STOP_EVENT_NAME = "EEG_StopEvent"

FILE_MAP_ALL_ACCESS = 0xF001F
EVENT_MODIFY_STATE = 0x0002
SYNCHRONIZE = 0x00100000
CREATE_NO_WINDOW = 0x08000000


class ControlBlock(ctypes.Structure):
    _fields_ = [
        ("targetSamplingRate", ctypes.c_float),
        ("stopRequested", ctypes.c_bool),
        ("useActiveElectrodes", ctypes.c_bool),
        ("streamData", ctypes.c_bool),
        ("saveToFile", ctypes.c_bool),
        ("recalibrateAmp", ctypes.c_bool),
        ("measureImpedance", ctypes.c_bool),
        ("autoRangeImpedance", ctypes.c_bool),
        ("useDCMode", ctypes.c_bool),
        ("enableHPFilter", ctypes.c_bool),
        ("enableLPFilter", ctypes.c_bool),
        ("enableNotchFilter", ctypes.c_bool),
        ("enableActiveShield", ctypes.c_bool),
        ("enableBiasDrive", ctypes.c_bool),
        ("useCommonReference", ctypes.c_bool),
        ("useDrivenRightLeg", ctypes.c_bool),
        ("enableLEDs", ctypes.c_bool),
        ("blinkLEDs", ctypes.c_bool),
        ("showImpedanceLEDs", ctypes.c_bool),
        ("requestStatusUpdate", ctypes.c_bool),
        ("requestBatteryStatus", ctypes.c_bool),
        ("requestTemperature", ctypes.c_bool),
    ]


class SharedBuffer(ctypes.Structure):
    _fields_ = [
        ("writeIndex", ctypes.c_int32),
        ("readIndex", ctypes.c_int32),
        ("data", ctypes.c_float * (BUFFER_SIZE * MAX_CHANNELS)),
        ("lostSamples", ctypes.c_int32),
        ("control", ControlBlock),
        ("impedances", ctypes.c_float * (MAX_CHANNELS + 2)),
        ("impSize", ctypes.c_int32),
        ("acquisitionReady", ctypes.c_bool),
    ]


class ActiChampDevice(DeviceInterface):
    def __init__(
        self,
        sampling_rate: float,
        channel_count: int,
        aux_channels: int = 0,
        install_dir: str | Path | None = None,
        timeout: float = 10.0,
        include_triggers: bool = True,
        use_active_electrodes: bool = False,
    ) -> None:
        self.sampling_rate = float(sampling_rate)
        self.channel_count = int(channel_count)
        self.default_aux = int(aux_channels)
        self.install_dir = Path(install_dir) if install_dir else Path(__file__).resolve().parent / "actichamp"
        self.timeout = float(timeout)
        self.include_triggers = include_triggers
        self.use_active_electrodes = use_active_electrodes

        self._process: Optional[subprocess.Popen[bytes]] = None
        self._buffer: SharedBuffer | None = None
        self._buffer_ptr: int | None = None
        self._data_view: np.ndarray | None = None
        self._spawned_producer = False

    # ------------------------------------------------------------------
    def connect(self) -> None:
        if os.name != "nt":
            raise RuntimeError("ActiChamp device requires Windows and the vendor SDK DLLs.")

        try:
            self._map_shared_buffer()
        except RuntimeError:
            self._start_producer()
            self._map_shared_buffer()

        self._initialize_buffer_state()
        self._wait_for_acquisition_ready()

    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> np.ndarray:
        if self._buffer is None or self._data_view is None:
            raise RuntimeError("ActiChamp device is not connected.")

        samples_needed = max(1, int(round(duration_seconds * self.sampling_rate)))
        channel_limit = self._channel_limit(aux_channels)
        out = np.zeros((samples_needed, channel_limit), dtype=float)

        copied = 0
        while copied < samples_needed:
            write_idx = int(self._buffer.writeIndex)
            read_idx = int(self._buffer.readIndex)
            available = write_idx - read_idx

            if available <= 0:
                if self._buffer.control.stopRequested:
                    break
                time.sleep(0.001)
                continue

            to_copy = min(available, samples_needed - copied)
            start = read_idx % BUFFER_SIZE
            first_block = min(to_copy, BUFFER_SIZE - start)

            if first_block > 0:
                block = self._data_view[start : start + first_block, :channel_limit]
                out[copied : copied + first_block, :] = block
                copied += first_block
                read_idx += first_block

            remaining = to_copy - first_block
            if remaining > 0:
                block = self._data_view[:remaining, :channel_limit]
                out[copied : copied + remaining, :] = block
                copied += remaining
                read_idx += remaining

            self._buffer.readIndex = read_idx

        return out[:copied]

    def disconnect(self) -> None:
        if self._buffer is not None:
            self._buffer.control.stopRequested = True
            self._signal_stop_event()

        if self._process is not None:
            try:
                self._process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._spawned_producer = False

        if self._buffer_ptr is not None:
            ctypes.windll.kernel32.UnmapViewOfFile(ctypes.c_void_p(self._buffer_ptr))
        self._buffer_ptr = None
        self._buffer = None
        self._data_view = None

    def prime(self, duration_seconds: float, aux_channels: int = 0) -> np.ndarray:
        return self.acquire(duration_seconds, aux_channels)

    def read_impedances(self) -> list[float]:
        """Return impedance values from the shared control block.

        The producer populates ``impedances`` before regular acquisition
        starts, exposing GND/REF followed by channel impedances. Values are
        reported in Ohms; unavailable entries remain negative.
        """

        if self._buffer is None:
            raise RuntimeError("ActiChamp shared memory is not mapped.")

        size = int(self._buffer.impSize)
        if size <= 0:
            return []

        limit = min(size, MAX_CHANNELS + 2)
        return [float(self._buffer.impedances[i]) for i in range(limit)]

    # ------------------------------------------------------------------
    def _channel_limit(self, aux_channels: int) -> int:
        total = self.channel_count + int(aux_channels or self.default_aux)
        if self.include_triggers:
            total += 1
        return min(total, MAX_CHANNELS)

    def _start_producer(self) -> None:
        exe_path = self.install_dir / "EEG_SharedMemoryProducer.exe"
        if not exe_path.exists():
            raise FileNotFoundError(f"ActiChamp producer executable not found at {exe_path}")

        self._process = subprocess.Popen(
            [str(exe_path)],
            cwd=str(self.install_dir),
            creationflags=CREATE_NO_WINDOW,
        )
        self._spawned_producer = True

    def _map_shared_buffer(self) -> None:
        kernel32 = ctypes.windll.kernel32
        size = ctypes.sizeof(SharedBuffer)
        deadline = time.time() + self.timeout
        handle = None
        ptr = None

        while time.time() < deadline:
            handle = kernel32.OpenFileMappingW(FILE_MAP_ALL_ACCESS, False, SHM_NAME)
            if handle:
                ptr = kernel32.MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size)
                kernel32.CloseHandle(handle)
                if ptr:
                    break
            time.sleep(0.05)

        if not ptr:
            raise RuntimeError("Unable to map ActiChamp shared memory. Is the producer running?")

        self._buffer_ptr = ptr
        self._buffer = SharedBuffer.from_address(ptr)
        data = np.ctypeslib.as_array(self._buffer.data)
        self._data_view = data.reshape(BUFFER_SIZE, MAX_CHANNELS)

    def _initialize_buffer_state(self) -> None:
        if self._buffer is None or self._data_view is None:
            raise RuntimeError("ActiChamp shared memory is not mapped.")

        if self._spawned_producer:
            self._buffer.writeIndex = 0
            self._buffer.readIndex = 0
            self._buffer.lostSamples = 0
            self._buffer.control.stopRequested = False
            self._data_view.fill(0.0)
            self._buffer.acquisitionReady = False

        self._buffer.control.targetSamplingRate = self.sampling_rate
        self._buffer.control.useActiveElectrodes = bool(self.use_active_electrodes)

    def _wait_for_acquisition_ready(self) -> None:
        if self._buffer is None:
            raise RuntimeError("ActiChamp shared memory is not mapped.")

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if bool(self._buffer.acquisitionReady):
                return
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for ActiChamp acquisition to become ready.")

    def _signal_stop_event(self) -> None:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, False, STOP_EVENT_NAME)
        if handle:
            kernel32.SetEvent(handle)
            kernel32.CloseHandle(handle)
