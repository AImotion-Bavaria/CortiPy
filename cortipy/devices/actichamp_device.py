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
import logging
import os
import subprocess
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional, TextIO

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ctypes equivalents for std::atomic<T> in SharedBuffer.h
#
# Assumptions (typical MSVC x64):
#   sizeof(std::atomic<int>)   == 4
#   sizeof(std::atomic<float>) == 4
#   sizeof(std::atomic<bool>)  == 1
#
# If in your build sizeof(std::atomic<bool>) == 4, change AtomicBool below
# to ctypes.c_uint32 and keep everything else the same.
# ---------------------------------------------------------------------------

AtomicInt = ctypes.c_int32
AtomicFloat = ctypes.c_float
AtomicBool = ctypes.c_bool


class ControlBlock(ctypes.Structure):
    _fields_ = [
        # std::atomic<float> targetSamplingRate;
        ("targetSamplingRate", AtomicFloat),

        # std::atomic<bool> stopRequested;
        ("stopRequested", AtomicBool),

        # std::atomic<bool> useActiveElectrodes;
        ("useActiveElectrodes", AtomicBool),

        # following: std::atomic<bool> ... all flags
        ("streamData", AtomicBool),
        ("saveToFile", AtomicBool),
        ("recalibrateAmp", AtomicBool),
        ("measureImpedance", AtomicBool),
        ("autoRangeImpedance", AtomicBool),
        ("useDCMode", AtomicBool),
        ("enableHPFilter", AtomicBool),
        ("enableLPFilter", AtomicBool),
        ("enableNotchFilter", AtomicBool),
        ("enableActiveShield", AtomicBool),
        ("enableBiasDrive", AtomicBool),
        ("useCommonReference", AtomicBool),
        ("useDrivenRightLeg", AtomicBool),
        ("enableLEDs", AtomicBool),
        ("blinkLEDs", AtomicBool),
        ("showImpedanceLEDs", AtomicBool),
        ("requestStatusUpdate", AtomicBool),
        ("requestBatteryStatus", AtomicBool),
        ("requestTemperature", AtomicBool),
    ]


class SharedBuffer(ctypes.Structure):
    _fields_ = [
        # std::atomic<int> writeIndex;
        ("writeIndex", AtomicInt),

        # std::atomic<int> readIndex;
        ("readIndex", AtomicInt),

        # float data[BUFFER_SIZE][MAX_CHANNELS];
        ("data", (ctypes.c_float * MAX_CHANNELS) * BUFFER_SIZE),

        # std::atomic<int> lostSamples;
        ("lostSamples", AtomicInt),

        # nested control block
        ("control", ControlBlock),

        # float impedances[MAX_CHANNELS + 2];
        ("impedances", ctypes.c_float * (MAX_CHANNELS + 2)),

        # std::atomic<int> impSize;
        ("impSize", AtomicInt),

        # std::atomic<bool> acquisitionReady;
        ("acquisitionReady", AtomicBool),
    ]


class _MemoryBasicInformation(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", wintypes.LPVOID),
        ("AllocationBase", wintypes.LPVOID),
        ("AllocationProtect", wintypes.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wintypes.DWORD),
        ("Protect", wintypes.DWORD),
        ("Type", wintypes.DWORD),
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
        self._producer_log: Optional[TextIO] = None
        self._buffer: ctypes.POINTER(SharedBuffer) | None = None
        self._buffer_ptr: int | None = None
        self._data_view: np.ndarray | None = None
        self._spawned_producer = False

        # Protect shared-memory access – important with Streamlit reruns / threads.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def connect(self) -> None:
        with self._lock:
            logger.info(
                "Connecting to ActiChamp (fs=%s, channels=%s, aux=%s, install_dir=%s)",
                self.sampling_rate,
                self.channel_count,
                self.default_aux,
                self.install_dir,
            )
            if os.name != "nt":
                logger.error("ActiChamp device requires Windows (detected os.name=%s)", os.name)
                raise RuntimeError("ActiChamp device requires Windows and the vendor SDK DLLs.")

            try:
                self._map_shared_buffer()
            except RuntimeError as exc:
                logger.info("Shared memory not available yet; attempting to start producer: %s", exc)
                self._start_producer()
                self._map_shared_buffer()

            self._initialize_buffer_state()
            self._wait_for_acquisition_ready()
            logger.info("ActiChamp acquisition ready for sampling.")

    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> np.ndarray:
        with self._lock:
            if self._buffer is None or self._data_view is None:
                raise RuntimeError("ActiChamp device is not connected.")

            samples_needed = max(1, int(round(duration_seconds * self.sampling_rate)))
            channel_limit = self._channel_limit(aux_channels)
            out = np.zeros((samples_needed, channel_limit), dtype=float)

            buf = self._buf

            copied = 0
            while copied < samples_needed:
                write_idx = int(buf.writeIndex)
                read_idx = int(buf.readIndex)
                available = write_idx - read_idx

                if available <= 0:
                    if bool(buf.control.stopRequested):
                        logger.info("ActiChamp producer requested stop; breaking acquisition loop.")
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

                buf.readIndex = read_idx

            return out[:copied]

    def disconnect(self) -> None:
        with self._lock:
            logger.info("Disconnecting ActiChamp device.")
            if self._buffer is not None:
                try:
                    self._buf.control.stopRequested = True
                except RuntimeError:
                    logger.warning("Shared memory invalid while setting stopRequested during disconnect.")
                self._signal_stop_event()

            if self._process is not None:
                try:
                    self._process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    logger.warning("ActiChamp producer did not exit in time; killing.")
                    self._process.kill()
            self._process = None
            self._spawned_producer = False

            if self._producer_log is not None:
                try:
                    self._producer_log.flush()
                    self._producer_log.close()
                except Exception:
                    logger.exception("Failed to close ActiChamp producer log file")
                self._producer_log = None

            if self._buffer_ptr is not None:
                ctypes.windll.kernel32.UnmapViewOfFile(ctypes.c_void_p(self._buffer_ptr))
            self._buffer_ptr = None
            self._buffer = None
            self._data_view = None

    def prime(self, duration_seconds: float, aux_channels: int = 0) -> np.ndarray:
        return self.acquire(duration_seconds, aux_channels)

    def read_impedances(self) -> list[float]:
        """Return impedance values from the shared control block."""
        with self._lock:
            if self._buffer is None:
                raise RuntimeError("ActiChamp shared memory is not mapped.")

            buf = self._buf
            size = int(buf.impSize)
            if size <= 0:
                return []

            limit = min(size, MAX_CHANNELS + 2)
            return [float(buf.impedances[i]) for i in range(limit)]

    # ------------------------------------------------------------------
    def _channel_limit(self, aux_channels: int) -> int:
        total = self.channel_count + int(aux_channels or self.default_aux)
        if self.include_triggers:
            total += 1
        return min(total, MAX_CHANNELS)

    def _start_producer(self) -> None:
        exe_path = self.install_dir / "EEG_SharedMemoryProducer.exe"
        if not exe_path.exists():
            logger.error("ActiChamp producer executable not found at %s", exe_path)
            raise FileNotFoundError(f"ActiChamp producer executable not found at {exe_path}")

        if self._process is not None:
            if self._process.poll() is None:
                logger.info("ActiChamp producer already running (pid=%s); reusing it", self._process.pid)
                return
            logger.info("Previous ActiChamp producer exited (code=%s); restarting", self._process.poll())
            self._process = None
            self._spawned_producer = False

        log_path = self.install_dir / "EEG_SharedMemoryProducer.log"
        log_file: Optional[TextIO] = None

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf-8")
            self._process = subprocess.Popen(
                [str(exe_path)],
                cwd=str(self.install_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
            )
            self._producer_log = log_file
            logger.info("Started ActiChamp producer at %s (log: %s)", exe_path, log_path)
        except Exception:
            if log_file is not None:
                log_file.close()
            logger.exception("Failed to launch ActiChamp producer executable")
            raise

        self._spawned_producer = True

    def _map_shared_buffer(self) -> None:
        kernel32 = ctypes.windll.kernel32
        size = ctypes.sizeof(SharedBuffer)
        deadline = time.time() + self.timeout
        handle = None
        ptr = None

        logger.debug("Mapping ActiChamp shared memory block '%s'", SHM_NAME)

        while time.time() < deadline:
            handle = kernel32.OpenFileMappingW(FILE_MAP_ALL_ACCESS, False, SHM_NAME)
            if handle:
                ptr = kernel32.MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size)
                kernel32.CloseHandle(handle)
                if ptr:
                    break
            time.sleep(0.05)

        if not ptr:
            logger.error("Unable to map ActiChamp shared memory block '%s' before timeout", SHM_NAME)
            raise RuntimeError("Unable to map ActiChamp shared memory. Is the producer running?")

        expected_size = ctypes.sizeof(SharedBuffer)
        mbi = _MemoryBasicInformation()
        if kernel32.VirtualQuery(ctypes.c_void_p(ptr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            if mbi.RegionSize < expected_size:
                ctypes.windll.kernel32.UnmapViewOfFile(ctypes.c_void_p(ptr))
                logger.error(
                    "ActiChamp shared memory layout mismatch (got %s bytes, expected >= %s). "
                    "Rebuild the producer with the current SharedBuffer.h.",
                    mbi.RegionSize,
                    expected_size,
                )
                raise RuntimeError("ActiChamp shared memory layout mismatch.")

        self._buffer_ptr = ptr
        self._buffer = ctypes.cast(ptr, ctypes.POINTER(SharedBuffer))

        try:
            buf = self._buf
            # buf.data is a (BUFFER_SIZE x MAX_CHANNELS) ctypes 2D array.
            first_row = buf.data[0]
            data_addr = ctypes.addressof(first_row)
            flat_type = ctypes.c_float * (BUFFER_SIZE * MAX_CHANNELS)
            flat = flat_type.from_address(data_addr)
            self._data_view = np.ndarray(
                (BUFFER_SIZE, MAX_CHANNELS), dtype=np.float32, buffer=flat
            )
        except Exception as exc:
            ctypes.windll.kernel32.UnmapViewOfFile(ctypes.c_void_p(ptr))
            self._buffer_ptr = None
            self._buffer = None
            logger.error("Failed to initialize ActiChamp buffer view: %s", exc)
            raise RuntimeError("Failed to initialize ActiChamp shared buffer view.") from exc

        logger.info("Mapped ActiChamp shared memory and initialized buffer view.")

    def _initialize_buffer_state(self) -> None:
        if self._buffer is None or self._data_view is None:
            raise RuntimeError("ActiChamp shared memory is not mapped.")

        logger.debug("Initializing ActiChamp buffer state (spawned=%s)", self._spawned_producer)

        buf = self._buf
        if self._spawned_producer:
            buf.writeIndex = 0
            buf.readIndex = 0
            buf.lostSamples = 0
            buf.control.stopRequested = False
            self._data_view.fill(0.0)
            buf.acquisitionReady = False
        else:
            buf.control.stopRequested = False
            buf.readIndex = buf.writeIndex

        buf.control.targetSamplingRate = self.sampling_rate
        buf.control.useActiveElectrodes = bool(self.use_active_electrodes)
        logger.debug(
            "ActiChamp control block updated (fs=%s, activeElectrodes=%s)",
            self.sampling_rate,
            self.use_active_electrodes,
        )

    def _wait_for_acquisition_ready(self) -> None:
        if self._buffer is None:
            raise RuntimeError("ActiChamp shared memory is not mapped.")

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if bool(self._buf.acquisitionReady):
                logger.info("ActiChamp acquisition flag reported ready.")
                return
            time.sleep(0.01)
        logger.error("Timed out waiting for ActiChamp acquisition to become ready.")
        raise TimeoutError("Timed out waiting for ActiChamp acquisition to become ready.")

    @property
    def _buf(self) -> SharedBuffer:
        """Safely dereference the shared buffer pointer."""
        if self._buffer is None:
            logger.error("ActiChamp shared memory is not mapped in _buf()")
            raise RuntimeError("ActiChamp shared memory is not mapped.")

        ptr_val = ctypes.cast(self._buffer, ctypes.c_void_p).value
        if not ptr_val:
            logger.error("ActiChamp shared memory pointer is null in _buf()")
            raise RuntimeError("ActiChamp shared memory pointer is null.")

        kernel32 = ctypes.windll.kernel32
        mbi = _MemoryBasicInformation()
        res = kernel32.VirtualQuery(
            ctypes.c_void_p(ptr_val),
            ctypes.byref(mbi),
            ctypes.sizeof(mbi),
        )
        expected_size = ctypes.sizeof(SharedBuffer)

        if not res:
            logger.error("VirtualQuery failed for shared memory pointer 0x%X", ptr_val)
            raise RuntimeError("ActiChamp shared memory pointer is invalid (VirtualQuery failed).")

        if mbi.RegionSize < expected_size:
            logger.error(
                "ActiChamp shared memory region too small before dereference "
                "(RegionSize=%s, expected>=%s).",
                mbi.RegionSize,
                expected_size,
            )
            raise RuntimeError("ActiChamp shared memory region too small before dereference.")

        try:
            return self._buffer.contents
        except (ValueError, OSError) as exc:
            logger.exception("ActiChamp shared memory pointer is invalid or unmapped.")
            raise RuntimeError("ActiChamp shared memory pointer is invalid or unmapped.") from exc
        except Exception as exc:
            logger.exception("Unexpected error while dereferencing ActiChamp shared memory pointer.")
            raise RuntimeError("Unexpected error while dereferencing ActiChamp shared memory pointer.") from exc

    def _signal_stop_event(self) -> None:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, False, STOP_EVENT_NAME)
        if handle:
            kernel32.SetEvent(handle)
            kernel32.CloseHandle(handle)