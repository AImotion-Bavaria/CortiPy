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

            # Wall-clock safety cap: sized from the requested duration so a producer that
            # streams at a different rate than `self.sampling_rate` cannot make a short
            # recording block for minutes (see RecordingTime overshoot reports).
            start_time = time.perf_counter()
            deadline = start_time + max(1.0, duration_seconds) * 3.0 + 5.0

            copied = 0
            while copied < samples_needed:
                if time.perf_counter() > deadline:
                    logger.warning(
                        "ActiChamp.acquire timed out: got %d/%d samples in %.1fs for a %.1fs "
                        "request @ %.0f Hz (effective ~%.0f Hz). Check that the producer honors "
                        "the requested sampling rate.",
                        copied, samples_needed, time.perf_counter() - start_time,
                        duration_seconds, self.sampling_rate,
                        copied / max(time.perf_counter() - start_time, 1e-6),
                    )
                    break
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

            elapsed = time.perf_counter() - start_time
            effective_fs = copied / elapsed if elapsed > 0 else self.sampling_rate
            if self.sampling_rate > 0 and abs(effective_fs - self.sampling_rate) > 0.15 * self.sampling_rate:
                logger.warning(
                    "ActiChamp.acquire delivered %d samples in %.2fs (~%.0f Hz) but fs is set to "
                    "%.0f Hz; RecordingTime will be off by ~%.1fx.",
                    copied, elapsed, effective_fs, self.sampling_rate,
                    self.sampling_rate / max(effective_fs, 1e-6),
                )
            else:
                logger.debug("ActiChamp.acquire: %d samples in %.2fs (~%.0f Hz)", copied, elapsed, effective_fs)

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

    def prepare_for_recording(self) -> None:
        with self._lock:
            if self._buffer is None:
                return
            self._buf.control.stopRequested = False
            self._buf.readIndex = self._buf.writeIndex

    def read_impedances(
        self,
        wait_seconds: float = 6.0,
        poll_interval: float = 0.1,
        settle_seconds: float = 1.0,
    ) -> list[float]:
        """Return impedance values from the shared control block.

        Impedance data is produced before a sampling rate is written to the
        shared control block.  The normal ``connect`` path sets that sampling
        rate and switches the producer into acquisition mode, so this method
        maps or starts the producer without calling ``connect`` when needed.
        """
        with self._lock:
            if self._buffer is None:
                try:
                    self._map_shared_buffer()
                except RuntimeError as exc:
                    logger.info("Shared memory not available for impedance read; starting producer: %s", exc)
                    self._start_producer()
                    self._map_shared_buffer()

            buf = self._buf
            buf.control.stopRequested = False
            buf.control.targetSamplingRate = 0.0
            # Tell the producer to switch the amplifier into impedance mode. Without this the
            # producer never measures impedances, so impSize stays 0 and every value stays 0 —
            # which looked like "impedance never updates".
            buf.control.measureImpedance = True
            buf.control.showImpedanceLEDs = True
            buf.impSize = 0
            for idx in range(MAX_CHANNELS + 2):
                buf.impedances[idx] = -1.0

            deadline = time.time() + max(0.0, float(wait_seconds))
            first_positive_at: float | None = None
            last_values: list[float] = []

            try:
                while True:
                    size = int(buf.impSize)
                    if size > 0:
                        limit = min(size, MAX_CHANNELS + 2)
                        last_values = [float(buf.impedances[i]) for i in range(limit)]
                        if any(value > 0 for value in last_values):
                            if first_positive_at is None:
                                first_positive_at = time.time()
                            if (time.time() - first_positive_at) >= max(0.0, float(settle_seconds)):
                                return last_values

                    if time.time() >= deadline:
                        break
                    time.sleep(max(0.01, float(poll_interval)))

                return last_values
            finally:
                # Leave impedance mode, or the next acquisition would start with the amplifier
                # still measuring impedance instead of streaming EEG.
                buf.control.measureImpedance = False
                buf.control.showImpedanceLEDs = False

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
        # robust mapping with explicit ctypes signatures and checks
        kernel32 = ctypes.windll.kernel32

        # set prototypes for safety (important!)
        kernel32.OpenFileMappingW.restype = wintypes.HANDLE
        kernel32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]

        kernel32.MapViewOfFile.restype = ctypes.c_void_p
        kernel32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                           ctypes.c_size_t]

        kernel32.VirtualQuery.restype = ctypes.c_size_t
        kernel32.VirtualQuery.argtypes = [wintypes.LPCVOID, ctypes.POINTER(_MemoryBasicInformation), ctypes.c_size_t]

        expected_size = ctypes.sizeof(SharedBuffer)
        deadline = time.time() + self.timeout

        handle = None
        ptr = None

        logger.debug("Mapping ActiChamp shared memory block '%s' (expected size=%s)", SHM_NAME, expected_size)

        # Try to open and map the whole mapping (pass 0 to MapViewOfFile to map entire mapping)
        while time.time() < deadline:
            # Open wide string explicitly (LPCWSTR)
            handle = kernel32.OpenFileMappingW(FILE_MAP_ALL_ACCESS, False, ctypes.c_wchar_p(SHM_NAME))
            if not handle:
                time.sleep(0.05)
                continue

            # Map entire mapping by specifying 0 for number of bytes (MapViewOfFile maps whole mapping if 0).
            ptr = kernel32.MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, 0)
            # Close the handle immediately — mapping remains valid.
            kernel32.CloseHandle(handle)
            handle = None

            if ptr:
                break

            time.sleep(0.05)

        if not ptr:
            logger.error("Unable to map ActiChamp shared memory block '%s' before timeout", SHM_NAME)
            raise RuntimeError("Unable to map ActiChamp shared memory. Is the producer running?")

        # Validate mapped region via VirtualQuery
        mbi = _MemoryBasicInformation()
        ok = kernel32.VirtualQuery(ctypes.c_void_p(ptr), ctypes.byref(mbi), ctypes.sizeof(mbi))
        if not ok:
            # defensive: unmap and raise
            kernel32.UnmapViewOfFile(ctypes.c_void_p(ptr))
            logger.error("VirtualQuery failed for mapped pointer 0x%016x", ptr)
            raise RuntimeError("ActiChamp shared memory pointer is invalid (VirtualQuery failed).")

        # Region must be >= expected size
        if mbi.RegionSize < expected_size:
            kernel32.UnmapViewOfFile(ctypes.c_void_p(ptr))
            logger.error(
                "ActiChamp shared memory layout mismatch (region %s bytes < expected %s bytes). Rebuild or match headers.",
                mbi.RegionSize,
                expected_size,
            )
            raise RuntimeError("ActiChamp shared memory layout mismatch.")

        # OK: set internal pointers
        self._buffer_ptr = ptr
        self._buffer = ctypes.cast(ptr, ctypes.POINTER(SharedBuffer))

        # create numpy view from the 2D float array
        try:
            # note: SharedBuffer.data is defined as array-of-rows, so get address of first element
            buf = self._buf  # will raise if pointer invalid
            # data field is an array of rows; addressof returns start of data[0]
            data_addr = ctypes.addressof(buf.data)
            flat_type = ctypes.c_float * (BUFFER_SIZE * MAX_CHANNELS)
            flat = flat_type.from_address(data_addr)
            self._data_view = np.ndarray((BUFFER_SIZE, MAX_CHANNELS), dtype=np.float32, buffer=flat)
        except Exception as exc:
            # cleanup mapping on failure
            kernel32.UnmapViewOfFile(ctypes.c_void_p(ptr))
            self._buffer_ptr = None
            self._buffer = None
            logger.exception("Failed to initialize ActiChamp buffer view: %s", exc)
            raise RuntimeError("Failed to initialize ActiChamp shared buffer view.") from exc

        logger.info("Mapped ActiChamp shared memory and initialized buffer view (ptr=0x%016x, region=%d bytes)",
                    self._buffer_ptr, mbi.RegionSize)

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
