"""g.tec UNICORN EEG hardware adapter (serial protocol)."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Sequence

import numpy as np
import serial

from .base import DeviceInterface


@dataclass(frozen=True)
class UnicornPacket:
    eeg: np.ndarray
    accelerometer: np.ndarray
    gyroscope: np.ndarray
    battery: float
    counter: float


class UnicornDevice(DeviceInterface):
    """Serial-port implementation compatible with the MATLAB Unicorn helpers.

    The adapter mirrors `py_devices/unicorn` (and the legacy MATLAB code) by:

    * opening the UNICORN's virtual COM port at 115200 baud,
    * sending the 0x61 0x7C 0x87 start command and verifying the 3×0x00 ACK,
    * streaming fixed-size 45-byte packets and decoding EEG/IMU/battery data.

    Returned samples always contain 16 columns: EEG(8), accelerometer(3),
    gyroscope(3), battery level, and the packet counter. Modules continue to
    slice the EEG subset exactly like the MATLAB implementation
    (`data(:, 1:8)`).
    """

    BAUD_RATE = 115_200
    PACKET_BYTES = 45
    EEG_CHANNELS = 8
    TOTAL_COLUMNS = 16  # 8 EEG + 3 accel + 3 gyro + battery + counter
    EEG_SCALE = 4_500_000 / 50_331_642  # ≈ 0.08944 µV per integer unit
    ACC_SCALE = 1 / 4096.0  # g
    GYRO_SCALE = 1 / 32.8  # °/s
    START_ACQ = bytes((0x61, 0x7C, 0x87))
    START_RESPONSE = bytes((0x00, 0x00, 0x00))
    STOP_ACQ = bytes((0x63, 0x5C, 0xC5))
    START_SEQUENCE = bytes((0xC0, 0x00))
    STOP_SEQUENCE = bytes((0x0D, 0x0A))

    def __init__(
        self,
        port: str,
        *,
        device_name: str | None = None,
        sampling_rate: float = 250.0,
        timeout: float = 5.0,
    ) -> None:
        self.port = port
        self.device_name = device_name or port
        self.sampling_rate = float(sampling_rate)
        self.timeout = float(timeout)
        self._serial: Optional[serial.Serial] = None
        self._pending = bytearray()

    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self._serial is not None:
            return
        ser = serial.Serial(
            port=self.port,
            baudrate=self.BAUD_RATE,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )
        try:
            for attr in ("dtr", "rts"):
                try:
                    setattr(ser, attr, True)
                except Exception:
                    pass
            time.sleep(0.15)
            self._pending.clear()
            ack = self._send_start_command(ser)
            if ack != self.START_RESPONSE:
                # Bluetooth serial stacks can expose stale bytes immediately after opening.
                # Flush and try once more before reporting the raw ACK for diagnosis.
                ack = self._send_start_command(ser)
            if ack != self.START_RESPONSE and not self._sync_to_packet(ser, initial=ack):
                ser.close()
                raise RuntimeError(
                    f"UNICORN device '{self.device_name}' did not acknowledge the start command "
                    f"on {self.port} (got {ack!r})."
                )
        except Exception:
            if getattr(ser, "is_open", True):
                ser.close()
            self._pending.clear()
            raise
        self._serial = ser

    def acquire(self, duration_seconds: float, aux_channels: int = 0) -> np.ndarray:
        if self._serial is None:
            raise RuntimeError("UnicornDevice must be connected before acquiring.")
        del aux_channels  # UNICORN does not expose AUX lines through this protocol
        sample_count = max(1, int(round(duration_seconds * self.sampling_rate)))
        rows = np.zeros((sample_count, self.TOTAL_COLUMNS), dtype=float)
        for idx in range(sample_count):
            packet = self._read_exact(self.PACKET_BYTES)
            decoded = self._decode_packet(packet)
            rows[idx, : self.EEG_CHANNELS] = decoded.eeg
            rows[idx, self.EEG_CHANNELS : self.EEG_CHANNELS + 3] = decoded.accelerometer
            rows[idx, self.EEG_CHANNELS + 3 : self.EEG_CHANNELS + 6] = decoded.gyroscope
            rows[idx, self.EEG_CHANNELS + 6] = decoded.battery
            rows[idx, self.EEG_CHANNELS + 7] = decoded.counter
        return rows

    def disconnect(self) -> None:
        if self._serial is None:
            return
        try:
            self._serial.write(self.STOP_ACQ)
            self._serial.read(len(self.START_RESPONSE))
        except serial.SerialException:
            pass
        finally:
            try:
                self._serial.close()
            finally:
                self._serial = None
                self._pending.clear()

    def prepare_for_recording(self) -> None:
        if self._serial is not None:
            self._serial.reset_input_buffer()

    # ------------------------------------------------------------------
    def _read_exact(self, size: int) -> bytes:
        """Read exactly ``size`` bytes from the serial port or raise."""
        assert self._serial is not None
        buffer = bytearray()
        if self._pending:
            take = min(size, len(self._pending))
            buffer.extend(self._pending[:take])
            del self._pending[:take]
        while len(buffer) < size:
            chunk = self._serial.read(size - len(buffer))
            if not chunk:
                raise RuntimeError("UNICORN stream ended unexpectedly.")
            buffer.extend(chunk)
        return bytes(buffer)

    def _send_start_command(self, ser: serial.Serial) -> bytes:
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        try:
            ser.reset_output_buffer()
        except Exception:
            pass
        ser.write(self.START_ACQ)
        try:
            ser.flush()
        except Exception:
            pass
        return ser.read(len(self.START_RESPONSE))

    def _sync_to_packet(self, ser: serial.Serial, initial: bytes = b"") -> bool:
        """Align to a valid packet when a Bluetooth serial stack misses the ACK."""
        deadline = time.time() + max(0.5, min(self.timeout, 2.0))
        buffer = bytearray(initial or b"")
        max_buffer = self.PACKET_BYTES * 4
        while time.time() < deadline:
            chunk = ser.read(max(1, self.PACKET_BYTES - len(buffer)))
            if chunk:
                buffer.extend(chunk)
                while len(buffer) >= self.PACKET_BYTES:
                    start = buffer.find(self.START_SEQUENCE)
                    if start < 0:
                        del buffer[:-1]
                        break
                    if start > 0:
                        del buffer[:start]
                    if len(buffer) < self.PACKET_BYTES:
                        break
                    packet = bytes(buffer[: self.PACKET_BYTES])
                    if packet[-2:] == self.STOP_SEQUENCE:
                        self._pending.extend(packet)
                        return True
                    del buffer[0]
                if len(buffer) > max_buffer:
                    del buffer[:-self.PACKET_BYTES]
            else:
                time.sleep(0.02)
        return False

    def _decode_packet(self, packet: bytes) -> UnicornPacket:
        if len(packet) != self.PACKET_BYTES:
            raise RuntimeError(f"Invalid UNICORN packet size: {len(packet)} (expected {self.PACKET_BYTES}).")
        if packet[:2] != self.START_SEQUENCE:
            raise RuntimeError("Malformed UNICORN packet header.")
        if packet[-2:] != self.STOP_SEQUENCE:
            raise RuntimeError("Malformed UNICORN packet footer.")

        eeg = np.empty(self.EEG_CHANNELS, dtype=float)
        for ch in range(self.EEG_CHANNELS):
            start = 3 + ch * 3
            raw = packet[start : start + 3]
            value = raw[0] * 256**2 + raw[1] * 256 + raw[2]
            if value & 0x800000:  # signed 24-bit
                value -= 1 << 24
            eeg[ch] = float(value) * self.EEG_SCALE

        accel = np.array(
            [
                self._read_int16(packet, 27) * self.ACC_SCALE,
                self._read_int16(packet, 29) * self.ACC_SCALE,
                self._read_int16(packet, 31) * self.ACC_SCALE,
            ],
            dtype=float,
        )
        gyro = np.array(
            [
                self._read_int16(packet, 33) * self.GYRO_SCALE,
                self._read_int16(packet, 35) * self.GYRO_SCALE,
                self._read_int16(packet, 37) * self.GYRO_SCALE,
            ],
            dtype=float,
        )
        battery = 100.0 * (packet[2] & 0x0F) / 15.0
        counter = float(
            packet[39]
            + (packet[40] << 8)
            + (packet[41] << 16)
            + (packet[42] << 24)
        )
        return UnicornPacket(eeg=eeg, accelerometer=accel, gyroscope=gyro, battery=battery, counter=counter)

    @staticmethod
    def _read_int16(packet: Sequence[int], start: int) -> int:
        """Interpret two bytes from ``packet`` (little-endian signed)."""
        value = packet[start] + (packet[start + 1] << 8)
        if value & 0x8000:
            value -= 0x10000
        return value
