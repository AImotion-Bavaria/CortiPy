"""g.tec UNICORN EEG hardware adapter (serial protocol)."""

from __future__ import annotations

from dataclasses import dataclass
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

    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self._serial is not None:
            return
        ser = serial.Serial(self.port, self.BAUD_RATE, timeout=self.timeout)
        ser.reset_input_buffer()
        ser.write(self.START_ACQ)
        ack = ser.read(len(self.START_RESPONSE))
        if ack != self.START_RESPONSE:
            ser.close()
            raise RuntimeError(
                f"UNICORN device '{self.device_name}' did not acknowledge the start command (got {ack!r})."
            )
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

    # ------------------------------------------------------------------
    def _read_exact(self, size: int) -> bytes:
        """Read exactly ``size`` bytes from the serial port or raise."""
        assert self._serial is not None
        buffer = bytearray()
        while len(buffer) < size:
            chunk = self._serial.read(size - len(buffer))
            if not chunk:
                raise RuntimeError("UNICORN stream ended unexpectedly.")
            buffer.extend(chunk)
        return bytes(buffer)

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
