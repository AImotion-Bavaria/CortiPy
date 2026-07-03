"""SSVEP measurement module."""

from __future__ import annotations

import numpy as np

from cortipy.core.context import ModuleContext
from cortipy.evaluation.ssvep import SsvepEvaluator
from cortipy.shared.notifications import info_end_live, info_start_live
from cortipy.shared.plotting import plot_fft_live
from cortipy.shared.signal import calc_fft

from .base import ModuleBase


class SsvepModule(ModuleBase):
    def __init__(self, evaluator: SsvepEvaluator | None = None) -> None:
        super().__init__("SSVEP", ["SSVEP"], evaluator or SsvepEvaluator())
        self.first_second_duration = 1.0
        self.time_step = 5.0

    def collect_measurements(self, context: ModuleContext) -> None:
        params = context.params
        param_block = params.setdefault("Parameters", {})
        recording_time = float(param_block.get("RecordingTime", 60))
        device = self.require_device(context)

        info_start_live()
        aux_ch = param_block.get("NumberAUXChannels", 0) if params.get("Device") == "ActiCHamp" else 0
        data = self._ensure_array(device.prime(self.first_second_duration, aux_ch))
        elapsed = 0.0
        fs = float(param_block.get("fs", 250))

        while True:
            remaining = recording_time - (elapsed + self.first_second_duration)
            if remaining <= 0:
                break
            duration = min(self.time_step, remaining)
            data_step = self._ensure_array(device.acquire(duration, aux_ch))
            data = np.vstack([data, data_step])
            data_ref = self._apply_reference(params, data)
            spectrum, freq = calc_fft(data_ref, fs)
            plot_fft_live(freq, spectrum, "Amplitude (uV)", "Periodogram Using FFT", params)
            elapsed += duration

        info_end_live()
        context.data_buffer = data
        params["data"] = data

    def _ensure_array(self, data) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        return arr

    def _apply_reference(self, params: dict, data: np.ndarray) -> np.ndarray:
        params_block = params.get("Parameters", {})
        device = params.get("Device")
        if device == "ActiCHamp":
            ref_channel = int(params_block.get("ReferenceChannel", 1)) - 1
            data_ref = data.copy()
            data_ref = data_ref - data_ref[:, [ref_channel]]
            return data_ref
        if device == "UNICORN":
            return data[:, :8]
        return data
