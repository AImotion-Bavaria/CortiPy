"""Alpha measurement module."""

from __future__ import annotations

import numpy as np

from cortipy.core.context import ModuleContext
from cortipy.evaluation.alpha import AlphaEvaluator
from cortipy.shared.notifications import beep, info_end_live, info_start_live
from cortipy.shared.plotting import plot_fft_live
from cortipy.shared.signal import calc_fft

from .base import ModuleBase


class AlphaModule(ModuleBase):
    """Python translation of the MATLAB `AlphaModule`."""

    def __init__(self, evaluator: AlphaEvaluator | None = None) -> None:
        super().__init__("Alpha Waves", ["Alpha"], evaluator or AlphaEvaluator())
        self.first_second_duration = 1.0

    def collect_measurements(self, context: ModuleContext) -> None:
        params = context.params
        param_block = params.setdefault("Parameters", {})
        device = self.require_device(context)

        trigger_mode = param_block.get("Trigger", "Fixed")
        if trigger_mode == "Fixed":
            time_step = float(param_block.get("TriggerTime", 5)) / 10.0
        else:
            time_step = 5.0

        recording_time = float(param_block.get("RecordingTime", 60))
        info_start_live()

        aux_ch = param_block.get("NumberAUXChannels", 0) if params.get("Device") == "ActiCHamp" else 0

        data = self._ensure_array(device.prime(self.first_second_duration, aux_ch))
        elapsed = 0.0
        timer_trigger = 1

        while True:
            if (elapsed + self.first_second_duration) >= recording_time:
                break
            duration = min(time_step, recording_time - elapsed - self.first_second_duration)
            if duration <= 0:
                break

            data_step = self._ensure_array(device.acquire(duration, aux_ch))
            data = np.vstack([data, data_step])

            data_ref = self._apply_reference(params, data)
            spectrum, freq = calc_fft(data_ref, float(param_block.get("fs", 250)))
            param_block.setdefault("LowestFrequency", 1)
            param_block.setdefault("HighestFrequency", 40)
            plot_fft_live(freq, spectrum, "Amplitude (uV)", "Periodogram Using FFT", params)

            elapsed += duration
            if (param_block.get("TriggerTime", 0) or 0) * timer_trigger <= elapsed:
                beep()
                timer_trigger += 1

        info_end_live()
        context.data_buffer = data
        params["data"] = data

    # ------------------------------------------------------------------
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
            trigger_channel = int(params_block.get("TriggerChannel", data.shape[1])) - 1
            mask = np.ones(data.shape[1], dtype=bool)
            if 0 <= trigger_channel < data.shape[1]:
                mask[trigger_channel] = False
            data_ref = data.copy()
            data_ref[:, mask] = data_ref[:, mask] - data_ref[:, [ref_channel]]
            return data_ref
        if device == "UNICORN":
            return data[:, :8]
        return data
