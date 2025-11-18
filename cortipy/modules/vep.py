"""Visual Evoked Potential acquisition module."""

from __future__ import annotations

import numpy as np

from cortipy.core.context import ModuleContext
from cortipy.evaluation.vep import VepEvaluator
from cortipy.shared.filtering import filter_vep
from cortipy.shared.notifications import info_end_live, info_start_live
from cortipy.shared.plotting import plot_live_avg_vep
from cortipy.shared.segmentation import seg_sig_fast
from cortipy.shared.triggers import trigger_adc

from .base import ModuleBase


class VepModule(ModuleBase):
    def __init__(self, evaluator: VepEvaluator | None = None) -> None:
        super().__init__("VEP", ["VEP"], evaluator or VepEvaluator())
        self.first_second_duration = 1.0
        self.time_step = 5.0
        self.max_time = 0.35

    def collect_measurements(self, context: ModuleContext) -> None:
        params = context.params
        param_block = params.setdefault("Parameters", {})
        param_block.setdefault("edge", "b")

        recording_time = float(param_block.get("RecordingTime", 60))
        fs = float(param_block.get("fs", 0))
        if fs <= 0:
            raise ValueError("VEP measurements require Params.Parameters.fs to be set.")

        device = self.require_device(context)
        info_start_live()
        aux_ch = self._resolve_aux_channels(params)

        data = self._ensure_array(device.prime(self.first_second_duration, aux_ch))
        elapsed = 0.0

        while True:
            remaining = recording_time - (elapsed + self.first_second_duration)
            if remaining <= 0:
                break
            duration = min(self.time_step, remaining)
            data_step = self._ensure_array(device.acquire(duration, aux_ch))
            data = np.vstack([data, data_step])

            self._update_live_plot(params, data, fs)
            elapsed += duration

        info_end_live()
        context.data_buffer = data
        params["data"] = data

    # ------------------------------------------------------------------
    def _resolve_aux_channels(self, params: dict) -> int:
        if params.get("Device") == "ActiCHamp":
            return int(params.get("Parameters", {}).get("NumberAUXChannels", 0))
        return 0

    def _ensure_array(self, data) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        return arr

    def _apply_reference(self, params: dict, data: np.ndarray) -> np.ndarray:
        param_block = params.get("Parameters", {})
        device = str(params.get("Device", "")).lower()
        data_ref = np.array(data, copy=True)

        if device == "actichamp":
            ref_idx = int(param_block.get("ReferenceChannel", 1)) - 1
            trig_idx = int(param_block.get("TriggerChannel", data_ref.shape[1])) - 1
            mask = np.ones(data_ref.shape[1], dtype=bool)
            if 0 <= trig_idx < mask.size:
                mask[trig_idx] = False
            data_ref[:, mask] = data_ref[:, mask] - data_ref[:, [ref_idx]]
        elif device == "unicorn":
            data_ref = data_ref[:, : min(8, data_ref.shape[1])]

        return data_ref

    def _update_live_plot(self, params: dict, data: np.ndarray, fs: float) -> None:
        param_block = params.get("Parameters", {})
        live_channel = max(1, int(param_block.get("LivePlotCH", 1))) - 1
        trigger_channel = int(param_block.get("TriggerChannel", data.shape[1])) - 1
        if trigger_channel < 0 or trigger_channel >= data.shape[1]:
            return
        if live_channel < 0 or live_channel >= data.shape[1]:
            return

        referenced = self._apply_reference(params, data)
        referenced[:, live_channel] = filter_vep(referenced[:, [live_channel]], 0.5, fs)
        triggered = trigger_adc(referenced, fs, trigger_channel, self.max_time, edge=param_block.get("edge", "b"))
        segments = seg_sig_fast(triggered, fs, self.max_time, trigger_channel)
        if segments.size == 0:
            return
        try:
            channel_data = segments[:, :, live_channel]
        except IndexError:
            return
        avg_signal = channel_data.mean(axis=0)
        avg_signal = avg_signal - np.mean(avg_signal)
        plot_live_avg_vep(avg_signal, params, self.max_time)
