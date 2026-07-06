"""P300 acquisition module with live ERP averaging."""

from __future__ import annotations

import numpy as np

from cortipy.core.context import ModuleContext
from cortipy.evaluation.p300 import P300Evaluator
from cortipy.shared import filter_vep, plot_live_erp, seg_sig_fast_p300, trigger_adc
from cortipy.shared.notifications import info_end_live, info_start_live

from .base import ModuleBase


class P300Module(ModuleBase):
    def __init__(self, evaluator: P300Evaluator | None = None) -> None:
        super().__init__("P300", ["P300"], evaluator or P300Evaluator())
        self.first_second_duration = 1.0
        self.time_step = 20.0
        self.max_time = 0.8
        self.high_pass_cutoff = 0.5

    def collect_measurements(self, context: ModuleContext) -> None:
        params = context.params
        param_block = params.setdefault("Parameters", {})
        param_block.setdefault("edge", "f")
        fs = float(param_block.get("fs", 0))
        if fs <= 0:
            raise ValueError("P300 measurements require Params.Parameters.fs to be set.")

        device = self.require_device(context)
        info_start_live()
        aux_ch = self._resolve_aux_channels(params)

        data = self._ensure_array(device.prime(self.first_second_duration, aux_ch))
        elapsed = 0.0
        recording_time = float(param_block.get("RecordingTime", 60))

        while True:
            remaining = recording_time - (elapsed + self.first_second_duration)
            if remaining <= 0:
                break
            duration = min(self.time_step, remaining)
            chunk = self._ensure_array(device.acquire(duration, aux_ch))
            data = np.vstack([data, chunk])
            elapsed += duration

            self._update_live_plot(params, data, fs)

        info_end_live()
        context.data_buffer = data
        params["data"] = data

    # ------------------------------------------------------------------
    def _ensure_array(self, data) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        return arr

    def _resolve_aux_channels(self, params: dict) -> int:
        if params.get("Device") == "ActiCHamp":
            return int(params.get("Parameters", {}).get("NumberAUXChannels", 0))
        return 0

    def _apply_reference(self, params: dict, data: np.ndarray) -> np.ndarray:
        param_block = params.get("Parameters", {})
        device = str(params.get("Device", "")).lower()
        referenced = np.array(data, copy=True)
        if device == "actichamp":
            ref_idx = int(param_block.get("ReferenceChannel", 1)) - 1
            trig_idx = int(param_block.get("TriggerChannel", data.shape[1])) - 1
            mask = np.ones(referenced.shape[1], dtype=bool)
            if 0 <= trig_idx < mask.size:
                mask[trig_idx] = False
            referenced[:, mask] = referenced[:, mask] - referenced[:, [ref_idx]]
        elif device == "unicorn":
            referenced = referenced[:, : min(8, referenced.shape[1])]
        return referenced

    def _update_live_plot(self, params: dict, data: np.ndarray, fs: float) -> None:
        param_block = params.get("Parameters", {})
        live_ch = max(1, int(param_block.get("LivePlotCH", 1))) - 1
        trig_idx = int(param_block.get("TriggerChannel", data.shape[1])) - 1
        if trig_idx < 0 or trig_idx >= data.shape[1]:
            return

        referenced = self._apply_reference(params, data)
        if live_ch < 0 or live_ch >= referenced.shape[1]:
            return
        referenced[:, live_ch] = filter_vep(referenced[:, [live_ch]], self.high_pass_cutoff, fs).ravel()

        triggered = trigger_adc(referenced, fs, trig_idx, self.max_time, edge=param_block.get("edge", "f"))
        segments = seg_sig_fast_p300(triggered, fs, self.max_time, trig_idx)
        if segments.size == 0:
            return

        try:
            channel_data = segments[:, :, live_ch]
        except IndexError:
            return
        avg_signal = channel_data.mean(axis=0)
        avg_signal = avg_signal - np.mean(avg_signal)
        plot_live_erp(avg_signal, params, self.max_time)
