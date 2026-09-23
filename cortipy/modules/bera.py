"""BERA acquisition module."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from cortipy.core.context import ModuleContext
from cortipy.evaluation.bera import BeraEvaluator
from cortipy.shared import (
    filter_bera,
    get_fsp_fmp,
    plot_live_avg_bera,
    prepro,
    residual_noise_eclipse,
    seg_sig_fast,
    trigger_adc,
)
from cortipy.shared.notifications import info_end_live, info_start_live
from cortipy.shared.reference import apply_eeg_reference

from .base import ModuleBase


class BeraModule(ModuleBase):
    """Python translation of the MATLAB `BeraModule`."""

    def __init__(self, evaluator: Optional[BeraEvaluator] = None) -> None:
        super().__init__("BERA", ["BERA"], evaluator or BeraEvaluator())
        self.first_second_duration = 1.0
        self.time_step = 5.0
        self.max_time = 15e-3
        self.sp_time = 7e-3

    def collect_measurements(self, context: ModuleContext) -> None:
        params = context.params
        if params.get("Device") != "ActiCHamp":
            raise RuntimeError("BERA live acquisition currently supports ActiCHamp only.")

        param_block = params.setdefault("Parameters", {})
        param_block.setdefault("edge", "r")
        params.setdefault("AnaWindow", [0.001, 0.011])
        fs = float(param_block.get("fs", 0))
        if fs <= 0:
            raise ValueError("BERA measurements require Params.Parameters.fs to be set.")

        device = self.require_device(context)
        info_start_live()
        aux_ch = int(param_block.get("NumberAUXChannels", 0))

        data = self._ensure_array(device.prime(self.first_second_duration, aux_ch))
        elapsed = 0.0
        rn_hist: List[float] = []
        fmp_hist: List[float] = []
        elapsed_hist: List[float] = []

        while True:
            remaining = param_block.get("RecordingTime", 60) - (elapsed + self.first_second_duration)
            if remaining <= 0:
                break
            duration = min(self.time_step, remaining)
            new_chunk = self._ensure_array(device.acquire(duration, aux_ch))
            data = np.vstack([data, new_chunk])
            elapsed += duration

            metrics = self._compute_live_metrics(params, data, fs)
            if metrics is not None:
                avg_signal, rn, fmp = metrics
                rn_hist.append(rn)
                fmp_hist.append(fmp)
                elapsed_hist.append(elapsed)
                plot_live_avg_bera(
                    avg_signal,
                    params,
                    self.max_time,
                    int(param_block.get("LivePlotCH", 1)),
                    elapsed_hist,
                    rn_hist,
                    fmp_hist,
                )

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
        param_block = params.get("Parameters", {})
        return apply_eeg_reference(data, param_block, zero_reference=False)

    def _compute_live_metrics(self, params: dict, data: np.ndarray, fs: float):
        param_block = params.get("Parameters", {})
        trig_idx = int(param_block.get("TriggerChannel", data.shape[1])) - 1
        if trig_idx < 0 or trig_idx >= data.shape[1]:
            return None
        referenced = self._apply_reference(params, data)
        exclude = {trig_idx, int(param_block.get("ReferenceChannel", 1)) - 1, data.shape[1] - 1}
        filtered = filter_bera(referenced, fs, exclude)
        triggered = trigger_adc(filtered, fs, trig_idx, self.max_time, edge=param_block.get("edge", "r"))
        segments = seg_sig_fast(triggered, fs, self.max_time, trig_idx)
        if segments.size == 0:
            return None
        live_ch = max(0, int(param_block.get("LivePlotCH", 1)) - 1)
        if live_ch >= segments.shape[2]:
            live_ch = segments.shape[2] - 1
        channel_segments = segments[:, :, live_ch]
        avg_signal = channel_segments.mean(axis=0)
        prepro_segments = prepro(channel_segments, 1e-6)
        prepro_avg = prepro(avg_signal, 1e-6)
        rn, _ = residual_noise_eclipse(prepro_segments, fs, tuple(params.get("AnaWindow", [0.001, 0.011])))
        _, _, fmp, _ = get_fsp_fmp(
            prepro_segments,
            prepro_avg,
            fs,
            self.sp_time,
            tuple(params.get("AnaWindow", [0.001, 0.011])),
        )
        return prepro_avg, rn, fmp
