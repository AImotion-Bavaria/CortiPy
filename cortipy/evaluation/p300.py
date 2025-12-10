"""P300 evaluation pipeline."""

from __future__ import annotations

import numpy as np
import logging

from cortipy.evaluation.base import EvaluatorBase
from cortipy.shared import filter_vep, plot_p300_results, seg_sig_fast, time_vector, trigger_adc

LOGGER = logging.getLogger("cortipy.evaluation.p300")


class P300Evaluator(EvaluatorBase):
    """Port of the MATLAB EvalP300main logic."""

    def __init__(self, show_plots: bool | None = None) -> None:
        self.show_plots = show_plots
        self.max_time = 0.8
        self.high_pass_cutoff = 0.5

    def evaluate(self, context) -> None:  # type: ignore[override]
        params = context.params
        if str(params.get("Method", "")).lower() != "p300":
            return

        data = params.get("data")
        if data is None:
            raise ValueError("P300 evaluation requires `params['data']`.")

        param_block = params.setdefault("Parameters", {})
        param_block.setdefault("edge", "f")
        fs = float(param_block.get("fs", 0))
        if fs <= 0:
            raise ValueError("P300 evaluation requires Params.Parameters.fs.")

        device = params.get("Device", "")
        referenced = self._apply_reference(device, param_block, np.asarray(data, dtype=float))

        trig_idx = int(param_block.get("TriggerChannel", referenced.shape[1])) - 1
        if trig_idx < 0 or trig_idx >= referenced.shape[1]:
            raise IndexError("TriggerChannel is out of bounds.")
        trig_values = referenced[:, trig_idx]
        LOGGER.info(
            "P300: fs=%.3f, data_shape=%s, trig_idx=%d, trig_min=%.4f, trig_max=%.4f",
            fs,
            referenced.shape,
            trig_idx,
            float(np.min(trig_values)),
            float(np.max(trig_values)),
        )

        mask = np.ones(referenced.shape[1], dtype=bool)
        mask[trig_idx] = False
        referenced[:, mask] = filter_vep(referenced[:, mask], self.high_pass_cutoff, fs)

        triggered = trigger_adc(referenced, fs, trig_idx, self.max_time, edge=param_block.get("edge", "f"))
        segments = seg_sig_fast(triggered, fs, self.max_time, trig_idx)
        if segments.size == 0:
            LOGGER.info("P300: no segments found (triggered shape=%s)", triggered.shape)
            params.setdefault("Evaluation", {})
            context.params = params
            return
        LOGGER.info("P300: segments extracted %s", segments.shape)

        ch_plot = self._channels_to_plot(device, param_block, referenced.shape[1], trig_idx)
        selected = segments[:, :, ch_plot]
        average = selected.mean(axis=0)
        average = average - np.mean(average, axis=0, keepdims=True)

        time_ms = time_vector(average, fs, unit="ms")
        evaluation = params.setdefault("Evaluation", {})
        evaluation["average_signals"] = {
            "voltage": average,
            "voltageUnit": "Amplitude (uV)",
            "time": time_ms,
            "timeUnit": "Time (ms)",
        }
        evaluation["nTargets"] = segments.shape[0]

        show_plots = self.show_plots if self.show_plots is not None else not params.get("ReportAnalyzer")
        if show_plots:
            plot_p300_results(average, params, self.max_time)

        context.params = params

    # ------------------------------------------------------------------
    def _apply_reference(self, device: str, param_block: dict, data: np.ndarray) -> np.ndarray:
        device = (device or "").lower()
        if device == "actichamp":
            ref_idx = int(param_block.get("ReferenceChannel", 1)) - 1
            trig_idx = int(param_block.get("TriggerChannel", data.shape[1])) - 1
            referenced = np.array(data, copy=True)
            mask = np.ones(referenced.shape[1], dtype=bool)
            mask[ref_idx] = False
            if 0 <= trig_idx < mask.size:
                mask[trig_idx] = False
            referenced[:, mask] = referenced[:, mask] - referenced[:, [ref_idx]]
            return referenced
        if device == "unicorn":
            return data[:, : min(8, data.shape[1])]
        return data

    def _channels_to_plot(self, device: str, param_block: dict, total_channels: int, trig_idx: int) -> np.ndarray:
        """Return channel indices excluding the trigger channel."""
        mask = np.ones(total_channels, dtype=bool)
        if 0 <= trig_idx < total_channels:
            mask[trig_idx] = False
        if device.lower() == "actichamp":
            return np.flatnonzero(mask)
        ch_count = min(total_channels, 8)
        limited_mask = mask.copy()
        limited_mask[ch_count:] = False
        return np.flatnonzero(limited_mask)
