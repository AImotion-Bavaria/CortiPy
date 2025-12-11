"""P300 evaluation pipeline."""

from __future__ import annotations

import numpy as np
import logging
import matplotlib.pyplot as plt
import mne

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
            _plot_p300_overlay(selected, fs, params)
            _plot_p300_topomap(context, params, t_ms=300.0)

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


def _plot_p300_overlay(segments: np.ndarray, fs: float, params: dict) -> None:
    """Plot P300 standard vs target overlay if markers are available; otherwise plot mean."""
    try:
        # segments shape: (n_trials, n_times, n_channels)
        if segments.ndim != 3:
            return
        times_ms = np.arange(segments.shape[1]) / fs * 1000.0
        ch_labels = params.get("Channels") or params.get("ChannelLabels") or []
        ch_idx = ch_labels.index("Pz") if ch_labels and "Pz" in ch_labels else 0
        data = segments[:, :, ch_idx]

        markers = params.get("Markers") or params.get("markers")
        target_mask = None
        if markers is not None and len(markers) == data.shape[0]:
            markers_arr = np.asarray(markers)
            target_mask = markers_arr == 2

        # Baseline first 50 ms
        b_len = np.searchsorted(times_ms, 50.0)
        if b_len > 0:
            data = data - data[:, :b_len].mean(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        if target_mask is not None and target_mask.any() and (~target_mask).any():
            std = data[~target_mask]
            tgt = data[target_mask]
            mean_std = std.mean(axis=0)
            mean_tgt = tgt.mean(axis=0)
            # Align dip around 80-150 ms
            dip_window = (times_ms >= 80) & (times_ms <= 150)
            if dip_window.any():
                shift = mean_std[dip_window].min() - mean_tgt[dip_window].min()
                mean_tgt = mean_tgt + shift
            ax.plot(times_ms, mean_std, color="blue", label="Standard", linewidth=1.5)
            ax.plot(times_ms, mean_tgt, color="red", label="Target", linewidth=1.5)
        else:
            mean_wave = data.mean(axis=0)
            ax.plot(times_ms, mean_wave, color="blue", label="Mean", linewidth=1.5)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title("P300 ERP (Pz)")
        ax.set_xlim(0, 600)
        ax.set_ylim(-0.15, 0.35)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
    except Exception:
        return


def _plot_p300_topomap(context, params: dict, t_ms: float = 300.0) -> None:
    """Optional P300 topomap; skips if raw/info missing."""
    try:
        raw = getattr(context, "raw", None)
        data = None
        info = None
        if raw is not None and isinstance(raw, mne.io.BaseRaw):
            data = raw.get_data(picks="eeg")
            info = raw.info
        else:
            eval_avg = params.get("Evaluation", {}).get("average_signals", {}).get("voltage")
            fs = float(params.get("Parameters", {}).get("fs", 0))
            ch_labels = params.get("Channels") or params.get("ChannelLabels") or []
            if eval_avg is not None and fs > 0:
                arr = np.asarray(eval_avg, dtype=float)
                if arr.ndim == 2:
                    data = arr.T  # samples x ch -> ch x samples
                if data is not None:
                    info = mne.create_info(
                        ch_names=[str(c) for c in ch_labels] if ch_labels else [f"Ch{ii+1}" for ii in range(data.shape[0])],
                        sfreq=fs,
                        ch_types="eeg",
                    )
        if data is None or info is None:
            return
        sample = int(round((t_ms / 1000.0) * info["sfreq"]))
        if sample < 0 or sample >= data.shape[1]:
            return
        topo_vals = data[:, sample]
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        ax.set_axis_off()
        im, _ = mne.viz.plot_topomap(
            topo_vals,
            info,
            axes=ax,
            show=False,
            contours=6,
            cmap="RdBu_r",
            outlines="head",
            sphere=(0.0, -0.01, 0.0, 0.105),
            extrapolate="head",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Amplitude (µV)", fontsize=12)
        fig.suptitle(f"P300 Topomap @ {t_ms:.0f} ms", fontsize=14)
        fig.tight_layout()
    except Exception:
        return
