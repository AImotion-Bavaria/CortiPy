"""BERA evaluation pipeline."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import mne

from cortipy.evaluation.base import EvaluatorBase
from cortipy.shared import (
    avg_seg_avg,
    block_weighted,
    filter_bera,
    get_fsp_fmp,
    plot_bera_results,
    prepro,
    residual_noise,
    residual_noise_eclipse,
    seg_sig_fast,
    time_vector,
    trigger_adc,
    wave_amplitude,
)


class BeraEvaluator(EvaluatorBase):
    """Port of the MATLAB EvalBERAmain routine."""

    def __init__(self, show_plots: bool | None = None) -> None:
        self.show_plots = show_plots
        self.max_time = 15e-3
        self.sp_time = 7e-3

    def evaluate(self, context) -> None:  # type: ignore[override]
        params = context.params
        if str(params.get("Method", "")).lower() != "bera":
            return

        data = params.get("data")
        if data is None:
            raise ValueError("BeraEvaluator requires `params['data']`.")

        param_block = params.setdefault("Parameters", {})
        param_block.setdefault("edge", "r")
        params.setdefault("AnaWindow", [0.001, 0.011])
        fs = float(param_block.get("fs", 0))
        if fs <= 0:
            raise ValueError("BERA evaluation requires Params.Parameters.fs.")

        device = (params.get("Device") or "").lower()
        data_array = np.asarray(data, dtype=float)
        if device in {"actichamp", "simulated"}:
            data_ref = self._apply_reference_actichamp(param_block, data_array)
            device_voltage = 1e-6
            timestamp_idx = data_array.shape[1] - 1
            exclude = {timestamp_idx, int(param_block.get("ReferenceChannel", 1)) - 1, int(param_block.get("TriggerChannel", data_array.shape[1])) - 1}
        elif device in {"biopac", "biopack"}:
            data_ref = data_array
            device_voltage = 1e-3
            exclude = {
                int(param_block.get("ChannelIpsi", 1)) - 1,
                int(param_block.get("TriggerChannel", data_array.shape[1])) - 1,
            }
        else:
            raise RuntimeError("BERA evaluation currently supports ActiCHamp or BIOPAC data.")

        filtered = filter_bera(data_ref, fs, exclude)
        trig_idx = int(param_block.get("TriggerChannel", filtered.shape[1])) - 1
        triggered = trigger_adc(filtered, fs, trig_idx, self.max_time, edge=param_block.get("edge", "r"))
        segments = seg_sig_fast(triggered, fs, self.max_time, trig_idx)
        if segments.size == 0:
            params.setdefault("Evaluation", {})
            context.params = params
            return

        num_cycles = segments.shape[0]
        ch_indices = self._channels_to_analyze(device, param_block, filtered.shape[1])
        selected_segments = segments[:, :, ch_indices]

        avg_signals = avg_seg_avg(selected_segments, 4)
        prepro_segments = prepro(selected_segments, device_voltage)
        prepro_average = prepro(avg_signals, device_voltage)

        evaluation = {
            "channels": [idx + 1 for idx in ch_indices],
            "average_signals": prepro_average,
            "segments_signalSameUnit": prepro_segments,
            "Fsp": [],
            "p_sp": [],
            "Fmp": [],
            "p_mp": [],
            "RN_elberlingDon": [],
            "RN_eclipse": [],
            "amp_V": [],
            "mark_max": [],
            "mark_min": [],
        }

        ana_window = tuple(params.get("AnaWindow", [0.001, 0.011]))
        time_ms = time_vector(prepro_average[0, :, 0], fs, unit="ms")

        for ch_idx in range(len(ch_indices)):
            channel_segments = prepro_segments[:, :, ch_idx]
            channel_avg = prepro_average[0, :, ch_idx]

            weighted = block_weighted(channel_segments, fs, self.sp_time)
            prepro_average[0, :, ch_idx] = weighted

            fsp, p_sp, fmp, p_mp = get_fsp_fmp(channel_segments, weighted, fs, self.sp_time, ana_window)
            rn_elberling = residual_noise(channel_segments, fs, ana_window)
            rn_eclipse, _ = residual_noise_eclipse(channel_segments, fs, ana_window)

            amp, idx_max, idx_min = wave_amplitude(5.6 - (4 * 0.21), 5.6 + (4 * 0.21), channel_avg, time_ms)

            evaluation["Fsp"].append(fsp)
            evaluation["p_sp"].append(p_sp)
            evaluation["Fmp"].append(fmp)
            evaluation["p_mp"].append(p_mp)
            evaluation["RN_elberlingDon"].append(rn_elberling)
            evaluation["RN_eclipse"].append(rn_eclipse)
            evaluation["amp_V"].append(amp)
            evaluation["mark_max"].append(idx_max)
            evaluation["mark_min"].append(idx_min)

        for key in ("Fsp", "p_sp", "Fmp", "p_mp", "RN_elberlingDon", "RN_eclipse", "amp_V"):
            evaluation[key] = np.asarray(evaluation[key], dtype=float)
        for key in ("mark_max", "mark_min"):
            evaluation[key] = np.asarray(
                [val if val is not None else np.nan for val in evaluation[key]], dtype=float
            )
        evaluation["time_ms"] = time_ms
        evaluation["num_cycles"] = num_cycles

        params["Evaluation"] = evaluation

        show_plots = self.show_plots if self.show_plots is not None else not params.get("ReportAnalyzer")
        if show_plots:
            plot_bera_results(prepro_average, params, num_cycles, time_ms)
            _plot_abr_trace(prepro_average, time_ms, params)
            _plot_abr_topomap(context, t_ms=7.0)

        context.params = params

    # ------------------------------------------------------------------
    def _apply_reference_actichamp(self, param_block: dict, data: np.ndarray) -> np.ndarray:
        ref_idx = int(param_block.get("ReferenceChannel", 1)) - 1
        trig_idx = int(param_block.get("TriggerChannel", data.shape[1])) - 1
        referenced = np.array(data, copy=True)
        mask = np.ones(referenced.shape[1], dtype=bool)
        if 0 <= trig_idx < mask.size:
            mask[trig_idx] = False
        if 0 <= ref_idx < mask.size:
            mask[ref_idx] = False
        referenced[:, mask] = referenced[:, mask] - referenced[:, [ref_idx]]
        return referenced

    def _channels_to_analyze(self, device: Optional[str], param_block: dict, total_channels: int) -> List[int]:
        device = (device or "").lower()
        if device in {"actichamp", "simulated"}:
            trigger_idx = int(param_block.get("TriggerChannel", total_channels)) - 1
            reference_idx = int(param_block.get("ReferenceChannel", 1)) - 1
            timestamp_idx = total_channels - 1
            return [
                idx
                for idx in range(total_channels)
                if idx not in {trigger_idx, reference_idx, timestamp_idx}
            ]
        if device in {"biopac", "biopack"}:
            idx_list = []
            ipsi = int(param_block.get("ChannelIpsi", 1)) - 1
            contra = int(param_block.get("ChannelContra", ipsi + 1)) - 1
            idx_list.append(max(0, min(ipsi, total_channels - 1)))
            if contra != ipsi:
                idx_list.append(max(0, min(contra, total_channels - 1)))
            return idx_list
        return list(range(total_channels))


def _plot_abr_trace(prepro_average: np.ndarray, time_ms: np.ndarray, params: dict) -> None:
    """Plot ABR single-channel trace similar to Experiment 1; skips on error."""
    try:
        # prepro_average shape: (1, samples, channels)
        data = prepro_average.squeeze()  # samples x channels
        if data.ndim != 2:
            return
        ch_idx = int(params.get("Parameters", {}).get("ChannelIpsi", 1)) - 1
        ch_idx = max(0, min(ch_idx, data.shape[1] - 1))
        trace = data[:, ch_idx]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_ms, trace, color="blue", linewidth=1.25, label=f"Ch {ch_idx+1}")
        ax.set_xlim(0.0, 15.0)
        ax.set_ylim(-0.3, 0.4)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title("ABR trace")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    except Exception:
        return


def _plot_abr_topomap(context, t_ms: float = 7.0) -> None:
    """Optional ABR topomap at a given latency; quietly skips if raw/info missing."""
    try:
        raw = getattr(context, "raw", None)
        if raw is None or not isinstance(raw, mne.io.BaseRaw):
            return
        sample = int(round((t_ms / 1000.0) * raw.info["sfreq"]))
        if sample < 0 or sample >= raw.n_times:
            return
        data = raw.get_data(picks="eeg")[:, sample]
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        ax.set_axis_off()
        im, _ = mne.viz.plot_topomap(
            data,
            raw.info,
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
        fig.suptitle(f"ABR Topomap @ {t_ms:.1f} ms", fontsize=14)
        fig.tight_layout()
    except Exception:
        return
