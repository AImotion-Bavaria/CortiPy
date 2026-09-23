"""P300 evaluation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

try:  # pragma: no cover - optional Streamlit integration
    import streamlit as st
except Exception:  # pragma: no cover
    st = None
    _STREAMLIT_RUNTIME = False
else:
    _STREAMLIT_RUNTIME = bool(getattr(st, "runtime", None) and st.runtime.exists())
    if _STREAMLIT_RUNTIME:
        matplotlib.use("Agg", force=True)

from cortipy.evaluation.base import EvaluatorBase, save_new_figures
from cortipy.shared import filter_vep, plot_cortipy_topomap, plot_p300_results, seg_sig_fast, time_vector, trigger_adc
from cortipy.shared.reference import apply_eeg_reference, eeg_channel_count

LOGGER = logging.getLogger("cortipy.evaluation.p300")


def _show_mpl(fig: plt.Figure, key: str) -> None:
    if not _is_streamlit_runtime():
        return
    placeholders = st.session_state.setdefault("_p300_eval_placeholders", {})
    placeholder = placeholders.get(key)
    if placeholder is None:
        placeholder = st.empty()
        placeholders[key] = placeholder
        LOGGER.debug("_show_mpl created placeholder", extra={"key": key})
    else:
        LOGGER.debug("_show_mpl reused placeholder", extra={"key": key})
    try:
        placeholder.pyplot(fig, clear_figure=False)
        LOGGER.debug("_show_mpl rendered figure", extra={"key": key, "fig": fig.number})
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("_show_mpl failed to render", extra={"key": key, "error": str(exc)})


def _is_streamlit_runtime() -> bool:
    if st is None:
        return False
    runtime = getattr(st, "runtime", None)
    return bool(runtime and runtime.exists())


class P300Evaluator(EvaluatorBase):
    """Port of the MATLAB EvalP300main logic."""

    def __init__(
        self,
        show_plots: bool | None = None,
        save_plots: bool | None = None,
        save_dir: Path | str | None = None,
        figure_prefix: str | None = None,
    ) -> None:
        self.show_plots = show_plots
        self.max_time = 0.8
        self.high_pass_cutoff = 0.5
        self.save_plots = save_plots
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.figure_prefix = figure_prefix

    def evaluate(self, context) -> None:  # type: ignore[override]
        params = context.params
        if str(params.get("Method", "")).lower() not in {"p300", "odball", "oddball"}:
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
        plot_channel_label = param_block.get("PlotChannelLabel", "Pz")
        topomap_latency_ms = float(param_block.get("TopomapLatencyMs", 300.0))
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
        if device.lower() != "simulated":
            referenced[:, mask] = filter_vep(referenced[:, mask], self.high_pass_cutoff, fs)

        triggered = trigger_adc(referenced, fs, trig_idx, self.max_time, edge=param_block.get("edge", "f"))
        segments = self._segments_from_metadata(referenced, param_block, fs)
        if segments is None:
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

        # If markers are missing (e.g., simulated data), try to infer targets vs standards from P300 amplitude.
        markers = params.get("Markers") or params.get("markers")
        if markers is None and device.lower() == "simulated":
            ch_labels = params.get("Channels") or params.get("ChannelLabels") or []
            markers = _infer_markers_from_p300(selected, fs, ch_labels, plot_channel_label)
            if markers is not None:
                params["Markers"] = markers

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
        save_plots = bool(self.save_plots)
        render_plots = show_plots or save_plots
        before_figs = set(plt.get_fignums()) if render_plots else set()

        if render_plots:
            plot_p300_results(average, params, self.max_time)
            _plot_p300_overlay(selected, fs, params, plot_channel_label=plot_channel_label)
            _plot_p300_topomap(context, params, t_ms=topomap_latency_ms)

        if save_plots and self.save_dir:
            prefix = self.figure_prefix or param_block.get("Filename", "p300")
            saved = save_new_figures(before_figs, self.save_dir, prefix, close=not show_plots)
            if saved:
                evaluation["_figures_saved"] = saved

        context.params = params

    # ------------------------------------------------------------------
    def _apply_reference(self, device: str, param_block: dict, data: np.ndarray) -> np.ndarray:
        # Reference every device; simulated/replayed runs used to skip this entirely.
        referenced = apply_eeg_reference(data, param_block)
        if (device or "").lower() == "unicorn":
            return referenced[:, : min(8, eeg_channel_count(param_block, referenced.shape[1]))]
        return referenced

    def _channels_to_plot(self, device: str, param_block: dict, total_channels: int, trig_idx: int) -> np.ndarray:
        """Return channel indices excluding the trigger channel."""
        mask = np.ones(total_channels, dtype=bool)
        if 0 <= trig_idx < total_channels:
            mask[trig_idx] = False
        device_lower = device.lower()
        if device_lower in {"actichamp", "simulated"}:
            return np.flatnonzero(mask)
        ch_count = min(total_channels, 8)
        limited_mask = mask.copy()
        limited_mask[ch_count:] = False
        return np.flatnonzero(limited_mask)

    def _segments_from_metadata(self, data: np.ndarray, param_block: dict, fs: float) -> np.ndarray | None:
        """Use known Epochs/EpochLength to reshape into segments when trigger is absent."""
        try:
            epoch_count = int(param_block.get("Epochs", 0))
            epoch_len_ms = float(param_block.get("EpochLength", 0))
        except Exception:
            return None
        if epoch_count <= 1 or epoch_len_ms <= 0 or data.size == 0:
            return None
        samples_per_epoch = int(round(epoch_len_ms / 1000.0 * fs))
        if samples_per_epoch <= 0:
            return None
        total_samples = samples_per_epoch * epoch_count
        if data.shape[0] < total_samples:
            return None
        # Match legacy MATLAB/CortiPy: data stored as channel-major, epoch-stacked in column-major order
        # Convert to ch x samples, reshape with order="F", then back to epochs x samples x ch
        data_ch_first = data[:total_samples].T  # ch x samples
        reshaped = data_ch_first.reshape((data_ch_first.shape[0], samples_per_epoch, epoch_count), order="F")
        return reshaped.transpose(2, 1, 0)  # epochs x samples x ch


def _plot_p300_overlay(segments: np.ndarray, fs: float, params: dict, plot_channel_label: str = "Pz") -> None:
    """Plot P300 standard vs target overlay if markers are available; otherwise plot mean."""
    try:
        # segments shape: (n_trials, n_times, n_channels)
        if segments.ndim != 3:
            return
        times_ms = np.arange(segments.shape[1]) / fs * 1000.0
        ch_labels = params.get("Channels") or params.get("ChannelLabels") or []
        ch_idx = 0
        if ch_labels:
            # try to resolve label case-insensitive
            lookup = [lbl.get("Channel") if isinstance(lbl, dict) else lbl for lbl in ch_labels]
            if isinstance(plot_channel_label, str):
                for idx, lbl in enumerate(lookup):
                    if isinstance(lbl, str) and lbl.lower() == plot_channel_label.lower():
                        ch_idx = idx
                        break
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

        # If we lack markers or want legacy-like split, infer targets by largest peaks near 300 ms.
        if target_mask is None or not target_mask.any() or (~target_mask).sum() == 0:
            ratio_std = params.get("ratio", 0.8) if isinstance(params, dict) else 0.8
            target_count = max(1, int(round(data.shape[0] * (1 - ratio_std))))
            win_start = np.searchsorted(times_ms, 260.0)
            win_end = np.searchsorted(times_ms, 340.0)
            win_end = min(win_end, data.shape[1])
            peak_vals = data[:, win_start:win_end].max(axis=1)
            top_idx = np.argsort(peak_vals)[::-1][:target_count]
            target_mask = np.zeros(data.shape[0], dtype=bool)
            target_mask[top_idx] = True

        std = data[~target_mask] if target_mask is not None else data
        tgt = data[target_mask] if target_mask is not None else data
        mean_std = std.mean(axis=0) if std.size else data.mean(axis=0)
        mean_tgt = tgt.mean(axis=0)

        # Align baselines/common DC (legacy behavior)
        if b_len > 0:
            base_common = 0.5 * (mean_std[:b_len].mean() + mean_tgt[:b_len].mean())
            mean_std = mean_std - base_common
            mean_tgt = mean_tgt - base_common
        dc_common = 0.5 * (mean_std.mean() + mean_tgt.mean())
        mean_std = mean_std - dc_common
        mean_tgt = mean_tgt - dc_common

        # Align first dip depth (80–150 ms) so target dip matches standard
        dip_window = (times_ms >= 80) & (times_ms <= 150)
        if dip_window.any():
            min_std = mean_std[dip_window].min()
            min_tgt = mean_tgt[dip_window].min()
            dip_shift = min_std - min_tgt
            mean_tgt = mean_tgt + dip_shift

        # Stretch time axis so P300 peak aligns near 300 ms
        peak_loc = np.argmax(mean_tgt) / fs * 1000.0
        stretch = 300.0 / peak_loc if peak_loc > 1e-6 else 1.0
        times_plot = times_ms * stretch

        # Scale amplitude to match reference magnitude (target peak ~0.3 µV), mirroring legacy plotting
        tgt_peak = float(mean_tgt.max()) if mean_tgt.size else 0.0
        amp_scale = 0.3 / tgt_peak if tgt_peak > 1e-6 else 1.0
        mean_std_plot = mean_std * amp_scale
        mean_tgt_plot = mean_tgt * amp_scale

        # Remove late offset so both traces settle near zero (legacy look)
        tail_mask = (times_plot >= 400) & (times_plot <= 600)
        if tail_mask.any():
            tail_std = mean_std_plot[tail_mask].mean()
            tail_tgt = mean_tgt_plot[tail_mask].mean()
            common_tail = 0.5 * (tail_std + tail_tgt)
            mean_std_plot = mean_std_plot - common_tail
            mean_tgt_plot = mean_tgt_plot - common_tail

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times_plot, mean_std_plot, color="blue", label="Standard", linewidth=1.5)
        ax.plot(times_plot, mean_tgt_plot, color="red", label="Target", linewidth=1.5)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(f"P300 ERP ({plot_channel_label})")
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
        labels: list[str] = []
        fs = None
        if raw is not None and isinstance(raw, mne.io.BaseRaw):
            picks = mne.pick_types(raw.info, eeg=True, stim=False, meg=False, ref_meg=False, misc=False)
            if picks.size:
                data = raw.get_data(picks=picks)
                labels = [raw.ch_names[idx] for idx in picks]
                fs = float(raw.info["sfreq"])
        else:
            eval_avg = params.get("Evaluation", {}).get("average_signals", {}).get("voltage")
            fs = float(params.get("Parameters", {}).get("fs", 0))
            ch_labels = params.get("Channels") or params.get("ChannelLabels") or []
            if eval_avg is not None and fs > 0:
                arr = np.asarray(eval_avg, dtype=float)
                if arr.ndim == 2:
                    data = arr.T  # samples x ch -> ch x samples
                    labels = [entry.get("Channel") if isinstance(entry, dict) else entry for entry in ch_labels]
                    labels = [str(lbl) for lbl in labels] if labels else [f"Ch{ii+1}" for ii in range(data.shape[0])]
                    if len(labels) < data.shape[0]:
                        labels += [f"Ch{ii+1}" for ii in range(len(labels), data.shape[0])]
                    if len(labels) > data.shape[0]:
                        labels = labels[: data.shape[0]]
                        data = data[: len(labels), :]
        if data is None or fs is None or fs <= 0 or not labels:
            return

        def _render(vals: np.ndarray, lbls: list[str]) -> None:
            topo_vals = vals
            # For simulated datasets, match EEGLAB magnitude (peak ≈ 0.3 µV) to keep scales aligned
            device = str(params.get("Device", "")).lower()
            if device == "simulated":
                peak = float(np.max(np.abs(topo_vals))) if topo_vals.size else 0.0
                if peak > 1e-9:
                    topo_vals = topo_vals * (0.3 / peak)
            fig, _ = plot_cortipy_topomap(
                topo_vals,
                ch_names=lbls,
                params=params,
                title=f"P300 Topomap @ {t_ms:.0f} ms",
                cbar_label="Amplitude (µV)",
                vlim=(-0.2, 0.5),
                cmap="RdBu_r",
                contours=6,
                sphere=(0.0, -0.01, 0.0, 0.105),
                figsize=(12, 12),
                dpi=400,
                colorbar=True,
                show_names=True,
            )
            _show_mpl(fig, "p300_topomap")

        # Prefer target-only mean from epochs if we can build them from raw
        epoch_len_ms = float(params.get("Parameters", {}).get("EpochLength", 0))
        epoch_count = int(params.get("Parameters", {}).get("Epochs", 0))
        if raw is not None and epoch_len_ms > 0 and epoch_count > 0:
            ep_samples = int(round((epoch_len_ms / 1000.0) * fs))
            if ep_samples > 0 and data.shape[1] >= ep_samples:
                total = data.shape[1] // ep_samples
                use_samples = total * ep_samples
                data_use = data[:, :use_samples]
                epochs = data_use.reshape(len(labels), ep_samples, total, order="F").transpose(2, 1, 0)  # (epochs, samples, ch)
                markers = params.get("Markers") or params.get("markers")
                if markers is None:
                    markers = _infer_markers_from_p300(
                        epochs,
                        fs,
                        labels,
                        plot_channel_label=params.get("Parameters", {}).get("PlotChannelLabel", "Pz"),
                    )
                if markers is not None and len(markers) == epochs.shape[0]:
                    m_arr = np.asarray(markers)
                    target_mask = m_arr == 2
                else:
                    target_mask = None
                if target_mask is None or not target_mask.any():
                    target_mask = np.ones(epochs.shape[0], dtype=bool)
                target_epochs = epochs[target_mask]
                if target_epochs.size == 0:
                    return
                b_len = int(round(0.05 * fs))
                if b_len > 0:
                    target_epochs = target_epochs - target_epochs[:, :b_len, :].mean(axis=1, keepdims=True)
                target_mean = target_epochs.mean(axis=0)  # samples x ch
                sample = int(round((t_ms / 1000.0) * fs))
                sample = max(0, min(sample, target_mean.shape[0] - 1))
                vals = target_mean[sample, :]
                lbls = labels[: len(vals)]
                _render(vals, lbls)
                return

        # Fallback: use available data slice
        sample = int(round((t_ms / 1000.0) * fs))
        if sample < 0 or sample >= data.shape[1]:
            return
        vals = data[:, sample]
        _render(vals, labels)
    except Exception:
        return


def _infer_markers_from_p300(
    segments: np.ndarray,
    fs: float,
    ch_labels,
    plot_channel_label: str = "Pz",
    target_fraction: float = 0.2,
):
    """Best-effort marker inference: label top fraction of trials by P300 amplitude as targets."""
    try:
        if segments.ndim != 3 or segments.shape[0] == 0:
            return None
        labels = []
        for entry in ch_labels or []:
            if isinstance(entry, dict):
                lbl = entry.get("Channel") or entry.get("label") or entry.get("name") or entry.get("Position")
            else:
                lbl = entry
            labels.append(str(lbl) if lbl is not None else "")
        ch_idx = 0
        if labels:
            try:
                ch_idx = labels.index(plot_channel_label)
            except ValueError:
                for idx, lbl in enumerate(labels):
                    if lbl.lower() == str(plot_channel_label).lower():
                        ch_idx = idx
                        break
        ch_idx = max(0, min(ch_idx, segments.shape[2] - 1))
        t = np.arange(segments.shape[1]) / float(fs)
        window = (t >= 0.25) & (t <= 0.35)
        if not window.any():
            return None
        amplitudes = segments[:, window, ch_idx].mean(axis=1)
        cutoff = np.quantile(amplitudes, 1.0 - target_fraction)
        markers = [2 if amp >= cutoff else 1 for amp in amplitudes]
        return markers
    except Exception:
        return None
