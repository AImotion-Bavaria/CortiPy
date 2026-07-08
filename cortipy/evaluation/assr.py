"""ASSR evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import mne

from cortipy.evaluation.base import EvaluatorBase, save_new_figures
from cortipy.shared import assr_calc_snr, assr_compute_psd as _assr_compute_psd, assr_f_test, calc_fft, plot_cortipy_topomap, plot_assr_spectrum
from cortipy.shared.reference import apply_eeg_reference

assr_compute_psd = _assr_compute_psd


class AssrEvaluator(EvaluatorBase):
    """Port of the MATLAB ASSR FFT/SNR evaluation."""

    def __init__(
        self,
        show_plots: bool | None = None,
        save_plots: bool | None = None,
        save_dir: Path | str | None = None,
        figure_prefix: str | None = None,
    ) -> None:
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.figure_prefix = figure_prefix

    def evaluate(self, context) -> None:  # type: ignore[override]
        params = context.params
        if str(params.get("Method", "")).lower() != "assr":
            return

        data = params.get("data")
        if data is None:
            raise ValueError("AssrEvaluator requires `params['data']` to be populated.")

        param_block = params.setdefault("Parameters", {})
        fs = float(param_block.get("fs", 0))
        if fs <= 0:
            raise ValueError("ASSR evaluation requires Params.Parameters.fs.")

        stim_freq = float(param_block.get("ASSRModulationFrequency", 0))
        if stim_freq <= 0 or not np.isfinite(stim_freq):
            stim_freq = 40.0
            param_block["ASSRModulationFrequency"] = stim_freq
        carrier_freq = float(param_block.get("ASSRCarrierFrequency", 0))
        view_low = stim_freq - 10.0
        view_high = stim_freq + 10.0
        f_min_noise = 2.0
        topomap_freq = float(param_block.get("TopomapFrequencyHz", stim_freq))
        param_block["TopomapFrequencyHz"] = topomap_freq

        data_array = np.asarray(data, dtype=float)
        device = str(params.get("Device") or "").lower()
        if device in {"actichamp", "unicorn"}:
            data_array = apply_eeg_reference(data_array, param_block)

        # Allow overriding plotting channel by label (e.g., T8) for comparison plots.
        plot_channel_label = param_block.get("PlotChannelLabel")
        if isinstance(plot_channel_label, str):
            idx_from_label = _channel_idx_from_label(params.get("Channels"), plot_channel_label)
            if idx_from_label is not None and 0 <= idx_from_label < data_array.shape[1]:
                param_block["ChannelIpsi"] = idx_from_label + 1
                param_block["ChannelContra"] = idx_from_label + 1  # keep ipsi/contra aligned for plotting
        # If ChannelIpsi still unset, default to T8 when available.
        if "ChannelIpsi" not in param_block:
            idx_from_label = _channel_idx_from_label(params.get("Channels"), "T8")
            if idx_from_label is not None and 0 <= idx_from_label < data_array.shape[1]:
                param_block["ChannelIpsi"] = idx_from_label + 1
                param_block.setdefault("PlotChannelLabel", "T8")
            else:
                # Fallback to first channel if T8 not found
                param_block["ChannelIpsi"] = 1
        # Ensure the label matches the chosen ChannelIpsi
        if "PlotChannelLabel" not in param_block and isinstance(params.get("Channels"), (list, tuple)):
            try:
                label_idx = int(param_block.get("ChannelIpsi", 1)) - 1
                ch_entry = params["Channels"][label_idx]
                if isinstance(ch_entry, dict):
                    param_block["PlotChannelLabel"] = ch_entry.get("Channel") or ch_entry.get("Position") or f"Ch{label_idx+1}"
                else:
                    param_block["PlotChannelLabel"] = str(ch_entry)
            except Exception:
                pass

        show_plots = self.show_plots if self.show_plots is not None else not params.get("ReportAnalyzer")
        save_plots = bool(self.save_plots)
        render_plots = show_plots or save_plots
        before_figs = set(plt.get_fignums()) if render_plots else set()

        evaluation = params.setdefault("Evaluation", {})
        ipsi_idx = int(param_block.get("ChannelIpsi", 1)) - 1
        if ipsi_idx < 0 or ipsi_idx >= data_array.shape[1]:
            raise IndexError("ChannelIpsi is out of bounds.")
        plot_label = plot_channel_label or f"Ch{ipsi_idx+1}"

        ipsi_metrics = self._evaluate_channel(
            data_array[:, ipsi_idx],
            fs,
            stim_freq,
            carrier_freq,
            view_low,
            view_high,
            f_min_noise,
            render_plots,
            title_prefix=plot_label,
            context=context,
            params=params,
        )
        evaluation["fft_ipsi"] = ipsi_metrics["fft"]
        evaluation["PSD_ipsi"] = ipsi_metrics["PSD"]
        evaluation["SNR2_45Hz"] = ipsi_metrics["SNR2_45Hz"]
        evaluation["SNR2_maxHz"] = ipsi_metrics["SNR2_maxHz"]
        evaluation["f_test"] = ipsi_metrics["f_test"]
        evaluation["f_test_threshold"] = ipsi_metrics["f_test_threshold"]

        contra_idx = int(param_block.get("ChannelContra", 0)) - 1
        if contra_idx >= 0 and contra_idx < data_array.shape[1]:
            contra_metrics = self._evaluate_channel(
                data_array[:, contra_idx],
                fs,
                stim_freq,
                carrier_freq,
                view_low,
                view_high,
                f_min_noise,
                render_plots,
                title_prefix="Contralateral",
                context=context,
                params=params,
            )
            evaluation["fft_contra"] = contra_metrics["fft"]
            evaluation["PSD_contra"] = contra_metrics["PSD"]

        if render_plots:
            _plot_assr_topomap_from_data(
                data_array,
                fs,
                params,
                stim_freq=topomap_freq,
                vlim_db=(-80, -20),
                contours=8,
            )

        if save_plots and self.save_dir:
            prefix = self.figure_prefix or param_block.get("Filename", "assr")
            saved = save_new_figures(before_figs, self.save_dir, prefix, close=not show_plots)
            if saved:
                evaluation["_figures_saved"] = saved

        params["Evaluation"] = evaluation
        context.params = params

    # ------------------------------------------------------------------
    def _evaluate_channel(
        self,
        signal: np.ndarray,
        fs: float,
        stim_freq: float,
        carrier_freq: float,
        view_low: float,
        view_high: float,
        f_min_noise: float,
        show_plots: bool,
        title_prefix: str,
        context,
        params: dict,
    ) -> Dict[str, Dict[str, np.ndarray] | float]:
        fft_vals, freq = calc_fft(signal, fs)
        fft_vals = np.asarray(fft_vals).squeeze()
        freq = np.asarray(freq)
        if freq.size > 1:
            f_signal_band = float(freq[1] - freq[0])
        elif freq.size == 1:
            f_signal_band = float(freq[0])
        else:
            f_signal_band = 0.0

        # PSD for plotting: use Welch (matches legacy/EEGLAB appearance)
        n_times = signal.shape[0]
        seg = min(1024, n_times)
        psd_welch, freq_welch = mne.time_frequency.psd_array_welch(
            np.asarray(signal),
            sfreq=fs,
            fmin=0.5,
            fmax=500.0,
            average="mean",
            n_fft=seg,
            n_per_seg=seg,
        )
        power_db_welch = 10 * np.log10(psd_welch + np.finfo(float).eps)
        # Drop last bin to avoid edge artifacts (legacy behavior)
        if power_db_welch.size > 1:
            power_db_welch = power_db_welch[:-1]
            freq_welch = freq_welch[:-1]

        # PSD for SNR/metrics (keep existing helper)
        snr_45 = assr_calc_snr(fft_vals, freq, stim_freq, f_min_noise, 45.0, f_signal_band)
        snr_max = assr_calc_snr(fft_vals, freq, stim_freq, f_min_noise, fs / 2.0, f_signal_band)
        f_value, critical = assr_f_test(
            fft_vals,
            freq,
            stim_freq,
            stim_freq - 3.65,
            stim_freq + 3.65,
            f_signal_band,
        )

        if show_plots:
            # PSD around the stimulation frequency (dB/Hz)
            ylabel_psd = "Power (dB/Hz)"
            title_psd = f"{title_prefix} PSD (fm={stim_freq} Hz, fc={carrier_freq} Hz)"
            plot_assr_spectrum(freq_welch, power_db_welch, view_low, view_high, ylabel_psd, title_psd)

            # Full-band PSD (EEGLAB-style) 0-500 Hz for comparison
            _plot_assr_full_psd(
                freq_welch,
                power_db_welch,
                title=f"{title_prefix} PSD (0-500 Hz)",
                xlim=(0, 500),
            )

        metrics: Dict[str, Dict[str, np.ndarray] | float] = {
            "fft": {"xdft": np.asarray(fft_vals), "xdftUnit": "Amplitude (uV)", "freq": freq, "freqUnit": "Frequency (Hz)"},
            "PSD": {
                "freq": freq_welch,
                "freqUnit": "Frequency (Hz)",
                "psdx": psd_welch,
                "psdxUnit": "Power Spectral Density (uV^2/Hz)",
                "dBpsdx": power_db_welch,
                "dBpsdxUnit": "Power Spectral Density (dB/Hz)",
            },
            "SNR2_45Hz": float(snr_45),
            "SNR2_maxHz": float(snr_max),
            "f_test": float(f_value) if np.isfinite(f_value) else float("nan"),
            "f_test_threshold": float(critical),
        }

        return metrics


def _plot_assr_full_psd(
    freq: np.ndarray,
    power_db: np.ndarray,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    smooth_hz: float | None = None,
) -> None:
    """EEGLAB-like PSD plot: dB/Hz with optional smoothing and limits."""
    freqs = np.asarray(freq)
    power = np.asarray(power_db)
    if freqs.size > 1 and power.shape[-1] == freqs.size + 1:
        freqs = freqs[:-1]
        power = power[:-1]
    if freqs.size > 1 and smooth_hz and smooth_hz > 0:
        step = freqs[1] - freqs[0]
        k = max(3, int(round(smooth_hz / step)))
        k = k + (k + 1) % 2  # make odd
        kernel = np.ones(k) / k
        power = np.convolve(power, kernel, mode="same")
    fig = plt.figure(figsize=(10, 4))
    ax = fig.gca()
    ax.plot(freqs, power, color="blue", linewidth=1.25)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB/Hz)")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        finite_power = power[np.isfinite(power)]
        if finite_power.size:
            pad = 5.0
            ax.set_ylim(finite_power.min() - pad, finite_power.max() + pad)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()


def _plot_assr_topomap(context, params: dict, stim_freq: float, vlim_db: Tuple[float, float], contours: int = 8) -> None:
    """Optional ASSR topomap at stim frequency; quietly skips if info/data missing."""
    try:
        raw = getattr(context, "raw", None)
        data = None
        labels: list[str] = []
        fs = None
        if raw is not None and isinstance(raw, mne.io.BaseRaw):
            picks = mne.pick_types(raw.info, eeg=True, stim=False, misc=False, meg=False, ref_meg=False)
            if picks.size:
                data = raw.get_data(picks=picks)
                labels = [raw.ch_names[idx] for idx in picks]
                fs = float(raw.info["sfreq"])
        else:
            data_arr = params.get("data")
            fs = float(params.get("Parameters", {}).get("fs", 0))
            ch_labels = params.get("Channels") or params.get("ChannelLabels") or []
            if data_arr is not None and fs > 0:
                arr = np.asarray(data_arr, dtype=float)
                if arr.ndim == 2:
                    data = arr.T  # expect time x channels -> transpose to ch x time if needed
                elif arr.ndim == 3:
                    data = arr.mean(axis=0).T  # trials x time x ch -> ch x time
                if data is not None:
                    labels = [str(c) for c in ch_labels] if ch_labels else [f"Ch{ii+1}" for ii in range(data.shape[0])]
                    # align labels to data rows
                    if len(labels) < data.shape[0]:
                        labels += [f"Ch{ii+1}" for ii in range(len(labels), data.shape[0])]
                    if len(labels) > data.shape[0]:
                        labels = labels[: data.shape[0]]
                        data = data[: len(labels), :]
        if data is None or fs is None or fs <= 0 or not labels:
            return
        n_times = data.shape[1]
        psd, freqs = mne.time_frequency.psd_array_welch(
            data,
            sfreq=fs,
            fmin=max(1.0, stim_freq - 2),
            fmax=stim_freq + 2,
            average="mean",
            n_fft=min(1024, n_times),
            n_per_seg=min(1024, n_times),
        )
        if psd.ndim != 2 or freqs.size == 0:
            return
        freq_idx = int(np.argmin(np.abs(freqs - stim_freq)))
        values = 10 * np.log10(psd[:, freq_idx] + np.finfo(float).eps)
        fig, _ = plot_cortipy_topomap(
            values,
            ch_names=labels,
            params=params,
            title=f"ASSR Topomap @ {stim_freq:.1f} Hz",
            cbar_label="Power (µV²/Hz)",
            cmap="turbo",
            vlim=vlim_db,
            contours=contours,
        )
    except Exception:
        # fail silently if topo cannot be rendered
        return


def _plot_assr_topomap_from_data(
    data_array: np.ndarray,
    fs: float,
    params: dict,
    stim_freq: float,
    vlim_db: Tuple[float, float],
    contours: int = 8,
) -> None:
    """Topomap using provided data array (samples x channels) to ensure export even without raw."""
    try:
        if data_array.ndim != 2 or data_array.size == 0:
            return
        data = np.asarray(data_array, dtype=float).T  # ch x samples
        labels = params.get("Channels") or params.get("ChannelLabels") or []
        labels = [entry.get("Channel") if isinstance(entry, dict) else entry for entry in labels]
        labels = [str(lbl) for lbl in labels] if labels else [f"Ch{ii+1}" for ii in range(data.shape[0])]
        if len(labels) < data.shape[0]:
            labels += [f"Ch{ii+1}" for ii in range(len(labels), data.shape[0])]
        if len(labels) > data.shape[0]:
            labels = labels[: data.shape[0]]
            data = data[: len(labels), :]
        n_times = data.shape[1]
        psd, freqs = mne.time_frequency.psd_array_welch(
            data,
            sfreq=fs,
            fmin=max(1.0, stim_freq - 2),
            fmax=stim_freq + 2,
            average="mean",
            n_fft=min(1024, n_times),
            n_per_seg=min(1024, n_times),
        )
        if psd.ndim != 2 or freqs.size == 0:
            return
        freq_idx = int(np.argmin(np.abs(freqs - stim_freq)))
        values = 10 * np.log10(psd[:, freq_idx] + np.finfo(float).eps)
        fig, _ = plot_cortipy_topomap(
            values,
            ch_names=labels,
            params=params,
            title=f"ASSR Topomap @ {stim_freq:.1f} Hz",
            cbar_label="Power (µV²/Hz)",
            cmap="turbo",
            vlim=vlim_db,
            contours=contours,
        )
    except Exception:
        return


def _channel_idx_from_label(channels, label) -> int | None:
    """Resolve a 0-based channel index from label or numeric string."""
    if label is None:
        return None
    try:
        idx = int(label) - 1
        if idx >= 0:
            return idx
    except Exception:
        pass
    labels = []
    for entry in channels or []:
        if isinstance(entry, dict):
            lbl = entry.get("Channel") or entry.get("label") or entry.get("name") or entry.get("Position")
        else:
            lbl = entry
        labels.append(str(lbl).lower())
    try:
        return labels.index(str(label).lower())
    except ValueError:
        return None
