"""SSVEP evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
import logging
import matplotlib
import numpy as np
import mne
from cortipy.shared.assr import SpectralResult

try:  # pragma: no cover - optional Streamlit integration
    import streamlit as st
except Exception:  # pragma: no cover
    st = None
    _STREAMLIT_RUNTIME = False
else:
    _STREAMLIT_RUNTIME = bool(getattr(st, "runtime", None) and st.runtime.exists())
    if _STREAMLIT_RUNTIME:
        matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from cortipy.evaluation.base import EvaluatorBase, save_new_figures
from cortipy.shared import (
    calc_fft,
    cca_correlations,
    compute_t2circ,
    plot_cortipy_topomap,
    ssvep_f_test,
    ssvep_snr,
)

LOGGER = logging.getLogger("cortipy.evaluation.ssvep")


class SsvepEvaluator(EvaluatorBase):
    """Python port of EvalSSVEPmain."""

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
        LOGGER.debug("SsvepEvaluator.evaluate invoked")
        params = context.params
        if str(params.get("Method", "")).lower() != "ssvep":
            LOGGER.debug("SsvepEvaluator skipped: method=%s", params.get("Method"))
            return

        data = params.get("data")
        if data is None:
            raise ValueError("SsvepEvaluator requires `params['data']`.")

        param_block = params.setdefault("Parameters", {})
        fs = float(np.nan_to_num(param_block.get("fs", 0), nan=0.0))
        if fs <= 0:
            # try to infer from RecordingTime and data length
            rec_time = param_block.get("RecordingTime")
            try:
                rec_time_f = float(rec_time)
            except (TypeError, ValueError):
                rec_time_f = None
            if rec_time_f and rec_time_f > 0:
                fs = float(data.shape[0]) / rec_time_f
        if fs <= 0 or not np.isfinite(fs):
            raise ValueError("SSVEP evaluation requires a positive sampling rate (fs).")

        data_ref = self._apply_reference(params, np.asarray(data, dtype=float))
        spectrum, freq = calc_fft(data_ref, fs)
        # Use Welch PSD (aligned with legacy/EEGLAB appearance)
        psd_result = _compute_ssvep_psd(data_ref, fs)

        evaluation = params.setdefault("Evaluation", {})
        evaluation["fft"] = {
            "xdft": spectrum,
            "xdftUnit": "Amplitude (uV)",
            "freq": freq,
            "freqUnit": "Frequency (Hz)",
        }
        evaluation["PSD"] = {
            "freq": psd_result.freq,
            "freqUnit": "Frequency (Hz)",
            "psdx": psd_result.psd,
            "psdxUnit": "Power Spectral Density (uV^2/Hz)",
            "dBpsdx": psd_result.dBpsd,
            "dBpsdxUnit": "Power Spectral Density (dB/Hz)",
        }

        stim_freqs_raw = param_block.get("StimFreq")
        stim_freqs = np.atleast_1d(np.asarray(stim_freqs_raw, dtype=float))
        stim_freqs = stim_freqs[np.isfinite(stim_freqs) & (stim_freqs > 0)]
        if stim_freqs.size == 0:
            stim_freqs = np.array([10.0], dtype=float)
            param_block["StimFreq"] = stim_freqs.tolist()
        freq_spacing = freq[1] - freq[0] if freq.size > 1 else 1.0
        snr_2_45 = ssvep_snr(spectrum, freq, stim_freqs, 2.0, 45.0, freq_spacing)
        snr_max = ssvep_snr(spectrum, freq, stim_freqs, 2.0, fs / 2.0, freq_spacing)
        evaluation["SNR_2_45Hz"] = snr_2_45
        evaluation["SNR_max"] = snr_max

        t2circ, p_values = compute_t2circ(data_ref, fs, stim_freqs)
        evaluation["T2circ"] = t2circ
        evaluation["p_values"] = p_values

        rho, f_cca = cca_correlations(data_ref, fs, stim_freqs)
        evaluation["rho"] = rho
        evaluation["f_CCA"] = f_cca

        f_test = ssvep_f_test(psd_result.psd, psd_result.freq, float(stim_freqs[0]))
        if f_test:
            evaluation["F_Test"] = f_test

        plot_channel_label = param_block.get("PlotChannelLabel", "Oz")
        # Resolve channel index from label; default to Oz when available.
        plot_idx = _channel_idx_from_label(params, plot_channel_label)
        if plot_idx is None:
            plot_idx = _channel_idx_from_label(params, "Oz")
        if plot_idx is None or plot_idx < 0 or plot_idx >= psd_result.dBpsd.shape[0]:
            plot_idx = 0
        param_block["PlotChannelLabel"] = plot_channel_label
        param_block["ChannelIpsi"] = plot_idx + 1
        param_block["ChannelContra"] = plot_idx + 1
        topomap_freq = float(param_block.get("TopomapFrequencyHz", stim_freqs[0]))

        show_plots = self.show_plots if self.show_plots is not None else not params.get("ReportAnalyzer")
        save_plots = bool(self.save_plots)
        render_plots = show_plots or save_plots
        before_figs = set(plt.get_fignums()) if render_plots else set()
        if render_plots:
            LOGGER.debug("Rendering SSVEP evaluation plots")
            fig_psd, ax_psd = _get_axes("ssvep_psd")
            plot_ssvep_power_db(
                ax_psd,
                psd_result.freq,
                _pick_psd_channel(psd_result.dBpsd, plot_idx),
                smooth=1.0,
                xlim=(0, 500),
                ylim=(-100, -20),
                title=f"SSVEP PSD @ {plot_channel_label}",
            )
            _show_mpl(fig_psd, "ssvep_psd")
            _plot_ssvep_topomap(
                context,
                params,
                stim_freq=float(topomap_freq),
                vlim_db=(-80, -20),
                contours=8,
            )

        if save_plots and self.save_dir:
            prefix = self.figure_prefix or param_block.get("Filename", "ssvep")
            saved = save_new_figures(before_figs, self.save_dir, prefix, close=not show_plots)
            if saved:
                evaluation["_figures_saved"] = saved

        context.params = params

    # ------------------------------------------------------------------
    def _apply_reference(self, params: dict, data: np.ndarray) -> np.ndarray:
        device = str(params.get("Device", "")).lower()
        param_block = params.get("Parameters", {})
        num_channels = int(param_block.get("NumberEEGChannels", data.shape[1]))
        if device == "actichamp":
            ref_idx = int(param_block.get("ReferenceChannel", 1)) - 1
            referenced = data - data[:, [ref_idx]]
            if referenced.shape[1] > num_channels:
                referenced = referenced[:, :num_channels]
            return referenced
        if device == "unicorn":
            return data[:, :8]
        # Generic fallback: keep the first `num_channels` channels without re-referencing.
        if data.shape[1] > num_channels:
            return data[:, :num_channels]
        return data


def _get_axes(key: str):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.gca()
    fig.tight_layout()
    return fig, ax


def _show_mpl(fig: plt.Figure, key: str) -> None:
    LOGGER.debug("_show_mpl called", extra={"key": key})
    if not _is_streamlit_runtime():
        LOGGER.debug("_show_mpl skipped (no streamlit runtime)", extra={"key": key})
        return
    placeholders = st.session_state.setdefault("_ssvep_eval_placeholders", {})
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


def _plot_ssvep_topomap(
    context,
    params: dict,
    stim_freq: float,
    vlim_db: tuple[float, float],
    contours: int = 8,
) -> None:
    """Optional SSVEP topomap at stim frequency; quietly skips if info/data missing."""
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
            data_arr = params.get("data")
            fs = float(params.get("Parameters", {}).get("fs", 0))
            ch_labels = params.get("Channels") or params.get("ChannelLabels") or []
            if data_arr is not None and fs > 0:
                arr = np.asarray(data_arr, dtype=float)
                if arr.ndim == 2:
                    data = arr.T
                elif arr.ndim == 3:
                    data = arr.mean(axis=0).T
                if data is not None:
                    resolved = []
                    for entry in ch_labels or []:
                        if isinstance(entry, dict):
                            lbl = entry.get("Channel") or entry.get("Position") or entry.get("label") or entry.get("name")
                        else:
                            lbl = entry
                        resolved.append(str(lbl) if lbl is not None else "")
                    labels = resolved or [f"Ch{ii+1}" for ii in range(data.shape[0])]
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
        # Fixed vlim to match EEGLAB/legacy appearance
        vlim_fixed = (-60.0, -20.0)
        fig, _ = plot_cortipy_topomap(
            values,
            ch_names=labels,
            params=params,
            title=f"SSVEP Topomap @ {stim_freq:.1f} Hz",
            cbar_label="Power (µV²/Hz)",
            cmap="turbo",
            vlim=vlim_fixed,
            contours=contours,
            sphere=(0.0, -0.01, 0.0, 0.105),
            figsize=(12, 12),
            dpi=400,
            colorbar=True,
            show_names=False,
        )
    except Exception:
        return


def plot_ssvep_power_db(
    ax: plt.Axes,
    freqs: np.ndarray,
    power_db: np.ndarray,
    window_hz: float = 1.0,
    xlim: tuple[float, float] = (0, 500),
    ylim: tuple[float, float] = (-100, -20),
    title: str | None = None,
    smooth: float | None = None,
) -> plt.Axes:
    """Plot SSVEP power (dB/Hz) with optional light smoothing to mirror EEGLAB-style PSD."""
    freqs = np.asarray(freqs)
    power_db = np.asarray(power_db)
    # choose smoothing window: prefer `smooth` if provided, else window_hz
    smooth_win = smooth if smooth is not None else window_hz
    if freqs.size > 1 and smooth_win and smooth_win > 0:
        step = freqs[1] - freqs[0]
        k = max(3, int(round(smooth_win / step)))
        k = k + (k + 1) % 2  # enforce odd length
        kernel = np.ones(k) / k
        power_db = np.convolve(power_db, kernel, mode="same")
    if power_db.size == freqs.size + 1:
        freqs = freqs[:-1]
        power_db = power_db[:-1]
    ax.plot(freqs, power_db, color="blue", linewidth=1.25)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV^2/Hz)")
    if title:
        ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    return ax


def _channel_idx_from_label(params: dict, label) -> int | None:
    """Resolve a 0-based channel index from a label or numeric value."""
    if label is None:
        return None
    # numeric labels can be passed directly
    try:
        lbl_int = int(label)
    except (TypeError, ValueError):
        lbl_int = None
    else:
        if lbl_int > 0:
            return lbl_int - 1
    labels = params.get("Channels") or params.get("ChannelLabels") or params.get("ChannelLabelsEEG") or []
    labels = [str(lab).lower() for lab in labels]
    try:
        return labels.index(str(label).lower())
    except ValueError:
        return None


def _pick_psd_channel(psd_db: np.ndarray, idx: int) -> np.ndarray:
    """Safely pick a PSD channel, falling back to the first channel."""
    psd_db = np.asarray(psd_db)
    if psd_db.ndim == 1:
        return psd_db
    if psd_db.ndim > 1:
        # handle shapes (channels, freqs) or (freqs, channels)
        if psd_db.shape[0] <= psd_db.shape[1]:
            # likely channels x freqs
            if idx < 0 or idx >= psd_db.shape[0]:
                idx = 0
            return psd_db[idx, :]
        else:
            # likely freqs x channels
            if idx < 0 or idx >= psd_db.shape[1]:
                idx = 0
            return psd_db[:, idx]
    return psd_db


def _compute_ssvep_psd(data_ref: np.ndarray, fs: float) -> SpectralResult:
    """Welch PSD up to 500 Hz with light smoothing and dropped last bin (EEGLAB-like)."""
    data_ref = np.asarray(data_ref, dtype=float)
    n_times = data_ref.shape[0]
    seg = min(2048, n_times)
    psd, freqs = mne.time_frequency.psd_array_welch(
        data_ref.T,
        sfreq=fs,
        fmin=0.5,
        fmax=500.0,
        average="mean",
        n_fft=seg,
        n_per_seg=seg,
    )
    power_db = 10 * np.log10(psd + np.finfo(float).eps)
    if freqs.size > 5:
        k = max(3, int(round(1 / (freqs[1] - freqs[0]))))
        k = k + (k + 1) % 2  # enforce odd
        kernel = np.ones(k) / k
        power_db = np.convolve(power_db, kernel, mode="same")
    if power_db.shape[1] > 1:
        power_db = power_db[:, :-1]
        freqs = freqs[:-1]
        psd = psd[:, :-1]
    return SpectralResult(psd=psd, freq=freqs, dBpsd=power_db)
