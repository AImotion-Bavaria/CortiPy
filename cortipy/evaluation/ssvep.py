"""SSVEP evaluation pipeline."""

from __future__ import annotations

from typing import Dict, Tuple

import logging
import matplotlib
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

import matplotlib.pyplot as plt

from cortipy.evaluation.base import EvaluatorBase
from cortipy.shared import (
    assr_compute_psd,
    calc_fft,
    cca_correlations,
    compute_t2circ,
    plot_psd_ssvep,
    ssvep_f_test,
    ssvep_snr,
)

LOGGER = logging.getLogger("cortipy.evaluation.ssvep")


class SsvepEvaluator(EvaluatorBase):
    """Python port of EvalSSVEPmain."""

    def __init__(self, show_plots: bool | None = None) -> None:
        self.show_plots = show_plots

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
        psd_result = assr_compute_psd(data_ref, fs)

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

        show_plots = self.show_plots if self.show_plots is not None else not params.get("ReportAnalyzer")
        if show_plots:
            LOGGER.debug("Rendering SSVEP evaluation plots")
            low = param_block.get("LowestFrequency", 2)
            high = param_block.get("HighestFrequency", 45)
            avg_psd = psd_result.dBpsd.mean(axis=1)
            fig_psd, ax_psd = _get_axes("ssvep_psd")
            plot_psd_ssvep(ax_psd, psd_result.freq, avg_psd, low, high)
            _show_mpl(fig_psd, "ssvep_psd")

            avg_fft = np.abs(spectrum).mean(axis=1)
            fig_fft, ax_fft = _get_axes("ssvep_fft")
            plot_psd_ssvep(ax_fft, freq, avg_fft, low, high)
            _show_mpl(fig_fft, "ssvep_fft")

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
