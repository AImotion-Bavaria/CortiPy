"""SSVEP evaluation pipeline."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

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


class SsvepEvaluator(EvaluatorBase):
    """Python port of EvalSSVEPmain."""

    def __init__(self, show_plots: bool | None = None) -> None:
        self.show_plots = show_plots

    def evaluate(self, context) -> None:  # type: ignore[override]
        params = context.params
        if str(params.get("Method", "")).lower() != "ssvep":
            return

        data = params.get("data")
        if data is None:
            raise ValueError("SsvepEvaluator requires `params['data']`.")

        param_block = params.setdefault("Parameters", {})
        fs = float(param_block.get("fs", 0))
        if fs <= 0:
            raise ValueError("SSVEP evaluation requires Params.Parameters.fs.")

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

        stim_freqs = np.atleast_1d(np.asarray(param_block.get("StimFreq"), dtype=float))
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
            low = param_block.get("LowestFrequency", 2)
            high = param_block.get("HighestFrequency", 45)
            avg_psd = psd_result.dBpsd.mean(axis=1)
            plot_psd_ssvep(_get_axes(), psd_result.freq, avg_psd, low, high)
            avg_fft = np.abs(spectrum).mean(axis=1)
            plot_psd_ssvep(_get_axes(), freq, avg_fft, low, high)

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
        raise RuntimeError("SSVEP evaluation supports ActiCHamp or UNICORN data.")


def _get_axes():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    return fig.gca()

