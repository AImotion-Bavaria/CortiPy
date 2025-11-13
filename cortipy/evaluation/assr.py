"""ASSR evaluation pipeline."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from cortipy.evaluation.base import EvaluatorBase
from cortipy.shared import (
    assr_calc_snr,
    assr_compute_psd,
    assr_f_test,
    calc_fft,
    plot_assr_spectrum,
)


class AssrEvaluator(EvaluatorBase):
    """Port of the MATLAB ASSR FFT/SNR evaluation."""

    def __init__(self, show_plots: bool | None = None) -> None:
        self.show_plots = show_plots

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
        carrier_freq = float(param_block.get("ASSRCarrierFrequency", 0))
        view_low = stim_freq - 10.0
        view_high = stim_freq + 10.0
        f_min_noise = 2.0

        data_array = np.asarray(data, dtype=float)
        device = params.get("Device")
        if device == "ActiCHamp":
            ref_idx = int(param_block.get("ReferenceChannel", 1)) - 1
            data_array = data_array - data_array[:, [ref_idx]]

        show_plots = self.show_plots if self.show_plots is not None else not params.get("ReportAnalyzer")

        evaluation = params.setdefault("Evaluation", {})
        f_resolution = fs / data_array.shape[0] if data_array.shape[0] > 0 else 0.0

        ipsi_idx = int(param_block.get("ChannelIpsi", 1)) - 1
        if ipsi_idx < 0 or ipsi_idx >= data_array.shape[1]:
            raise IndexError("ChannelIpsi is out of bounds.")

        ipsi_metrics = self._evaluate_channel(
            data_array[:, ipsi_idx],
            fs,
            stim_freq,
            carrier_freq,
            view_low,
            view_high,
            f_min_noise,
            show_plots,
            title_prefix="Ipsilateral",
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
                show_plots,
                title_prefix="Contralateral",
            )
            evaluation["fft_contra"] = contra_metrics["fft"]
            evaluation["PSD_contra"] = contra_metrics["PSD"]

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

        psd_result = assr_compute_psd(signal, fs)
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
            ylabel_psd = "Power Spectral Density (dB/Hz)"
            title_psd = f"{title_prefix} PSD (fm={stim_freq} Hz, fc={carrier_freq} Hz)"
            plot_assr_spectrum(psd_result.freq, psd_result.dBpsd, view_low, view_high, ylabel_psd, title_psd)

            ylabel_fft = "Amplitude (uV)"
            title_fft = f"{title_prefix} FFT (fm={stim_freq} Hz, fc={carrier_freq} Hz)"
            plot_assr_spectrum(freq, np.abs(fft_vals), view_low, view_high, ylabel_fft, title_fft)

        metrics: Dict[str, Dict[str, np.ndarray] | float] = {
            "fft": {"xdft": np.asarray(fft_vals), "xdftUnit": "Amplitude (uV)", "freq": freq, "freqUnit": "Frequency (Hz)"},
            "PSD": {
                "freq": psd_result.freq,
                "freqUnit": "Frequency (Hz)",
                "psdx": psd_result.psd,
                "psdxUnit": "Power Spectral Density (uV^2/Hz)",
                "dBpsdx": psd_result.dBpsd,
                "dBpsdxUnit": "Power Spectral Density (dB/Hz)",
            },
            "SNR2_45Hz": float(snr_45),
            "SNR2_maxHz": float(snr_max),
            "f_test": float(f_value) if np.isfinite(f_value) else float("nan"),
            "f_test_threshold": float(critical),
        }

        return metrics
