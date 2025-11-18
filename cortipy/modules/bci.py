"""SSVEP-BCI acquisition with FFT/T²circ/CCA metrics and Bluetooth output."""

from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cortipy.core.context import ModuleContext
from cortipy.evaluation.ssvep import SsvepEvaluator
from cortipy.shared import (
    calc_fft,
    cca_correlations,
    classify_fft,
    compute_t2circ,
    describe_prediction,
    get_frequency_indices,
    plot_psd_ssvep,
    send_prediction_bt,
)
from cortipy.shared.notifications import info_end_live, info_start_live

from .base import ModuleBase


class BciModule(ModuleBase):
    """Python translation of the MATLAB SSVEP-BCI module."""

    def __init__(self, evaluator: SsvepEvaluator | None = None) -> None:
        super().__init__("SSVEP-BCI", ["BCI"], evaluator or SsvepEvaluator())
        self.warmup_duration = 1.0
        self.step_duration = 1.0
        self.window_duration = 4.0
        self._unknown_device_warned = False
        self._spectrum_fig = None

    def collect_measurements(self, context: ModuleContext) -> None:
        params = self._validate_parameters(context.params)
        context.params = params
        param_block = params["Parameters"]

        device = self.require_device(context)
        info_start_live()
        aux_ch = param_block.get("NumberAUXChannels", 0) if params.get("Device") == "ActiCHamp" else 0
        bluetooth_conn, bt_cleanup = self._try_open_bluetooth(param_block)

        data = self._ensure_array(device.prime(self.warmup_duration, aux_ch))
        elapsed = self.warmup_duration
        fs = float(param_block["fs"])
        stim_freqs = np.atleast_1d(np.asarray(param_block["StimFreq"], dtype=float))
        window_samples = max(1, int(round(self.window_duration * fs)))
        freq_range = self._resolve_frequency_range(param_block, stim_freqs)
        detection_channels = self._resolve_detection_channels(param_block, data.shape[1])

        prediction_history: List[np.ndarray] = []
        prediction_times: List[float] = []
        t2_history: List[np.ndarray] = []
        p_history: List[np.ndarray] = []
        cca_history: List[np.ndarray] = []
        cca_freqs = np.array([])

        try:
            while elapsed < float(param_block["RecordingTime"]):
                step = min(self.step_duration, param_block["RecordingTime"] - elapsed)
                chunk = self._ensure_array(device.acquire(step, aux_ch))
                data = np.vstack([data, chunk])
                elapsed += step

                window = self._select_recent_window(data, window_samples)
                window_ref = self._apply_reference(params, window)
                aggregated = self._aggregate_channels(window_ref, detection_channels)

                spectrum, freq = calc_fft(aggregated, fs)
                freq_idx = get_frequency_indices(freq, stim_freqs)
                prediction = classify_fft(freq, spectrum, freq_idx, freq_range[0], freq_range[1])
                prediction_history.append(prediction)
                prediction_times.append(elapsed)

                t2_values, p_values = compute_t2circ(window_ref[:, detection_channels], fs, stim_freqs)
                t2_history.append(t2_values)
                p_history.append(p_values)

                cca_rho, cca_freqs = cca_correlations(window_ref[:, detection_channels], fs, stim_freqs)
                cca_history.append(cca_rho)

                self._plot_spectrum(freq, np.abs(spectrum), freq_range, params)
                self._dispatch_prediction(prediction, stim_freqs, bluetooth_conn)
                description = describe_prediction(prediction, stim_freqs)
                if description:
                    print(f"[BCI] {description} @ {elapsed:.1f}s")
        finally:
            info_end_live()
            if bt_cleanup:
                with contextlib.suppress(Exception):
                    bt_cleanup()

        context.data_buffer = data
        params["data"] = data
        evaluation = params.setdefault("Evaluation", {})
        evaluation["BCI"] = {
            "StimFreq": stim_freqs,
            "PredictionHistory": np.vstack(prediction_history) if prediction_history else np.empty((0, len(stim_freqs))),
            "PredictionTimes": np.asarray(prediction_times, dtype=float),
            "T2circ": np.vstack(t2_history) if t2_history else np.empty((0, len(detection_channels))),
            "T2circPValue": np.vstack(p_history) if p_history else np.empty((0, len(detection_channels))),
            "CCA": {
                "rho": np.vstack(cca_history) if cca_history else np.empty((0, len(cca_freqs))),
                "freq": cca_freqs,
            },
        }
        params["Evaluation"] = evaluation

    # ------------------------------------------------------------------
    def _validate_parameters(self, params):
        param_block = params.setdefault("Parameters", {})
        required = ["fs", "RecordingTime", "NumberEEGChannels", "ReferenceChannel", "StimFreq"]
        missing = [key for key in required if key not in param_block]
        if missing:
            raise ValueError(f"Missing BCI parameter(s): {', '.join(missing)}")
        stim_freq = param_block["StimFreq"]
        if isinstance(stim_freq, (int, float)):
            param_block["StimFreq"] = [float(stim_freq)]
        elif isinstance(stim_freq, str):
            param_block["StimFreq"] = [float(tok) for tok in stim_freq.split()]
        return params

    def _ensure_array(self, data) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        return arr

    def _select_recent_window(self, data: np.ndarray, window_samples: int) -> np.ndarray:
        if data.shape[0] <= window_samples:
            return data
        return data[-window_samples:, :]

    def _apply_reference(self, params: dict, data: np.ndarray) -> np.ndarray:
        params_block = params.get("Parameters", {})
        device = str(params.get("Device", "")).lower()
        referenced = np.array(data, copy=True)
        if device == "actichamp":
            ref_idx = int(params_block.get("ReferenceChannel", 1)) - 1
            num_cols = referenced.shape[1]
            mask = np.ones(num_cols, dtype=bool)
            trig_idx = int(params_block.get("TriggerChannel", num_cols)) - 1
            if 0 <= trig_idx < num_cols:
                mask[trig_idx] = False
            mask &= np.arange(num_cols) < int(params_block.get("NumberEEGChannels", num_cols))
            referenced[:, mask] = referenced[:, mask] - referenced[:, [ref_idx]]
        elif device == "unicorn":
            referenced = referenced[:, : min(referenced.shape[1], params_block.get("NumberEEGChannels", 8))]
        else:
            referenced = referenced[:, : min(referenced.shape[1], params_block.get("NumberEEGChannels", referenced.shape[1]))]
            if not self._unknown_device_warned:
                print(f"[BCI] Warning: device '{params.get('Device')}' not fully supported; using raw EEG channels.")
                self._unknown_device_warned = True
        return referenced

    def _resolve_detection_channels(self, params_block: dict, total_columns: int) -> np.ndarray:
        num_eeg = min(total_columns, int(params_block.get("NumberEEGChannels", total_columns)))
        channels = list(range(num_eeg))
        excluded = set()
        for key in ("ReferenceChannel", "TriggerChannel"):
            idx = int(params_block.get(key, 0)) - 1
            if 0 <= idx < num_eeg:
                excluded.add(idx)
        filtered = [ch for ch in channels if ch not in excluded]
        if not filtered:
            filtered = channels
        return np.asarray(filtered, dtype=int)

    def _aggregate_channels(self, data: np.ndarray, channel_idx: np.ndarray) -> np.ndarray:
        if channel_idx.size == 0:
            return np.mean(data, axis=1)
        return np.mean(data[:, channel_idx], axis=1)

    def _plot_spectrum(self, freq: np.ndarray, spectrum: np.ndarray, freq_range: Tuple[float, float], params: dict) -> None:
        import matplotlib.pyplot as plt

        if self._spectrum_fig is None or not plt.fignum_exists(self._spectrum_fig.number):
            self._spectrum_fig = plt.figure(figsize=(10, 4))
        ax = self._spectrum_fig.gca()
        ax.clear()
        plot_psd_ssvep(ax, freq, spectrum, freq_range[0], freq_range[1])
        ax.set_title(f"BCI Spectrum — {params.get('Device', '')}")
        self._spectrum_fig.tight_layout()
        self._spectrum_fig.canvas.draw_idle()

    def _dispatch_prediction(self, prediction: np.ndarray, stim_freqs: np.ndarray, bt_connection) -> None:
        send_prediction_bt(prediction, bt_connection)

    def _resolve_frequency_range(self, params_block: dict, stim_freqs: np.ndarray) -> Tuple[float, float]:
        default_low = max(0.1, float(np.min(stim_freqs)) - 3) if stim_freqs.size else 0.1
        default_high = float(np.max(stim_freqs)) + 5 if stim_freqs.size else 40.0
        low_f = float(params_block.get("LowestFrequency", default_low))
        high_f = float(params_block.get("HighestFrequency", default_high))
        if high_f <= low_f:
            high_f = low_f + 10
        return low_f, high_f

    def _try_open_bluetooth(self, params_block: dict) -> Tuple[Optional[object], Optional[callable]]:
        target = params_block.get("BluetoothDevice") or params_block.get("BluetoothAddress")
        if not target:
            return None, None
        channel = int(params_block.get("BluetoothChannel", 1))
        try:
            import bluetooth  # type: ignore
        except ImportError:
            print("[BCI] Bluetooth support unavailable; skipping wireless output.")
            return None, None
        try:
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            sock.connect((target, channel))

            def cleanup() -> None:
                with contextlib.suppress(Exception):
                    sock.close()

            return sock, cleanup
        except Exception as exc:
            print(f"[BCI] Could not connect to Bluetooth target '{target}': {exc}")
            return None, None
