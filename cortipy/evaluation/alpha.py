"""Alpha evaluation pipeline: PSD, spectrograms, and SNR metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import logging
import matplotlib
import numpy as np
from scipy.signal import resample, spectrogram

try:  # pragma: no cover - optional Streamlit integration
    import streamlit as st
except Exception:  # pragma: no cover
    st = None
    _STREAMLIT_RUNTIME = False
else:
    _STREAMLIT_RUNTIME = bool(getattr(st, "runtime", None) and st.runtime.exists())
    if _STREAMLIT_RUNTIME:
        matplotlib.use("Agg", force=True)
    # Note: runtime detection is refreshed inside _is_streamlit_runtime.

import matplotlib.pyplot as plt

from cortipy.evaluation.base import EvaluatorBase, save_new_figures
from cortipy.shared.signal import hann_window, time_vector

EPS = np.finfo(np.float64).eps
LOGGER = logging.getLogger("cortipy.evaluation.alpha")


def _show_mpl(fig: plt.Figure, key: str) -> None:
    if not _is_streamlit_runtime():
        LOGGER.debug("_show_mpl skipped (no streamlit runtime)", extra={"key": key})
        return
    placeholders = st.session_state.setdefault("_alpha_eval_placeholders", {})
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
        LOGGER.debug("_is_streamlit_runtime: st is None")
        return False
    runtime = getattr(st, "runtime", None)
    exists = bool(runtime and runtime.exists())
    LOGGER.debug("_is_streamlit_runtime", extra={"exists": exists})
    return exists


class AlphaEvaluator(EvaluatorBase):
    """Port of the MATLAB alpha evaluation stack."""

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
        LOGGER.debug("AlphaEvaluator.evaluate invoked")
        params = context.params
        if str(params.get("Method", "")).lower() != "alpha":
            LOGGER.debug("AlphaEvaluator skipped: method=%s", params.get("Method"))
            context.params = params
            return

        data = params.get("data")
        if data is None:
            raise ValueError("AlphaEvaluator requires `params['data']`.")

        param_block = params.setdefault("Parameters", {})
        fs = float(param_block.get("fs", 0))
        if fs <= 0:
            raise ValueError("Alpha evaluation requires a positive sampling rate (Params.Parameters.fs).")

        data_ref, n_channels, trigger_idx = _apply_reference(np.asarray(data, dtype=float), params)

        time_seconds = time_vector(data_ref, fs, unit="s")
        trigger_mode = str(param_block.get("Trigger", "")).lower()
        if trigger_mode == "fixed":
            trigger_time = float(param_block.get("TriggerTime", 0))
            period = 2.0 * trigger_time if trigger_time > 0 else 0
            data_ref, trigger_idx = _append_synthetic_trigger(data_ref, time_seconds, trigger_time, period, params)

        show_plots = self.show_plots if self.show_plots is not None else not params.get("ReportAnalyzer")
        save_plots = bool(self.save_plots)
        render_plots = show_plots or save_plots
        before_figs = set(plt.get_fignums()) if render_plots else set()
        LOGGER.debug("AlphaEvaluator show_plots=%s save_plots=%s", show_plots, save_plots)

        trigger_channel = trigger_idx if trigger_idx is not None else data_ref.shape[1] - 1
        trigger_times = trigger_timestamp(data_ref, trigger_channel, without_first_seconds=1, fs=fs)
        if trigger_mode == "external" and trigger_times.size > 0:
            trigger_times = trigger_times[:-1]

        window_width = float(param_block.get("AlphaWindowWidth", 2.0))
        overlap = float(param_block.get("AlphaWindowOverlap", 0.5))
        analysis_window = hann_window(int(round(window_width * fs)), periodic=True)

        alpha_power_rows: List[np.ndarray] = []
        psd_rows: List[np.ndarray] = []
        mean_closed: List[float] = []
        mean_open: List[float] = []
        std_closed: List[float] = []
        std_open: List[float] = []
        channel_numbers: List[int] = []
        channel_labels: List[str] = []
        time_axis: Optional[np.ndarray] = None
        freq_axis: Optional[np.ndarray] = None

        start_state = str(param_block.get("Start", "eyesClosed")).lower()
        start_closed = start_state != "eyesopen"

        for ch in range(min(n_channels, data_ref.shape[1])):
            if trigger_idx is not None and ch == trigger_idx:
                continue

            channel_signal = data_ref[:, ch]
            band_power, _, T, F, power_spectrogram = calc_psd_power_time(
                channel_signal, analysis_window, overlap, fs
            )
            time_axis = T if time_axis is None else time_axis
            freq_axis = F if freq_axis is None else freq_axis

            if render_plots:
                LOGGER.debug("Calling plot_psd_time for channel %s", ch + 1)
                plot_psd_time(T, band_power, ch + 1)

            if render_plots and trigger_times.size:
                LOGGER.debug("Calling plot_spectrogram for channel %s", ch + 1)
                plot_spectrogram(F, power_spectrogram, T, trigger_times, ch + 1)

            indices = nearest_indices(T, trigger_times) if trigger_times.size else np.array([])
            closed_vals, open_vals = split_alpha_segments(band_power, indices, start_closed)

            mean_closed.append(float(np.mean(closed_vals)) if closed_vals.size else np.nan)
            mean_open.append(float(np.mean(open_vals)) if open_vals.size else np.nan)
            std_closed.append(float(np.std(closed_vals)) if closed_vals.size else np.nan)
            std_open.append(float(np.std(open_vals)) if open_vals.size else np.nan)

            alpha_power_rows.append(band_power)
            psd_rows.append(power_spectrogram)
            channel_numbers.append(ch + 1)
            channel_labels.append(resolve_channel_label(params.get("Channels"), ch))

        if not alpha_power_rows or time_axis is None or freq_axis is None:
            params.setdefault("Evaluation", {})  # ensure key exists even if empty
            context.params = params
            return

        alpha_power_matrix = np.vstack(alpha_power_rows)
        psd_cube = np.stack(psd_rows)

        evaluation = params.setdefault("Evaluation", {})
        evaluation["alphaPower"] = {
            "time": time_axis,
            "timeUnit": "Time (s)",
            "dBpsdx": alpha_power_matrix,
            "dBpsdxUnit": "Power (dB) [rel. to 1 uV/sqrt(Hz)]",
        }
        evaluation["PSD"] = {
            "time": time_axis,
            "timeUnit": "Time (s)",
            "freq": freq_axis,
            "freqUnit": "Frequency (Hz)",
            "psdx": psd_cube,
            "psdxInfo": "(CH, F, T)",
            "psdxUnit": "Power Spectral Density (uV^2/Hz)",
        }

        trig_channel = trigger_idx if trigger_idx is not None else -1
        evaluation["trigger"] = {
            "time": time_seconds,
            "timeUnit": "Time (s)",
            "voltage": data_ref[:, trig_channel] if trig_channel >= 0 else np.zeros_like(time_seconds),
            "voltageUnit": "Amplitude (uV)",
        }

        dif_closed_open = np.divide(
            np.asarray(mean_closed),
            np.asarray(mean_open),
            out=np.full(len(mean_closed), np.nan),
            where=np.asarray(mean_open) != 0,
        )
        evaluation["Stat"] = {
            "Channel": channel_numbers,
            "difClosedOpen": dif_closed_open,
            "meanClosed": np.asarray(mean_closed),
            "meanOpen": np.asarray(mean_open),
            "STDclosed": np.asarray(std_closed),
            "STDopen": np.asarray(std_open),
        }

        evaluation = alpha_snr_ram(
            num_channels=len(channel_numbers),
            evaluation=evaluation,
            trigger_times=trigger_times,
            time_axis=time_axis,
            freq_axis=freq_axis,
            psd_cube=psd_cube,
            params=param_block,
        )

        if render_plots:
            plot_alpha_matrix(
                evaluation,
                reference_channel=int(param_block.get("ReferenceChannel", 1)),
                channel_labels=channel_labels,
            )

        params["Evaluation"] = evaluation
        if save_plots and self.save_dir:
            prefix = self.figure_prefix or param_block.get("Filename", "alpha")
            saved = save_new_figures(before_figs, self.save_dir, prefix, close=not show_plots)
            if saved:
                evaluation["_figures_saved"] = saved
        context.params = params


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _apply_reference(data: np.ndarray, params: MutableMapping[str, Any]) -> Tuple[np.ndarray, int, Optional[int]]:
    param_block = params.setdefault("Parameters", {})
    device = str(params.get("Device", "")).lower()
    trig_idx = int(param_block.get("TriggerChannel", data.shape[1])) - 1

    data_ref = np.array(data, dtype=float, copy=True)
    n_channels = data_ref.shape[1]

    if device == "actichamp":
        reference_idx = int(param_block.get("ReferenceChannel", 1)) - 1
        mask = np.arange(n_channels) != trig_idx
        data_ref[:, mask] = data_ref[:, mask] - data_ref[:, [reference_idx]]
        n_channels = min(int(param_block.get("NumberEEGChannels", n_channels)), n_channels)
    elif device == "unicorn":
        n_channels = min(8, n_channels)
    else:
        n_channels = data_ref.shape[1]

    return data_ref, n_channels, trig_idx if trig_idx < data_ref.shape[1] else None


def _append_synthetic_trigger(
    data: np.ndarray,
    time_axis: np.ndarray,
    trigger_time: float,
    period: float,
    params: MutableMapping[str, Any],
) -> Tuple[np.ndarray, int]:
    if trigger_time <= 0 or period <= 0:
        return data, data.shape[1] - 1
    trigger_signal = (np.mod(time_axis, period) < trigger_time).astype(float)
    augmented = np.hstack([data, trigger_signal[:, None]])
    params.setdefault("Parameters", {})["TriggerChannel"] = augmented.shape[1]
    return augmented, augmented.shape[1] - 1


def calc_psd_power_time(
    data: np.ndarray,
    window: np.ndarray,
    overlap_sec: float,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_arr = np.asarray(data, dtype=float)
    if data_arr.size == 0:
        raise ValueError("Input signal is empty for PSD calculation.")

    nperseg = len(window)
    if nperseg <= 0:
        raise ValueError("Window length must be positive for PSD calculation.")

    if data_arr.shape[-1] < nperseg:
        nperseg = data_arr.shape[-1]
        window = hann_window(nperseg, periodic=True)

    noverlap = max(0, min(nperseg - 1, int(round(overlap_sec * fs))))

    freq, time_axis, power = spectrogram(
        data_arr,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    amplitude = np.sqrt(np.maximum(power, 0.0))
    alpha_idx = np.where((freq >= 8.0) & (freq <= 13.0))[0]
    band_power_linear = amplitude[alpha_idx, :].sum(axis=0)
    band_power_db = 20.0 * np.log10(np.maximum(band_power_linear, EPS))
    return band_power_db, power, time_axis, freq, power


def plot_psd_time(time_axis: np.ndarray, band_power: np.ndarray, channel_number: int) -> None:
    LOGGER.debug("plot_psd_time called", extra={"channel": channel_number, "points": band_power.size})
    if time_axis.size < 2:
        return
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(time_axis[1:], band_power[1:], color="k")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (dB) [rel. to 1 uV/√Hz]")
    plt.title(f"Channel {channel_number}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _show_mpl(fig, f"alpha_psd_time_{channel_number}")


def plot_spectrogram(
    freq_axis: np.ndarray,
    power_spectrogram: np.ndarray,
    time_axis: np.ndarray,
    trigger_times: np.ndarray,
    channel_number: int,
) -> None:
    LOGGER.debug(
        "plot_spectrogram called",
        extra={"channel": channel_number, "freq_points": freq_axis.size, "time_points": time_axis.size},
    )
    freq_mask = (freq_axis >= 7.0) & (freq_axis <= 20.0)
    if not freq_mask.any():
        return
    import matplotlib.pyplot as plt
    fig = plt.figure()
    print("plot_spectrogram")
    plt.pcolormesh(
        time_axis[1:],
        freq_axis[freq_mask],
        power_spectrogram[freq_mask, 1:],
        shading="auto",
        cmap="summer",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Channel {channel_number}")
    cbar = plt.colorbar()
    cbar.set_label("PSD (uV^2/Hz)")
    for trig in trigger_times:
        plt.axvline(trig, color="w", linestyle="-", linewidth=1.2)
    plt.tight_layout()
    _show_mpl(fig, f"alpha_spectrogram_{channel_number}")


def trigger_timestamp(
    data: np.ndarray,
    trigger_channel: int,
    without_first_seconds: float,
    fs: float,
) -> np.ndarray:
    target_fs = 250.0
    trigger = data[:, trigger_channel]
    if fs > target_fs:
        num_samples = max(1, int(round(len(trigger) * target_fs / fs)))
        trigger = resample(trigger, num_samples)
        fs = target_fs
    samples_to_ignore = int(round(without_first_seconds * fs))
    diff_signal = np.diff(trigger, prepend=trigger[0])
    signal = np.square(np.abs(diff_signal))
    if signal.size == 0 or signal.max() == 0:
        return np.array([])
    signal[:samples_to_ignore] = 0
    signal = signal / np.max(signal)
    mask = signal < 0.05
    prev = np.concatenate(([False], mask[:-1]))
    transitions = prev & ~mask
    indices = np.nonzero(transitions)[0]
    trig_times = indices / fs
    if trig_times.size > 1:
        diffs = np.diff(trig_times)
        keep = np.ones_like(trig_times, dtype=bool)
        keep[1:] = diffs >= 2.0
        trig_times = trig_times[keep]
    return trig_times


def nearest_indices(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    if haystack.size == 0 or needles.size == 0:
        return np.array([], dtype=int)
    hay = haystack.ravel()
    ned = needles.ravel()
    idx = np.searchsorted(hay, ned)
    idx = np.clip(idx, 0, len(hay) - 1)
    mask = (idx > 0) & (
        (idx == len(hay) - 1) | ((ned - hay[idx - 1]) <= (hay[idx] - ned))
    )
    idx[mask] -= 1
    return idx.astype(int)


def split_alpha_segments(
    alpha_power: np.ndarray,
    indices: np.ndarray,
    start_closed: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if indices.size < 2:
        return np.array([]), np.array([])
    closed_values: List[np.ndarray] = []
    open_values: List[np.ndarray] = []
    for i in range(indices.size - 1):
        start_idx = indices[i]
        end_idx = indices[i + 1]
        if end_idx <= start_idx:
            continue
        segment = alpha_power[start_idx:end_idx]
        if start_closed:
            (closed_values if i % 2 == 0 else open_values).append(segment)
        else:
            (open_values if i % 2 == 0 else closed_values).append(segment)
    closed = np.concatenate(closed_values) if closed_values else np.array([])
    open_ = np.concatenate(open_values) if open_values else np.array([])
    return closed, open_


def resolve_channel_label(channels: Optional[Sequence[Any]], idx: int) -> str:
    if channels and idx < len(channels):
        entry = channels[idx]
        if isinstance(entry, dict):
            for key in ("Position", "label", "name"):
                if key in entry:
                    return str(entry[key])
        elif isinstance(entry, (list, tuple)) and entry:
            return str(entry[0])
        elif isinstance(entry, str):
            return entry
    return f"Ch{idx + 1}"


def alpha_snr_ram(
    num_channels: int,
    evaluation: MutableMapping[str, Any],
    trigger_times: np.ndarray,
    time_axis: np.ndarray,
    freq_axis: np.ndarray,
    psd_cube: np.ndarray,
    params: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    if num_channels == 0 or trigger_times.size == 0:
        evaluation["SNR"] = np.array([])
        evaluation["SNR_dB"] = np.array([])
        evaluation["R_AM"] = np.array([])
        evaluation["R_AM_dB"] = np.array([])
        return evaluation

    after_trig = 3 if str(params.get("Trigger", "")).lower() == "external" else 0
    before_trig = 1 if str(params.get("Trigger", "")).lower() == "external" else 0

    alpha_indices = (freq_axis >= 8.0) & (freq_axis <= 12.0)
    noise_indices = (freq_axis >= 5.0) & (freq_axis <= 35.0)
    noise_indices &= ~((freq_axis >= 7.0) & (freq_axis <= 13.0))

    mask_closed, mask_open = eye_state_masks(trigger_times, time_axis, after_trig, before_trig)

    snr = np.zeros(num_channels)
    snr_db = np.zeros(num_channels)
    r_am = np.zeros(num_channels)
    r_am_db = np.zeros(num_channels)
    debug_metrics: list[dict[str, float | int]] = []

    for ch in range(num_channels):
        channel_psd = psd_cube[ch, :, :]
        closed_alpha = channel_psd[alpha_indices][:, mask_closed]
        closed_noise = channel_psd[noise_indices][:, mask_closed]
        open_alpha = channel_psd[alpha_indices][:, mask_open]

        alpha_power = float(np.mean(closed_alpha)) if closed_alpha.size else np.nan
        noise_power = float(np.mean(closed_noise)) if closed_noise.size else np.nan
        alpha_power_open = float(np.mean(open_alpha)) if open_alpha.size else np.nan

        if (
            np.isfinite(alpha_power)
            and np.isfinite(noise_power)
            and np.isfinite(alpha_power_open)
            and alpha_power > 0
            and noise_power > 0
            and alpha_power_open > 0
        ):
            snr[ch] = alpha_power / noise_power
            r_am[ch] = alpha_power / alpha_power_open
            snr_db[ch] = 20 * np.log10(snr[ch])
            r_am_db[ch] = 20 * np.log10(r_am[ch])
        else:
            snr[ch] = np.nan
            r_am[ch] = np.nan
            snr_db[ch] = np.nan
            r_am_db[ch] = np.nan

        debug_metrics.append(
            {
                "channel": ch + 1,
                "alpha_closed": alpha_power,
                "alpha_open": alpha_power_open,
                "noise_closed": noise_power,
            }
        )

    evaluation["SNR"] = snr
    evaluation["SNR_dB"] = snr_db
    evaluation["R_AM"] = r_am
    evaluation["R_AM_dB"] = r_am_db
    evaluation["alpha_debug"] = {
        "mask_closed_samples": int(mask_closed.sum()),
        "mask_open_samples": int(mask_open.sum()),
        "mask_total": int(mask_closed.size),
        "channels": debug_metrics,
    }
    if np.any(np.isfinite(snr) & (snr > 1e6)):
        print("Alpha SNR diagnostic:", evaluation["alpha_debug"])
    return evaluation


def eye_state_masks(
    trigger_times: np.ndarray,
    time_axis: np.ndarray,
    after_trig: float,
    before_trig: float,
) -> Tuple[np.ndarray, np.ndarray]:
    trigger_times = np.asarray(trigger_times, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)
    if trigger_times.size == 0:
        empty = np.zeros_like(time_axis, dtype=bool)
        return empty, empty

    def _interval(start: float, end: float) -> np.ndarray:
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            return np.zeros_like(time_axis, dtype=bool)
        return (time_axis >= start) & (time_axis <= end)

    closed_mask = np.zeros_like(time_axis, dtype=bool)
    open_mask = np.zeros_like(time_axis, dtype=bool)
    num_trigger = trigger_times.size
    odd_or_even = num_trigger % 2

    if odd_or_even == 0:
        closed_indices = range(0, num_trigger - 1, 2)
        open_indices = range(1, num_trigger, 2)
    else:
        closed_indices = range(0, num_trigger, 2)
        open_indices = range(1, num_trigger - 1, 2)

    for trig in closed_indices:
        if trig == 0:
            start = after_trig
        else:
            start = trigger_times[trig - 1] + after_trig
        end_idx = trig
        if end_idx >= num_trigger:
            continue
        end = trigger_times[end_idx] - before_trig
        closed_mask |= _interval(start, end)

    for trig in open_indices:
        start = trigger_times[trig - 1] + after_trig
        end_idx = trig
        if end_idx >= num_trigger:
            continue
        end = trigger_times[end_idx] - before_trig
        open_mask |= _interval(start, end)

    return closed_mask, open_mask


def plot_alpha_matrix(
    evaluation: MutableMapping[str, Any],
    reference_channel: int,
    channel_labels: Sequence[str],
) -> None:
    LOGGER.debug("plot_alpha_matrix called", extra={"channels": len(channel_labels)})
    metrics_source = evaluation.get("R_AM")
    if metrics_source is None or len(metrics_source) == 0:
        return
    import matplotlib.pyplot as plt
    if len(channel_labels) != len(metrics_source):
        channel_labels = [f"Ch{i+1}" for i in range(len(metrics_source))]
    if evaluation.get("R_AM_dB") is None:
        return
    metrics = np.column_stack(
        [evaluation["R_AM"], evaluation["R_AM_dB"], evaluation["SNR"], evaluation["SNR_dB"]]
    )
    import json
    def to_python(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_python(v) for v in obj]
        return obj

    results = {
        "R_AM": evaluation["R_AM"],
        "R_AM_dB": evaluation["R_AM_dB"],
        "SNR": evaluation["SNR"],
        "SNR_dB": evaluation["SNR_dB"]
    }

    with open("results.json", "w") as f:
        json.dump(to_python(results), f, indent=4)
    fig, ax = plt.subplots()
    im = ax.imshow(metrics, aspect="auto", cmap="cool")
    norm = im.norm
    for row_idx in range(metrics.shape[0]):
        for col_idx in range(metrics.shape[1]):
            value = metrics[row_idx, col_idx]
            if np.isnan(value):
                label = "nan"
                text_color = "black"
            else:
                label = f"{value:.2f}"
                text_color = "white" if norm(value) > 0.6 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color=text_color, fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["R_AM", "R_AM_dB", "SNR", "SNR_dB"])
    ax.set_yticks(range(len(channel_labels)))
    ax.set_yticklabels(channel_labels)
    ref_idx = reference_channel - 1
    ref_label = channel_labels[ref_idx] if 0 <= ref_idx < len(channel_labels) else str(reference_channel)
    ax.set_title(f"Parameter matrix reference: {ref_label}")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Channel")
    fig.colorbar(im, ax=ax, label="Value")
    fig.tight_layout()
    _show_mpl(fig, "alpha_matrix")
