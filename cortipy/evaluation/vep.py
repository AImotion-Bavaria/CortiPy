"""Visual Evoked Potential evaluation pipeline."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional, Sequence, Tuple

import logging
import matplotlib
import numpy as np
import mne
from scipy import stats
from scipy.ndimage import uniform_filter1d

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
from cortipy.shared.filtering import filter_vep
from cortipy.shared.segmentation import seg_sig_fast
from cortipy.shared.signal import time_vector
from cortipy.shared.triggers import trigger_adc

LOGGER = logging.getLogger("cortipy.evaluation.vep")


def _show_mpl(fig: plt.Figure, key: str) -> None:
    if not _is_streamlit_runtime():
        # Non-streamlit: defer to a single blocking plt.show later.
        return
    placeholders = st.session_state.setdefault("_vep_eval_placeholders", {})
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

MAX_TIME = 0.51
HP_CUTOFF = 0.5
SNR_SIGNAL_WINDOW = (0.08, 0.125)  # seconds
SNR_NOISE_WINDOW = (0.46, 0.50)
RN_ANALYSIS_WINDOW = (0.020, 0.250)


class VepEvaluator(EvaluatorBase):
    """Port of the MATLAB EvalVEPmain workflow."""

    def __init__(self, show_plots: bool | None = None) -> None:
        self.show_plots = show_plots

    def evaluate(self, context) -> None:  # type: ignore[override]
        LOGGER.debug("VepEvaluator.evaluate invoked")
        params = context.params
        if str(params.get("Method", "")).lower() != "vep":
            LOGGER.debug("VepEvaluator skipped: method=%s", params.get("Method"))
            return

        data = params.get("data")
        if data is None:
            raise ValueError("VepEvaluator requires `params['data']` to be available.")

        param_block = params.setdefault("Parameters", {})
        param_block.setdefault("edge", "b")
        fs = float(param_block.get("fs", 0))
        if fs <= 0:
            raise ValueError("VEP evaluation requires Params.Parameters.fs to be set.")

        show_plots = self.show_plots if self.show_plots is not None else not params.get("ReportAnalyzer")
        referenced, trig_idx = _apply_reference(
            np.asarray(data, dtype=float),
            params.get("Device"),
            param_block,
        )
        referenced = _filter_non_trigger_channels(referenced, trig_idx, fs)
        triggered = trigger_adc(referenced, fs, trig_idx, MAX_TIME, edge=param_block.get("edge", "b"))
        segments = seg_sig_fast(triggered, fs, MAX_TIME, trig_idx)
        if segments.size == 0:
            params.setdefault("Evaluation", {})
            context.params = params
            return

        if 0 <= trig_idx < segments.shape[2]:
            segments = np.delete(segments, trig_idx, axis=2)

        average_signals = segments.mean(axis=0)
        average_signals = average_signals - np.mean(average_signals, axis=0, keepdims=True)

        evaluation = params.setdefault("Evaluation", {})
        evaluation["average_signals"] = {
            "voltage": average_signals,
            "voltageUnit": "Amplitude (uV)",
            "time": time_vector(average_signals, fs, unit="ms"),
            "timeUnit": "Time (ms)",
            "segments_signalSameUnit": segments,
        }

        peak_stats = vep_amplitude_latency(average_signals, fs)
        evaluation.update(peak_stats)
        evaluation["Amplitude"] = peak_stats["P100"]["peak_values"] - peak_stats["N135"]["peak_values"]
        evaluation["tRes"] = t_test_grand_average(average_signals, peak_stats["P100"]["peak_times"], fs)
        evaluation["SNR_time"] = vep_snr_general(average_signals, fs)
        snr_peak, rn_values = snr_peak_metrics(segments, average_signals, fs)
        evaluation["SNR_Peak"] = snr_peak
        evaluation["RN_micV"] = rn_values

        params["Evaluation"] = evaluation

        if show_plots:
            LOGGER.debug("Rendering VEP evaluation plots")
            _ensure_interactive_backend()
            plt.rcParams["figure.max_open_warning"] = 0
            plt.rcParams["figure.raise_window"] = True
            plot_vep(average_signals, params, peak_stats)
            plot_vep_matrix(evaluation, param_block.get("ReferenceChannel", 1), params.get("Channels"), average_signals)
            _plot_vep_all_channels(average_signals, evaluation["average_signals"]["time"], params, peak_stats)
            if not _is_streamlit_runtime():
                try:
                    fig_nums = plt.get_fignums()
                    LOGGER.info(
                        "VEP: displaying %d figures (backend=%s); close windows to continue",
                        len(fig_nums),
                        plt.get_backend(),
                    )
                    was_interactive = plt.isinteractive()
                    plt.ioff()
                    plt.show(block=True)
                    if was_interactive:
                        plt.ion()
                    LOGGER.info("VEP: figures closed by user")
                except Exception as exc:  # pragma: no cover
                    LOGGER.debug("plt.show failed", extra={"error": str(exc)})

        context.params = params


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _apply_reference(data: np.ndarray, device: Optional[str], params: MutableMapping[str, Any]) -> Tuple[np.ndarray, int]:
    array = np.array(data, dtype=float, copy=True)
    trigger_idx = int(params.get("TriggerChannel", array.shape[1])) - 1
    dev = (device or "").lower()

    if dev == "actichamp":
        reference_idx = int(params.get("ReferenceChannel", 1)) - 1
        mask = np.ones(array.shape[1], dtype=bool)
        if 0 <= trigger_idx < mask.size:
            mask[trigger_idx] = False
        array[:, mask] = array[:, mask] - array[:, [reference_idx]]
    elif dev == "unicorn":
        array = array[:, : min(array.shape[1], 8)]

    return array, trigger_idx


def _filter_non_trigger_channels(data: np.ndarray, trigger_idx: int, fs: float) -> np.ndarray:
    mask = np.ones(data.shape[1], dtype=bool)
    if 0 <= trigger_idx < mask.size:
        mask[trigger_idx] = False
    filtered = np.array(data, copy=True)
    if mask.any():
        filtered[:, mask] = filter_vep(filtered[:, mask], HP_CUTOFF, fs)
    return filtered


def vep_amplitude_latency(mean_data: np.ndarray, fs: float) -> Dict[str, Dict[str, np.ndarray]]:
    smoothed = uniform_filter1d(mean_data, size=28, axis=0, mode="nearest")
    derivative = np.diff(smoothed, axis=0, prepend=smoothed[0:1])
    d_smoothed = uniform_filter1d(derivative, size=16, axis=0, mode="nearest")

    t_ms = time_vector(mean_data, fs, unit="ms")
    num_channels = mean_data.shape[1]

    N75_values = np.full(num_channels, np.nan)
    N75_times = np.full(num_channels, np.nan)
    P100_values = np.full(num_channels, np.nan)
    P100_times = np.full(num_channels, np.nan)
    N135_values = np.full(num_channels, np.nan)
    N135_times = np.full(num_channels, np.nan)

    idx_peak_start = _first_index_greater(t_ms, 20)
    idx_peak_end = _last_index_less_equal(t_ms, 250)
    idx_deriv_start = _first_index_greater(t_ms, 60)
    idx_deriv_end = _last_index_less_equal(t_ms, 160)

    for ch in range(num_channels):
        d_segment = d_smoothed[idx_deriv_start : idx_deriv_end + 1, ch]
        if d_segment.size == 0:
            continue
        local_max_idx = idx_deriv_start + np.argmax(d_segment)
        local_min_idx = idx_deriv_start + np.argmin(d_segment)
        idx_1 = min(local_min_idx, local_max_idx)
        idx_2 = max(local_min_idx, local_max_idx)

        if idx_2 > idx_1:
            range_indices = np.arange(idx_1, idx_2 + 1)
            sign_changes = np.diff(np.sign(d_smoothed[range_indices, ch]))
            zero_idx = range_indices[np.where(sign_changes != 0)[0] + 1]
            if zero_idx.size:
                idx_p100 = int(np.round(np.mean(zero_idx)))
                P100_times[ch] = t_ms[idx_p100]
                P100_values[ch] = mean_data[idx_p100, ch]

        N75_times[ch] = _search_zero_crossing_backward(d_smoothed[:, ch], idx_1, idx_peak_start, t_ms)
        N135_times[ch] = _search_zero_crossing_forward(d_smoothed[:, ch], idx_2, idx_peak_end, t_ms)

        if np.isfinite(N75_times[ch]):
            idx_n75 = _nearest_index(t_ms, N75_times[ch])
            N75_values[ch] = mean_data[idx_n75, ch]
        if np.isfinite(N135_times[ch]):
            idx_n135 = _nearest_index(t_ms, N135_times[ch])
            N135_values[ch] = mean_data[idx_n135, ch]

    return {
        "N75": {"peak_values": N75_values, "peak_times": N75_times},
        "P100": {"peak_values": P100_values, "peak_times": P100_times},
        "N135": {"peak_values": N135_values, "peak_times": N135_times},
    }


def t_test_grand_average(grand_average: np.ndarray, peak_times_ms: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    peak_time = np.nan_to_num(peak_times_ms[0], nan=100.0)
    window = ((peak_time - 30.0) / 1000.0, (peak_time + 50.0) / 1000.0)
    time = np.arange(grand_average.shape[0]) / fs
    mask = (time >= max(0.0, window[0])) & (time <= min(MAX_TIME, window[1]))
    if not mask.any():
        mask[:] = True

    sample = grand_average[mask, :]
    t_stat, p_value = stats.ttest_1samp(sample, popmean=0.0, axis=0, nan_policy="omit")
    h = (p_value < 0.05).astype(int)
    return {"t": t_stat, "p": p_value, "h": h}


def vep_snr_general(avg_signal: np.ndarray, fs: float) -> np.ndarray:
    signal_idx = _slice_from_window(SNR_SIGNAL_WINDOW, fs, avg_signal.shape[0])
    noise_idx = _slice_from_window(SNR_NOISE_WINDOW, fs, avg_signal.shape[0])

    snr_values = np.zeros(avg_signal.shape[1])
    for ch in range(avg_signal.shape[1]):
        signal_window = avg_signal[signal_idx, ch]
        noise_window = avg_signal[noise_idx, ch]
        signal_power = np.sum((signal_window - signal_window.mean()) ** 2)
        noise_power = np.sum((noise_window - noise_window.mean()) ** 2)
        if noise_power > 0:
            snr_values[ch] = 20 * np.log10(signal_power / noise_power)
        else:
            snr_values[ch] = np.nan
    return snr_values


def snr_peak_metrics(segments_signal: np.ndarray, avg_signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    num_channels = segments_signal.shape[2]
    rn_values = np.zeros(num_channels)
    amp_rn = np.zeros(num_channels)

    for ch in range(num_channels):
        rn_values[ch], diff_wave = residual_noise_eclipse(segments_signal[:, :, ch], fs, RN_ANALYSIS_WINDOW)
        amp_rn[ch] = np.max(diff_wave) - np.min(diff_wave)

    analysis_idx = _slice_from_window(RN_ANALYSIS_WINDOW, fs, avg_signal.shape[0])
    amp_signal = np.max(avg_signal[analysis_idx, :], axis=0) - np.min(avg_signal[analysis_idx, :], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_peak = 20 * np.log10(amp_signal / amp_rn)
        snr_peak[~np.isfinite(snr_peak)] = np.nan
    return snr_peak, rn_values


def residual_noise_eclipse(sweeps: np.ndarray, fs: float, analysis_window: Tuple[float, float]) -> Tuple[float, np.ndarray]:
    if sweeps.ndim != 2:
        raise ValueError("sweeps must be 2-D (epochs x time).")
    idx = _slice_from_window(analysis_window, fs, sweeps.shape[1])
    odd = sweeps[0::2, :][:, idx]
    even = sweeps[1::2, :][:, idx]
    if odd.size == 0 or even.size == 0:
        avg_odd = sweeps[:, idx].mean(axis=0)
        avg_even = avg_odd
    else:
        avg_odd = odd.mean(axis=0)
        avg_even = even.mean(axis=0)
    diff_wave = (avg_odd - avg_even) / 2.0
    rn = float(np.std(diff_wave))
    return rn, diff_wave


def plot_vep(avg_signal: np.ndarray, params: dict, peaks: Dict[str, Dict[str, np.ndarray]]) -> None:
    param_block = params.get("Parameters", {})
    fs = float(param_block.get("fs", 1.0))
    t_ms = time_vector(avg_signal, fs, unit="ms")
    idx_max = np.argmin(np.abs(t_ms - MAX_TIME * 1000.0))
    channels = params.get("Channels")
    for ch in range(avg_signal.shape[1]):
        #LOGGER.debug("plot_vep called", extra={"channel": ch + 1, "samples": avg_signal.shape[0]})
        fig = plt.figure()
        plt.plot(t_ms[: idx_max or None], avg_signal[: idx_max or None, ch], color="b")
        for name in ("P100", "N75", "N135"):
            value = peaks[name]["peak_values"][ch]
            time_ms = peaks[name]["peak_times"][ch]
            if np.isfinite(value) and np.isfinite(time_ms):
                plt.plot(time_ms, value, "r*", ms=8)
                plt.text(time_ms, value, name, color="r")

        title = _channel_title(params, ch, channels)
        plt.title(title)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (uV)")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        _show_mpl(fig, f"vep_channel_{ch+1}")


def plot_vep_matrix(
    evaluation: MutableMapping[str, Any],
    reference_channel: int,
    channels: Optional[Sequence[Any]],
    mean_voltages: np.ndarray,
) -> None:
    LOGGER.debug("plot_vep_matrix called", extra={"channels": mean_voltages.shape[1]})
    channel_labels = _channel_labels(channels, mean_voltages.shape[1])
    amplitude = evaluation["P100"]["peak_values"] - evaluation["N135"]["peak_values"]
    matrix = np.column_stack(
        [
            amplitude,
            evaluation["P100"]["peak_times"] - 100,
            evaluation["RN_micV"],
            evaluation["SNR_time"],
            evaluation["SNR_Peak"],
            evaluation["tRes"]["h"] * 10,
        ]
    )

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect="auto", cmap="cool")
    norm = im.norm
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                label = "nan"
                text_color = "black"
            else:
                label = f"{value:.2f}"
                text_color = "white" if norm(value) > 0.6 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color=text_color, fontsize=8)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(["P100-N135", "P100 latency", "STD RN", "SNR time", "SNR peak", "t-test"])
    ax.set_yticks(range(len(channel_labels)))
    ax.set_yticklabels(channel_labels)
    ref_idx = reference_channel - 1
    ref_label = channel_labels[ref_idx] if 0 <= ref_idx < len(channel_labels) else str(reference_channel)
    ax.set_title(f"Parameter matrix; REF {ref_label}")
    fig.colorbar(im, ax=ax, label="Value")
    fig.tight_layout()
    _show_mpl(fig, "vep_matrix")

    fig2, ax2 = plt.subplots()
    amplitude_plot = np.nan_to_num(amplitude, nan=0.0)
    ax2.bar(channel_labels, amplitude_plot)
    ax2.set_ylabel("Amplitude (uV)")
    ax2.set_xlabel("Channel")
    ax2.set_title(f"Amplitudes; REF {ref_label}")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    _show_mpl(fig2, "vep_amplitudes")
    _plot_vep_topomaps(evaluation, amplitude, channel_labels)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _first_index_greater(t: np.ndarray, value: float) -> int:
    idx = np.where(t > value)[0]
    return int(idx[0]) if idx.size else 0


def _last_index_less_equal(t: np.ndarray, value: float) -> int:
    idx = np.where(t <= value)[0]
    return int(idx[-1]) if idx.size else t.size - 1


def _search_zero_crossing_backward(data: np.ndarray, start_idx: int, min_idx: int, t_axis: np.ndarray) -> float:
    for i in range(start_idx, min_idx, -1):
        if i - 1 < 0:
            break
        if data[i] * data[i - 1] < 0:
            return float(t_axis[i])
    return np.nan


def _search_zero_crossing_forward(data: np.ndarray, start_idx: int, max_idx: int, t_axis: np.ndarray) -> float:
    for i in range(start_idx, max_idx):
        if i + 1 >= data.size:
            break
        if data[i] * data[i + 1] < 0:
            return float(t_axis[i])
    return np.nan


def _nearest_index(t: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(t - value)))


def _slice_from_window(window: Tuple[float, float], fs: float, max_samples: int) -> slice:
    start = max(0, int(round(window[0] * fs)))
    end = min(max_samples, int(round(window[1] * fs)))
    if end <= start:
        end = min(max_samples, start + 1)
    return slice(start, end)


def _channel_title(params: dict, channel_idx: int, channels: Optional[Sequence[Any]]) -> str:
    device = params.get("Device", "")
    labels = _channel_labels(channels, channel_idx + 1)
    label = labels[channel_idx] if channel_idx < len(labels) else f"Channel {channel_idx + 1}"
    if device == "ActiCHamp":
        ref_idx = int(params.get("Parameters", {}).get("ReferenceChannel", 1)) - 1
        ref_label = labels[ref_idx] if 0 <= ref_idx < len(labels) else str(ref_idx + 1)
        return f"Channel {channel_idx + 1} / {label}; REF {ref_label}"
    if device == "UNICORN":
        return f"Channel {channel_idx + 1} / {label}"
    return f"VEP Channel {channel_idx + 1}"


def _channel_labels(channels: Optional[Sequence[Any]], count: int) -> list[str]:
    result = []
    if channels:
        for entry in channels:
            label = None
            if isinstance(entry, dict):
                label = entry.get("Position") or entry.get("label") or entry.get("name")
            elif isinstance(entry, (list, tuple)) and entry:
                label = entry[0]
            elif isinstance(entry, str):
                label = entry
            result.append(str(label) if label is not None else f"Ch{len(result)+1}")
            if len(result) >= count:
                break
    while len(result) < count:
        result.append(f"Ch{len(result)+1}")
    return result


def _ensure_interactive_backend() -> None:
    """Switch to an interactive backend when possible to display pop-up windows."""
    try:
        current = plt.get_backend()
    except Exception:
        return
    lower = str(current).lower()
    non_gui_backends = {"agg", "pdf", "pgf", "svg", "cairo", "template", "ps"}
    if lower not in non_gui_backends:
        return

    for candidate in ("TkAgg", "Qt5Agg", "MacOSX"):
        try:
            plt.switch_backend(candidate)
            LOGGER.info("VEP: switched matplotlib backend to %s", candidate)
            return
        except Exception:
            continue


def _plot_vep_topomaps(
    evaluation: MutableMapping[str, Any], amplitude: np.ndarray, channel_labels: Sequence[str]
) -> None:
    """Render head topographies for key peak metrics."""
    maps = [
        ("P100", np.asarray(evaluation["P100"]["peak_values"], dtype=float)),
        ("N75", np.asarray(evaluation["N75"]["peak_values"], dtype=float)),
        ("N135", np.asarray(evaluation["N135"]["peak_values"], dtype=float)),
        ("P100 - N135", np.asarray(amplitude, dtype=float)),
    ]
    info, kept_idx = _topomap_info(channel_labels)
    kept_idx = np.asarray(kept_idx, dtype=int)
    for name, values in maps:
        fig, ax = plt.subplots()
        data = values[kept_idx] if kept_idx.size and kept_idx.size == values.shape[0] else values
        im, _ = mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            contours=4,
            names=channel_labels,
            sphere=(0.0, -0.01, 0.0, 0.11),
            outlines="head",
        )
        ax.set_title(f"{name} topography")
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        fig.tight_layout()
        _show_mpl(fig, f"vep_topomap_{name.replace(' ', '_').lower()}")


def _topomap_info(channel_labels: Sequence[str]):
    """Build an MNE Info using only channels with known standard positions."""
    montage = None
    for candidate in ("standard_1005", "standard_1020"):
        try:
            montage = mne.channels.make_standard_montage(candidate)
            break
        except Exception:
            continue

    known = montage.get_positions().get("ch_pos", {}) if montage is not None else {}

    kept_idx: list[int] = []
    ch_pos: dict[str, tuple[float, float, float]] = {}
    for idx, label in enumerate(channel_labels):
        pos = known.get(label)
        if pos is None:
            continue
        ch_pos[label] = pos
        kept_idx.append(idx)

    # If nothing matched, fall back to a small circle with all channels.
    if not ch_pos:
        total = max(1, len(channel_labels))
        for idx, label in enumerate(channel_labels):
            angle = 2 * np.pi * idx / total + 0.1 * idx  # deterministic jitter
            radius = 0.06 + 0.01 * (idx % total) / max(1, total)
            ch_pos[label] = (radius * np.cos(angle), radius * np.sin(angle), 0.0)
            kept_idx.append(idx)

    info = mne.create_info(list(ch_pos.keys()), sfreq=1.0, ch_types="eeg")
    try:
        custom_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        info.set_montage(custom_montage)
    except Exception:
        pass
    if len(kept_idx) != len(channel_labels):
        LOGGER.info("VEP: topomap using %d/%d channels with known positions", len(kept_idx), len(channel_labels))
    return info, kept_idx


def _plot_vep_all_channels(avg_signal: np.ndarray, time_ms: np.ndarray, params: dict, peaks: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Plot all channel waveforms with the grand average highlighted."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for ch in range(avg_signal.shape[1]):
        ax.plot(time_ms, avg_signal[:, ch], color="gray", alpha=0.4, linewidth=1)
    grand_avg = np.nanmean(avg_signal, axis=1)
    ax.plot(time_ms, grand_avg, color="tab:blue", linewidth=2.5, label="Average")

    for name, color in (("P100", "red"), ("N75", "green"), ("N135", "purple")):
        values = peaks[name]["peak_values"]
        times = peaks[name]["peak_times"]
        if np.isfinite(values).any():
            avg_val = np.nanmean(values)
            avg_time = np.nanmean(times)
            if np.isfinite(avg_val) and np.isfinite(avg_time):
                ax.plot(avg_time, avg_val, marker="o", color=color, markersize=6, label=f"{name} avg")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title("VEP waveforms (all channels + average)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _show_mpl(fig, "vep_all_channels")
