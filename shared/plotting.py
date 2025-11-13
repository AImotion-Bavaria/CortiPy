"""Plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from cortipy.shared.signal import time_vector


_fft_fig_cache: Dict[str, plt.Figure] = {}
_vep_fig_cache: Dict[str, plt.Figure] = {}
_bera_fig_cache: Dict[str, plt.Figure] = {}


@dataclass
class ChannelInfo:
    label: str


def _resolve_channels(params: dict, count: int) -> Sequence[ChannelInfo]:
    channels = params.get("Channels")
    if isinstance(channels, Iterable):
        result = []
        for item in channels:
            label = ""
            if isinstance(item, dict):
                label = item.get("Position") or item.get("label") or ""
            elif isinstance(item, (list, tuple)) and item:
                label = str(item[0])
            else:
                label = str(item)
            result.append(ChannelInfo(label=label or f"CH {len(result)+1}"))
        if len(result) >= count:
            return result[:count]
    return [ChannelInfo(label=f"CH {idx+1}") for idx in range(count)]


def plot_fft_live(freq: np.ndarray, data: np.ndarray, ylabel: str, title: str, params: dict) -> None:
    """Reimplementation of the MATLAB `plotFFT_Live` helper."""
    if freq.ndim != 1:
        raise ValueError("`freq` must be a 1-D vector.")
    figure_key = str(params.get("Parameters", {}).get("Filename", "cortipy-live"))
    fig = _fft_fig_cache.get(figure_key)
    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure(num=figure_key, figsize=(14, 6))
        _fft_fig_cache[figure_key] = fig
    ax = fig.gca()
    ax.clear()

    params_block = params.get("Parameters", {})
    view_min = float(params_block.get("LowestFrequency", 0))
    view_max = float(params_block.get("HighestFrequency", freq.max()))
    view_max = max(view_min + 1.0, view_max)

    ix_low = np.argmin(np.abs(freq - view_min))
    ix_high = np.argmin(np.abs(freq - view_max))
    if ix_high <= ix_low:
        ix_low, ix_high = 0, len(freq) - 1

    eeg_bands = [
        ("Delta", (0.5, 4), "#e1e1e1"),
        ("Theta", (4, 8), "#ccccff"),
        ("Alpha", (8, 12), "#f5f5c5"),
        ("Beta", (12, 30), "#ffd6d6"),
        ("Gamma", (30, view_max), "#d6f5d6"),
    ]

    y_vals = data[ix_low:ix_high, :]
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    y_pad = 0.05 * (y_max - y_min)

    for name, (low, high), color in eeg_bands:
        x1 = max(view_min, low)
        x2 = min(view_max, high)
        if x1 >= x2:
            continue
        ax.fill_between([x1, x2], [y_min - y_pad] * 2, [y_max + y_pad] * 2, color=color, alpha=0.2)
        ax.text((x1 + x2) / 2.0, y_max, name, ha="center", va="bottom", fontsize=9, alpha=0.7)

    colors = [
        "red",
        "green",
        "blue",
        "magenta",
        "black",
        "cyan",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    device = params.get("Device", "")
    offset = 1 if device == "ActiCHamp" else 2
    channel_labels = _resolve_channels(params, data.shape[1])
    for idx in range(data.shape[1]):
        label_idx = idx + offset - 1
        label = (
            channel_labels[label_idx].label
            if 0 <= label_idx < len(channel_labels)
            else f"Channel {idx+1}"
        )
        color = colors[idx % len(colors)]
        ax.plot(freq[ix_low:ix_high], data[ix_low:ix_high, idx], color=color, lw=2, label=label)

    ax.set_xlim(view_min, view_max)
    ticks = np.arange(np.ceil(view_min / 5) * 5, np.floor(view_max / 5) * 5 + 1, 5)
    ticks = np.unique(np.concatenate(([view_min], ticks, [view_max])))
    ax.set_xticks(ticks)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw_idle()


def plot_live_avg_vep(avg_signal: np.ndarray, params: dict, max_time: float) -> None:
    """Live plot for the running VEP average of the selected channel."""
    param_block = params.get("Parameters", {})
    fs = float(param_block.get("fs", 1.0))
    live_channel = int(param_block.get("LivePlotCH", 1))
    figure_key = f"vep-{param_block.get('Filename', 'cortipy-live')}"

    fig = _vep_fig_cache.get(figure_key)
    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure(
            num=figure_key,
            figsize=(12, 5),
        )
        _vep_fig_cache[figure_key] = fig
    ax = fig.gca()
    ax.clear()

    t_ms = time_vector(avg_signal, fs, unit="ms")
    if t_ms.size == 0:
        return
    idx_max = np.argmin(np.abs(t_ms - max_time * 1000.0))
    ax.plot(t_ms[: idx_max or None], avg_signal[: idx_max or None], color="b")

    device = params.get("Device", "")
    channels = _resolve_channels(params, int(param_block.get("NumberEEGChannels", len(avg_signal))))
    if device == "ActiCHamp":
        ref_idx = int(param_block.get("ReferenceChannel", 1))
        ref_label = channels[ref_idx].label if 0 <= ref_idx < len(channels) else str(ref_idx)
        label_idx = live_channel
        label = channels[label_idx].label if 0 <= label_idx < len(channels) else f"Channel {live_channel}"
        title = f"Channel {live_channel} / {label}; REF {ref_label}"
    elif device == "UNICORN":
        label_idx = live_channel + 1
        label = channels[label_idx].label if 0 <= label_idx < len(channels) else f"Channel {live_channel}"
        title = f"Channel {live_channel} / {label}"
    else:
        title = f"VEP Channel {live_channel}"

    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw_idle()


def plot_live_avg_bera(
    avg_signal: np.ndarray,
    params: dict,
    max_time: float,
    live_plot_ch: int,
    elapsed_time: Sequence[float],
    rn_hist: Sequence[float],
    fmp_hist: Sequence[float],
) -> None:
    param_block = params.get("Parameters", {})
    fs = float(param_block.get("fs", 1.0))
    figure_key = f"bera-{param_block.get('Filename', 'cortipy-live')}-ch{live_plot_ch}"

    fig = _bera_fig_cache.get(figure_key)
    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure(num=figure_key, figsize=(12, 7))
        _bera_fig_cache[figure_key] = fig
    fig.clf()

    ax_wave = fig.add_axes([0.1, 0.45, 0.85, 0.5])
    ax_hist = fig.add_axes([0.1, 0.15, 0.85, 0.25])

    avg_signal = np.asarray(avg_signal, dtype=float)
    t_ms = time_vector(avg_signal, fs, unit="ms")
    if t_ms.size == 0:
        return
    idx_max = np.argmin(np.abs(t_ms - max_time * 1000.0))
    ax_wave.plot(t_ms[: idx_max or None], avg_signal[: idx_max or None], color="r")
    ax_wave.set_title(f"BERA Channel {live_plot_ch}")
    ax_wave.set_ylabel("Amplitude (nV)")
    ax_wave.set_xlabel("Time (ms)")
    ax_wave.set_xlim(0, max(15, max_time * 1000.0))
    ax_wave.set_ylim(-1000, 1000)
    ax_wave.grid(True, alpha=0.3)

    elapsed = np.asarray(elapsed_time, dtype=float)
    rn = np.asarray(rn_hist, dtype=float)
    fmp = np.asarray(fmp_hist, dtype=float)
    valid_rn = np.isfinite(rn)
    valid_fmp = np.isfinite(fmp)
    if valid_rn.any():
        ax_hist.plot(elapsed[valid_rn], rn[valid_rn], "-o", label="RN_eclipse", color="tab:blue")
        ax_hist.set_ylabel("RN_eclipse (nV)", color="tab:blue")
        ax_hist.tick_params(axis="y", labelcolor="tab:blue")
    ax_hist.set_xlabel("Elapsed Time (s)")
    ax_hist.grid(True, alpha=0.3)

    ax_hist2 = ax_hist.twinx()
    if valid_fmp.any():
        ax_hist2.plot(elapsed[valid_fmp], fmp[valid_fmp], "-x", label="Fmp", color="tab:red")
        ax_hist2.set_ylabel("Fmp", color="tab:red")
        ax_hist2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    fig.canvas.draw_idle()


def plot_bera_results(
    average_data: np.ndarray,
    params: dict,
    num_cycles: int,
    time_ms: np.ndarray,
) -> None:
    param_block = params.get("Parameters", {})
    evaluation = params.get("Evaluation", {})
    avg = np.asarray(average_data, dtype=float)
    if avg.ndim != 3:
        return
    t = np.asarray(time_ms, dtype=float)
    ipsi = int(param_block.get("ChannelIpsi", 1)) - 1
    contra = int(param_block.get("ChannelContra", ipsi + 1)) - 1
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx in range(avg.shape[0]):
        style = "-" if idx == 0 else "--"
        alpha = 1.0 if idx == 0 else 0.4
        width = 2.5 if idx == 0 else 1.0
        color = "tab:blue" if idx == 0 else "gray"
        ax.plot(t, avg[idx, :, max(min(ipsi, avg.shape[2] - 1), 0)], style, color=color, alpha=alpha, linewidth=width)
        if contra < avg.shape[2]:
            ax.plot(
                t,
                avg[idx, :, contra],
                "--",
                color="tab:orange",
                alpha=alpha,
                linewidth=width,
            )

    for center, half, label in [(1.69, 0.26, "Wave I"), (3.82, 0.32, "Wave III"), (5.60, 0.42, "Wave V")]:
        ax.axvspan(center - half, center + half, color="red", alpha=0.05)
        ax.text(center, ax.get_ylim()[1] * 0.9, label, ha="center", color="red")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (nV)")
    ax.set_xlim(0, 15)
    ax.set_ylim(-1000, 1000)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"BERA — {param_block.get('Filename', '')}")

    stats_lines = []
    if evaluation.get("Fsp") is not None and len(evaluation["Fsp"]) > 0:
        stats_lines.append(f"Fsp: {evaluation['Fsp'][0]:.2f} (p={evaluation['p_sp'][0]:.2f})")
    if evaluation.get("Fmp") is not None and len(evaluation["Fmp"]) > 0:
        stats_lines.append(f"Fmp: {evaluation['Fmp'][0]:.2f} (p={evaluation['p_mp'][0]:.2f})")
    if evaluation.get("RN_elberlingDon") is not None and len(evaluation["RN_elberlingDon"]) > 0:
        stats_lines.append(f"RN_E-D: {evaluation['RN_elberlingDon'][0]:.1f} nV")
    if evaluation.get("RN_eclipse") is not None and len(evaluation["RN_eclipse"]) > 0:
        stats_lines.append(f"RN_Eclipse: {evaluation['RN_eclipse'][0]:.1f} nV")
    stats_lines.append(f"Cycles: {num_cycles}")
    ax.text(
        0.98,
        0.02,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    fig.tight_layout()
    fig.canvas.draw_idle()


def plot_live_erp(avg_signal: np.ndarray, params: dict, max_time: float) -> None:
    """Live plotting for P300 ERP averages."""
    param_block = params.get("Parameters", {})
    fs = float(param_block.get("fs", 1.0))
    figure_key = f"p300-live-{param_block.get('Filename', 'cortipy-live')}"

    fig = _vep_fig_cache.get(figure_key)
    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure(num=figure_key, figsize=(10, 5))
        _vep_fig_cache[figure_key] = fig
    ax = fig.gca()
    ax.clear()

    t_ms = time_vector(avg_signal, fs, unit="ms")
    if t_ms.size == 0:
        return
    idx_max = np.argmin(np.abs(t_ms - max_time * 1000.0))
    ax.plot(t_ms[: idx_max or None], avg_signal[: idx_max or None], color="tab:blue")

    device = params.get("Device", "")
    live_ch = int(param_block.get("LivePlotCH", 1))
    channels = _resolve_channels(params, int(param_block.get("NumberEEGChannels", len(avg_signal))))
    if device == "ActiCHamp":
        ref_idx = int(param_block.get("ReferenceChannel", 1))
        ref_label = channels[ref_idx].label if 0 <= ref_idx < len(channels) else str(ref_idx)
        label_idx = live_ch
        label = channels[label_idx].label if 0 <= label_idx < len(channels) else f"Channel {live_ch}"
        title = f"Channel {live_ch} / {label}; REF {ref_label}"
    elif device == "UNICORN":
        label_idx = live_ch + 1
        label = channels[label_idx].label if 0 <= label_idx < len(channels) else f"Channel {live_ch}"
        title = f"Channel {live_ch} / {label}"
    else:
        title = f"P300 Channel {live_ch}"

    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw_idle()


def plot_p300_results(avg_signal: np.ndarray, params: dict, max_time: float) -> None:
    param_block = params.get("Parameters", {})
    fs = float(param_block.get("fs", 1.0))
    t_ms = time_vector(avg_signal, fs, unit="ms")
    idx_max = np.argmin(np.abs(t_ms - max_time * 1000.0)) if t_ms.size else 0
    channels = _resolve_channels(params, avg_signal.shape[1])
    device = params.get("Device", "")

    for ch in range(avg_signal.shape[1]):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_ms[: idx_max or None], avg_signal[: idx_max or None, ch], color="tab:blue")
        if device == "ActiCHamp":
            ref_idx = int(param_block.get("ReferenceChannel", 1))
            ref_label = channels[ref_idx].label if 0 <= ref_idx < len(channels) else str(ref_idx)
            label = channels[ch].label if ch < len(channels) else f"Channel {ch+1}"
            title = f"Channel {ch+1} / {label}; REF {ref_label}"
        elif device == "UNICORN":
            label_idx = ch + 1
            label = channels[label_idx].label if 0 <= label_idx < len(channels) else f"Channel {ch+1}"
            title = f"Channel {ch+1} / {label}"
        else:
            title = f"P300 Channel {ch+1}"
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (uV)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()


def plot_assr_spectrum(
    freq: np.ndarray,
    values: np.ndarray,
    view_min: float,
    view_max: float,
    ylabel: str,
    title: str,
) -> None:
    mask = (freq >= view_min) & (freq <= view_max)
    if not mask.any():
        mask = slice(None)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq[mask], values[mask], color="tab:blue", linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
