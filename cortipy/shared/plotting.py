"""Plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import matplotlib
import numpy as np
import mne

try:  # pragma: no cover - optional Streamlit support
    import streamlit as st

    _STREAMLIT_RUNTIME = bool(getattr(st, "runtime", None) and st.runtime.exists())
except Exception:  # pragma: no cover - Streamlit may not be installed
    st = None
    _STREAMLIT_RUNTIME = False

if _STREAMLIT_RUNTIME:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from cortipy.shared.signal import time_vector

__all__ = [
    "plot_fft_live",
    "plot_live_avg_vep",
    "plot_live_avg_bera",
    "plot_live_erp",
    "plot_p300_results",
    "plot_bera_results",
    "plot_cortipy_topomap",
    "plot_assr_spectrum",
    "apply_standard_montage",
    "topomap_info_from_labels",
    "_resolve_channels",
    "_topomap_info_from_labels",
]


_fft_fig_cache: Dict[str, plt.Figure] = {}
_vep_fig_cache: Dict[str, plt.Figure] = {}
_bera_fig_cache: Dict[str, plt.Figure] = {}


def _streamlit_placeholder(key: str):
    if not _STREAMLIT_RUNTIME or st is None:
        return None
    placeholders = st.session_state.setdefault("_mpl_placeholders", {})
    placeholder = placeholders.get(key)
    if placeholder is None:
        placeholder = st.empty()
        placeholders[key] = placeholder
    return placeholder


def _show_in_streamlit(fig: plt.Figure, key: str) -> None:
    placeholder = _streamlit_placeholder(key)
    if placeholder is not None:
        placeholder.pyplot(fig, clear_figure=False)


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


def _channel_plot_title(prefix: str, params: dict, channel_no: int, count: int) -> str:
    """Title a single-channel trace. ``channel_no`` is the 1-based EEG channel number.

    ``params["Channels"]`` is positional over the EEG columns, so this works for every
    device.  It used to be branched per device to skip the GND/Ref rows that were
    prepended to the montage; those no longer sit in the list.
    """
    param_block = params.get("Parameters", {}) or {}
    labels = [info.label for info in _resolve_channels(params, max(count, channel_no))]
    idx = channel_no - 1
    label = labels[idx] if 0 <= idx < len(labels) else f"Ch {channel_no}"
    title = f"{prefix} {channel_no} / {label}"

    try:
        ref_idx = int(param_block.get("ReferenceChannel")) - 1
    except (TypeError, ValueError):
        ref_idx = -1
    if 0 <= ref_idx < len(labels):
        title = f"{title}; REF {labels[ref_idx]}"
    return title


def topomap_info_from_labels(labels: Sequence[str], params: dict | None = None):
    """Public helper: best-effort Info + kept indices for scalp plots with position aliases."""
    return _topomap_info_from_labels(labels, params=params)


def _topomap_info_from_labels(labels: Sequence[str], params: dict | None = None):
    """Best-effort Info + kept indices for scalp plots."""
    montage = None
    known = {}
    for candidate in ("standard_1005", "standard_1020"):
        try:
            montage = mne.channels.make_standard_montage(candidate)
            known = montage.get_positions().get("ch_pos", {})
            if known:
                break
        except Exception:
            continue

    pos_map = {}
    if isinstance(params, dict) and isinstance(params.get("Channels"), Iterable):
        pos_map = {ch.get("Channel"): ch.get("Position") for ch in params["Channels"] if isinstance(ch, dict)}

    kept_idx: list[int] = []
    ch_pos: dict[str, tuple[float, float, float]] = {}
    for idx, label in enumerate(labels):
        target = pos_map.get(label, label) if pos_map else label
        pos = known.get(target)
        if pos is None:
            pos = known.get(label)
        if pos is None:
            continue
        ch_pos[label] = pos
        kept_idx.append(idx)

    if len(ch_pos) < len(labels):
        total = max(1, len(labels))
        for idx, label in enumerate(labels):
            if label in ch_pos:
                continue
            angle = 2 * np.pi * idx / total + 0.1 * idx
            base_radius = 0.045
            jitter = (abs(hash(label)) % 1000) / 1e6
            radius = base_radius + jitter
            ch_pos[label] = (radius * np.cos(angle), radius * np.sin(angle), 0.0)
            kept_idx.append(idx)

    info = mne.create_info(list(ch_pos.keys()), sfreq=1.0, ch_types="eeg")
    try:
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        info.set_montage(montage)
    except Exception:
        pass
    return info, kept_idx


def apply_standard_montage(raw: mne.io.BaseRaw, params: dict | None = None) -> None:
    """Assign a montage using channel labels/params; fallback to synthetic circle to avoid warnings."""
    info, _ = _topomap_info_from_labels(raw.ch_names, params=params)
    montage = getattr(info, "get_montage", lambda: None)()
    if montage is not None:
        try:
            raw.set_montage(montage, on_missing="ignore")
        except Exception:
            return


def plot_cortipy_topomap(
    values: np.ndarray,
    ch_names: Sequence[str] | None = None,
    *,
    params: dict | None = None,
    info: mne.Info | None = None,
    title: str | None = None,
    cbar_label: str | None = None,
    cmap: str = "RdBu_r",
    vlim: tuple[float, float] | None = None,
    contours: int = 6,
    show_names: bool = True,
    sphere: tuple[float, float, float, float] | None = (0.0, -0.01, 0.0, 0.105),
    figsize: tuple[float, float] = (7.0, 7.0),
    dpi: int = 200,
    colorbar: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Consistent CortiPy scalp plot helper used across evaluators."""
    data = np.asarray(values, dtype=float).squeeze()
    labels = list(ch_names) if ch_names is not None else []

    topo_info = info
    kept_idx_seq: Sequence[int] | None = None
    if topo_info is None:
        if not labels:
            raise ValueError("plot_cortipy_topomap requires `ch_names` when `info` is not provided.")
        topo_info, kept_idx_seq = _topomap_info_from_labels(labels, params=params)
    else:
        if not labels:
            labels = list(topo_info["ch_names"])
        if len(topo_info["ch_names"]) != len(labels):
            try:
                topo_info = topo_info.copy().pick_channels(labels, ordered=True)
            except Exception:
                labels = list(topo_info["ch_names"])

    kept_idx_arr = np.asarray(kept_idx_seq, dtype=int) if kept_idx_seq else None
    if kept_idx_arr is not None:
        if data.shape[0] >= len(labels):
            data = data[kept_idx_arr]
            labels = [labels[idx] for idx in kept_idx_arr if idx < len(labels)]
        elif data.shape[0] == len(kept_idx_arr):
            labels = [labels[idx] for idx in kept_idx_arr if idx < len(labels)]

    target_len = min(data.shape[0], len(labels), len(topo_info["ch_names"]))
    data = data[:target_len]
    labels = labels[:target_len]
    if len(topo_info["ch_names"]) != target_len:
        try:
            topo_info = topo_info.copy().pick_channels(labels, ordered=True)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_axis_off()
    try:
        im, _ = mne.viz.plot_topomap(
            data,
            topo_info,
            axes=ax,
            show=False,
            contours=contours,
            cmap=cmap,
            outlines="head",
            sphere=sphere,
            extrapolate="head",
            names=labels if show_names else None,
            show_names=show_names,
        )
    except TypeError:
        # Older MNE versions may not support some kwargs; fall back to minimal call.
        im, _ = mne.viz.plot_topomap(
            data,
            topo_info,
            axes=ax,
            show=False,
            contours=contours,
            cmap=cmap,
            outlines="head",
            sphere=sphere,
            extrapolate="head",
            names=None if show_names else None,
        )
        if show_names:
            try:
                from mne.channels.layout import _find_topomap_coords

                coords = _find_topomap_coords(topo_info, picks=range(len(labels)))
                for (x, y), label in zip(coords, labels):
                    ax.text(x, y, label, ha="center", va="center", fontsize=7)
            except Exception:
                pass
    if vlim is not None:
        im.set_clim(vmin=vlim[0], vmax=vlim[1])
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=12)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.canvas.draw_idle()
    return fig, ax


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

    # Channels is positional over the EEG columns, so data column i is channel i (like
    # _channel_plot_title). The old per-device offset was stale compensation for GND/Ref rows
    # that used to be prepended to the montage; it shifted every label by one.
    channel_labels = _resolve_channels(params, data.shape[1])
    for idx in range(data.shape[1]):
        label = (
            channel_labels[idx].label
            if 0 <= idx < len(channel_labels)
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
    _show_in_streamlit(fig, figure_key)


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

    count = int(param_block.get("NumberEEGChannels", len(avg_signal)))
    title = _channel_plot_title("Channel", params, live_channel, count)

    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw_idle()
    _show_in_streamlit(fig, figure_key)
    _show_in_streamlit(fig, figure_key)


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
    _show_in_streamlit(fig, figure_key)


def plot_bera_results(
    average_data: np.ndarray,
    params: dict,
    num_cycles: int,
    time_ms: np.ndarray,
) -> None:
    param_block = params.get("Parameters", {})
    evaluation = params.get("Evaluation", {})
    figure_key = f"bera-results-{param_block.get('Filename', 'cortipy')}"
    avg = np.asarray(average_data, dtype=float)
    if avg.ndim != 3:
        return
    t = np.asarray(time_ms, dtype=float)
    ipsi = int(param_block.get("ChannelIpsi", 1)) - 1
    contra = int(param_block.get("ChannelContra", ipsi + 1)) - 1
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False, gridspec_kw={"height_ratios": [2, 1]})
    full_ax, zoom_ax = axes

    def _plot_traces(ax, xlim: tuple[float, float]) -> None:
        ax.set_xlim(*xlim)
        for idx in range(avg.shape[0]):
            style = "-" if idx == 0 else "--"
            alpha = 1.0 if idx == 0 else 0.4
            width = 2.5 if idx == 0 else 1.0
            color = "tab:blue" if idx == 0 else "gray"
            ax.plot(t, avg[idx, :, max(min(ipsi, avg.shape[2] - 1), 0)], style, color=color, alpha=alpha, linewidth=width)
            if contra < avg.shape[2]:
                ax.plot(t, avg[idx, :, contra], "--", color="tab:orange", alpha=alpha, linewidth=width)
        for center, half, label in [(1.69, 0.26, "Wave I"), (3.82, 0.32, "Wave III"), (5.60, 0.42, "Wave V")]:
            ax.axvspan(center - half, center + half, color="red", alpha=0.05)
            ax.text(center, ax.get_ylim()[1] * 0.9, label, ha="center", color="red")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, alpha=0.3)

    _plot_traces(full_ax, (0, 15))
    # Auto-scale y around the plotted data with a small margin to emphasize the waveform.
    y_min, y_max = np.inf, -np.inf
    for line in full_ax.get_lines():
        if line.get_ydata().size:
            y_min = min(y_min, np.nanmin(line.get_ydata()))
            y_max = max(y_max, np.nanmax(line.get_ydata()))
    if y_min == np.inf or y_max == -np.inf:
        y_min, y_max = -5, 5
    span = y_max - y_min
    margin = 0.2 * span if span > 0 else 1.0
    full_ax.set_ylim(y_min - margin, y_max + margin)
    full_ax.set_title(f"BERA — {param_block.get('Filename', '')}")

    # Zoomed early-latency view to focus on Waves I–V.
    _plot_traces(zoom_ax, (0, 7))
    zoom_ax.set_ylim(full_ax.get_ylim())
    zoom_ax.set_xlabel("Time (ms)")

    stats_lines = []
    if evaluation.get("Fsp") is not None and len(evaluation["Fsp"]) > 0:
        stats_lines.append(f"Fsp: {evaluation['Fsp'][0]:.2f} (p={evaluation['p_sp'][0]:.2f})")
    if evaluation.get("Fmp") is not None and len(evaluation["Fmp"]) > 0:
        stats_lines.append(f"Fmp: {evaluation['Fmp'][0]:.2f} (p={evaluation['p_mp'][0]:.2f})")
    if evaluation.get("RN_elberlingDon") is not None and len(evaluation["RN_elberlingDon"]) > 0:
        stats_lines.append(f"RN_E-D: {evaluation['RN_elberlingDon'][0]:.3f} µV")
    if evaluation.get("RN_eclipse") is not None and len(evaluation["RN_eclipse"]) > 0:
        stats_lines.append(f"RN_Eclipse: {evaluation['RN_eclipse'][0]:.3f} µV")
    stats_lines.append(f"Cycles: {num_cycles}")
    zoom_ax.text(
        0.98,
        0.02,
        "\n".join(stats_lines),
        transform=zoom_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    fig.tight_layout()
    fig.canvas.draw_idle()
    _show_in_streamlit(fig, figure_key)


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

    live_ch = int(param_block.get("LivePlotCH", 1))
    channels = _resolve_channels(params, int(param_block.get("NumberEEGChannels", len(avg_signal))))
    title = _channel_plot_title("Channel", params, live_ch, len(channels))

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

    # Overview plot with all channels plus a grand average for quick quality inspection.
    fig_all, ax_all = plt.subplots(figsize=(10, 5))
    window_mask = slice(0, idx_max or None)
    ax_all.plot(t_ms[window_mask], avg_signal[window_mask], color="tab:gray", alpha=0.35, linewidth=1)
    grand_avg = avg_signal.mean(axis=1)
    ax_all.plot(t_ms[window_mask], grand_avg[window_mask], color="tab:blue", linewidth=2.5, label="Grand average")
    ax_all.axvspan(250, 500, color="orange", alpha=0.08, label="Typical P300 window")
    ax_all.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_all.set_title("P300 grand average and butterfly")
    ax_all.set_xlabel("Time (ms)")
    ax_all.set_ylabel("Amplitude (uV)")
    ax_all.grid(True, alpha=0.3)
    ax_all.legend(loc="upper right")
    fig_all.tight_layout()
    fig_all.canvas.draw_idle()
    _show_in_streamlit(fig_all, f"p300-results-{param_block.get('Filename', 'cortipy')}-overview")

    # Topomap of mean amplitude in the P300 window.
    _plot_p300_topomap(avg_signal, t_ms, channels, filename=param_block.get("Filename", "p300"))

    for ch in range(avg_signal.shape[1]):
        fig, ax = plt.subplots(figsize=(8, 4))
        figure_key = f"p300-results-{param_block.get('Filename', 'cortipy')}-ch{ch+1}"
        ax.plot(t_ms[: idx_max or None], avg_signal[: idx_max or None, ch], color="tab:blue")
        ax.set_title(_channel_plot_title("Channel", params, ch + 1, len(channels)))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (uV)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw_idle()
        _show_in_streamlit(fig, figure_key)


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
    fig.canvas.draw_idle()
    _show_in_streamlit(fig, f"assr-{title}")


def _plot_p300_topomap(avg_signal: np.ndarray, t_ms: np.ndarray, channels: Sequence[ChannelInfo], filename: str) -> None:
    if avg_signal.ndim != 2 or avg_signal.size == 0 or t_ms.size == 0:
        return
    window = (t_ms >= 250) & (t_ms <= 500)
    if not window.any():
        return
    data = np.nanmean(avg_signal[window], axis=0)
    labels = [ch.label for ch in channels]
    try:
        fig, _ = plot_cortipy_topomap(
            data,
            ch_names=labels,
            title="P300 mean amplitude (250-500 ms)",
            cbar_label="Amplitude (µV)",
            contours=6,
        )
        _show_in_streamlit(fig, f"p300-results-{filename}-topomap")
    except Exception:
        return
