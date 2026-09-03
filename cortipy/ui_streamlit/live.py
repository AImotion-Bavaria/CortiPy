"""Live preview and measurement plotting helpers for the Streamlit UI."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from cortipy.shared.filtering import filter_vep
from cortipy.shared.triggers import trigger_adc


try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None

from cortipy.devices import DeviceFactory, DeviceInterface
from cortipy.shared.channels import channel_labels
from cortipy.shared.reference import apply_eeg_reference, eeg_channel_count
from cortipy.ui_streamlit.fields import coerce_number
from cortipy.ui_streamlit.plot_windows import (
    open_window_once as _open_plot_window_once,
    safe_window_key as _safe_window_key,
    write_matplotlib_window as _write_matplotlib_window,
    write_plotly_window as _write_plotly_window,
)

LOGGER = logging.getLogger(__name__)


def _eeg_view(buffer: np.ndarray, params: Optional[Dict[str, Any]]) -> tuple[np.ndarray, List[str]]:
    """Referenced, EEG-only view of a raw device buffer, with a label per column.

    The raw buffer trails non-EEG columns — UNICORN sends accelerometer, gyroscope,
    battery and packet counter; ActiCHamp appends AUX and a trigger.  Drawing those on a
    microvolt axis is what made the live preview unreadable, and the live view was the one
    place that never applied the reference.
    """
    array = np.asarray(buffer, dtype=float)
    if array.ndim != 2 or array.size == 0:
        return array, []
    if not isinstance(params, dict):
        return array, channel_labels(None, array.shape[1])

    param_block = params.get("Parameters", {}) or {}
    referenced = apply_eeg_reference(array, param_block)
    count = eeg_channel_count(param_block, referenced.shape[1])
    count = max(1, min(int(count), referenced.shape[1]))
    return referenced[:, :count], channel_labels(params, count)


# The rolling window is fixed: an operator watching the trace wants a steady, comparable
# view, and a slider that changes it mid-run only makes traces incomparable between runs.
LIVE_WINDOW_SECONDS = 10.0

# How often the pop-out window reloads itself while streaming.
LIVE_REFRESH_SECONDS = 0.75

# VEP averages improve as more complete epochs arrive, so refresh this plot at the
# same cadence as the acquisition module rather than on every small display chunk.
VEP_REFRESH_SECONDS = 5.0

# Display-only limit. Acquisition, VEP processing, evaluation, and saving retain fs.
LIVE_DISPLAY_MAX_HZ = 250.0

# Y-axis scaling. "Auto" fits each channel to its own data; the fixed steps let you compare
# channels (and runs) on identical axes, which auto-scaling actively prevents.
LIVE_SCALE_OPTIONS: Dict[str, Optional[float]] = {
    "Auto": None,
    "± 25 µV": 25.0,
    "± 50 µV": 50.0,
    "± 100 µV": 100.0,
    "± 250 µV": 250.0,
    "± 500 µV": 500.0,
    "± 1000 µV": 1000.0,
}
DEFAULT_LIVE_SCALE = "Auto"

# Which live plot to draw. Available for every method.
PLOT_STACKED = "Stacked per-channel"
PLOT_OVERLAID = "Overlaid signal"
PLOT_FFT = "FFT / spectrum"
PLOT_SINGLE = "Single channel"
PLOT_VEP_AVERAGE = "VEP average"



LIVE_PLOT_TYPES = (
    PLOT_STACKED,
    PLOT_OVERLAID,
    PLOT_FFT,
    PLOT_SINGLE,
)
DEFAULT_LIVE_PLOT = PLOT_STACKED

def resolve_live_scale(label: Optional[str]) -> Optional[float]:
    """µV half-range for a scaling choice; None means auto-fit."""
    return LIVE_SCALE_OPTIONS.get(str(label or DEFAULT_LIVE_SCALE), None)


def _ch_label(labels: List[str], idx: int) -> str:
    return labels[idx] if 0 <= idx < len(labels) else f"Ch {idx + 1}"


def _channel_limits(values: np.ndarray, scale: Optional[float]) -> tuple[float, float]:
    """Y limits for one lane: the fixed range, or a padded fit around the data."""
    if scale and scale > 0:
        return -float(scale), float(scale)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.12 * (hi - lo)
    return lo - pad, hi + pad


def _decimate_for_display(
    buffer: np.ndarray,
    time_axis: np.ndarray,
    fs: float,
    max_rate: float = LIVE_DISPLAY_MAX_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce plotted points without changing the acquisition or analysis buffers."""
    if fs <= max_rate or fs <= 0 or buffer.shape[0] <= 1:
        return buffer, time_axis
    stride = max(1, int(math.ceil(fs / max_rate)))
    return buffer[::stride], time_axis[::stride]


def _stacked_channel_figure(
    buffer: np.ndarray,
    time_axis: np.ndarray,
    xlim: tuple[float, float],
    indices: List[int],
    ch_labels: List[str],
    *,
    fs: float,
    scale: Optional[float] = None,
) -> plt.Figure:
    """One lane per channel with its name and min/max on the left."""
    n = max(1, len(indices))
    fig, axes = plt.subplots(
        n, 1, sharex=True, figsize=(11.0, max(2.4, 0.85 * n + 0.9)), squeeze=False
    )
    axes = [row[0] for row in axes]
    colors = plt.cm.tab10.colors

    for lane, ch in enumerate(indices):
        ax = axes[lane]
        values = buffer[:, ch]
        finite = values[np.isfinite(values)]
        lo = float(np.min(finite)) if finite.size else 0.0
        hi = float(np.max(finite)) if finite.size else 0.0

        ax.plot(time_axis, values, color=colors[lane % len(colors)], linewidth=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*_channel_limits(values, scale))
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(axis="y", labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        # Name and measured range, down the left-hand margin.
        ax.text(
            -0.085, 0.5, _ch_label(ch_labels, ch),
            transform=ax.transAxes, ha="right", va="center",
            fontsize=9, fontweight="bold",
        )
        ax.text(
            -0.085, 0.08, f"min {lo:+.1f}\nmax {hi:+.1f} µV",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#5b6770", linespacing=1.3,
        )

    axes[-1].set_xlabel("Time (s) — newest on the right" if fs > 0 else "Samples")
    scale_note = "auto scale" if not scale else f"± {scale:g} µV"
    axes[0].set_title(
        f"Live preview — {LIVE_WINDOW_SECONDS:g} s window, {scale_note}",
        fontsize=10, loc="left",
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.93, bottom=0.12, hspace=0.25)
    return fig


def _overlaid_channel_figure(
    buffer: np.ndarray,
    time_axis: np.ndarray,
    xlim: tuple[float, float],
    indices: List[int],
    ch_labels: List[str],
    *,
    fs: float,
    scale: Optional[float] = None,
) -> plt.Figure:
    """All selected channels overlaid on one time axis."""
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    colors = plt.cm.tab10.colors
    for lane, ch in enumerate(indices):
        ax.plot(time_axis, buffer[:, ch], color=colors[lane % len(colors)],
                linewidth=0.8, label=_ch_label(ch_labels, ch))
    ax.set_xlim(*xlim)
    if scale and scale > 0:
        ax.set_ylim(-float(scale), float(scale))
    ax.set_xlabel("Time (s) — newest on the right" if fs > 0 else "Samples")
    ax.set_ylabel("Amplitude (µV)")
    ax.grid(True, alpha=0.25)
    if len(indices) <= 12:
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    scale_note = "auto scale" if not scale else f"± {scale:g} µV"
    ax.set_title(f"Live preview — {LIVE_WINDOW_SECONDS:g} s window, {scale_note}", fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def _fft_figure(
    buffer: np.ndarray,
    indices: List[int],
    ch_labels: List[str],
    *,
    fs: float,
) -> plt.Figure:
    """Amplitude spectrum of the selected channels, up to Nyquist."""
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    colors = plt.cm.tab10.colors
    n = buffer.shape[0]
    if n > 1 and fs > 0:
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        for lane, ch in enumerate(indices):
            mag = np.abs(np.fft.rfft(buffer[:, ch])) / n
            if mag.size > 2:
                mag[1:-1] *= 2
            ax.plot(freqs, mag, color=colors[lane % len(colors)],
                    linewidth=0.8, label=_ch_label(ch_labels, ch))
        ax.set_xlim(0, fs / 2.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (µV)")
    ax.grid(True, alpha=0.25)
    if len(indices) <= 12:
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_title("Live spectrum", fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def _single_channel_figure(
    buffer: np.ndarray,
    time_axis: np.ndarray,
    xlim: tuple[float, float],
    indices: List[int],
    ch_labels: List[str],
    *,
    fs: float,
    scale: Optional[float] = None,
) -> plt.Figure:
    """One large plot of the first selected channel, with its min/max."""
    ch = indices[0] if indices else 0
    values = buffer[:, ch]
    finite = values[np.isfinite(values)]
    lo = float(np.min(finite)) if finite.size else 0.0
    hi = float(np.max(finite)) if finite.size else 0.0
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    ax.plot(time_axis, values, color=plt.cm.tab10.colors[0], linewidth=0.9)
    ax.set_xlim(*xlim)
    ax.set_ylim(*_channel_limits(values, scale))
    ax.set_xlabel("Time (s) — newest on the right" if fs > 0 else "Samples")
    ax.set_ylabel("Amplitude (µV)")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{_ch_label(ch_labels, ch)}   (min {lo:+.1f} / max {hi:+.1f} µV)", fontsize=11, loc="left")
    fig.tight_layout()
    return fig



def _vep_average_figure(
    average_signal: Optional[np.ndarray] = None,
    fs: float = 0.0,
    channel_label: str = "VEP channel",
    status: Optional[str] = None,
) -> plt.Figure:
    """Build the live VEP average figure, including an optional averaged epoch."""
    fig, ax = plt.subplots(figsize=(11.0, 4.5))

    if average_signal is not None and average_signal.size and fs > 0:
        time_ms = np.arange(average_signal.size, dtype=float) / fs * 1000.0
        ax.plot(time_ms, average_signal, color="tab:blue", linewidth=1.2)
        ax.set_title(f"Live VEP average - {channel_label}")
        if status:
            ax.text(
                0.01, 0.96, status, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#5b6770",
            )
    else:
        ax.set_title("Live VEP average")
        if status:
            ax.text(0.5, 0.5, status, transform=ax.transAxes, ha="center", va="center", color="#5b6770")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_xlim(0, 510)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def _live_vep_average(
    buffer: np.ndarray,
    fs: float,
    params: Optional[Dict[str, Any]],
) -> tuple[Optional[np.ndarray], str, str]:
    """Reference, edge-detect, segment, and average the live VEP epochs."""
    if buffer.ndim != 2 or buffer.shape[0] < 4 or fs <= 0 or not isinstance(params, dict):
        return None, "VEP channel", "Waiting for live VEP data"

    param_block = params.get("Parameters", {}) or {}
    max_time = float(param_block.get("VEPMaxTime", 0.35) or 0.35)
    if max_time <= 0:
        max_time = 0.35
    trigger_idx = int(param_block.get("TriggerChannel", buffer.shape[1])) - 1
    if trigger_idx < 0 or trigger_idx >= buffer.shape[1]:
        return None, "VEP channel", "Trigger channel is not available"

    referenced = apply_eeg_reference(np.asarray(buffer, dtype=float), param_block)
    eeg_count = max(1, min(eeg_channel_count(param_block, referenced.shape[1]), referenced.shape[1]))
    filter_mask = np.zeros(referenced.shape[1], dtype=bool)
    filter_mask[:eeg_count] = True
    filter_mask[trigger_idx] = False
    try:
        if filter_mask.any():
            referenced[:, filter_mask] = filter_vep(referenced[:, filter_mask], 0.5, fs)
        triggered = trigger_adc(
            referenced,
            fs,
            trigger_idx,
            max_time,
            edge=str(param_block.get("edge", "b")),
        )
        trigger_positions = np.flatnonzero(triggered[:, trigger_idx] > 0.5)
    except (IndexError, ValueError, TypeError):
        return None, "VEP channel", "Unable to process the VEP trigger channel"

    epoch_samples = max(1, int(round(max_time * fs)))
    epochs = [
        referenced[position : position + epoch_samples]
        for position in trigger_positions
        if position + epoch_samples <= referenced.shape[0]
    ]
    elapsed_seconds = buffer.shape[0] / float(fs)
    trigger_rate = trigger_positions.size / elapsed_seconds if elapsed_seconds > 0 else 0.0
    if not epochs:
        return None, "VEP channel", (
            f"Triggers: {trigger_positions.size} total ({trigger_rate:.2f} Hz); "
            "complete epochs: 0"
        )
    segments = np.stack(epochs, axis=0)

    live_channel = max(1, int(param_block.get("LivePlotCH", 1))) - 1
    if live_channel < 0 or live_channel >= eeg_count or live_channel >= segments.shape[2]:
        live_channel = 0
    average_signal = segments[:, :, live_channel].mean(axis=0)
    average_signal -= np.mean(average_signal)
    labels = channel_labels(params, eeg_count)
    label_index = live_channel
    if trigger_idx < label_index:
        label_index -= 1
    label_index = max(0, min(label_index, len(labels) - 1)) if labels else 0
    return average_signal, _ch_label(labels, label_index), (
        f"Triggers: {trigger_positions.size} total ({trigger_rate:.2f} Hz); "
        f"averaged epochs: {segments.shape[0]}"
    )


def build_live_figure(
    plot_type: str,
    buffer: np.ndarray,
    time_axis: np.ndarray,
    xlim: tuple[float, float],
    indices: List[int],
    ch_labels: List[str],
    *,
    fs: float,
    scale: Optional[float] = None,
) -> plt.Figure:
    """Dispatch to the requested live plot. Every type works for every method."""
    if plot_type == PLOT_OVERLAID:
        return _overlaid_channel_figure(buffer, time_axis, xlim, indices, ch_labels, fs=fs, scale=scale)
    if plot_type == PLOT_FFT:
        return _fft_figure(buffer, indices, ch_labels, fs=fs)
    if plot_type == PLOT_SINGLE:
        return _single_channel_figure(buffer, time_axis, xlim, indices, ch_labels, fs=fs, scale=scale)
    return _stacked_channel_figure(buffer, time_axis, xlim, indices, ch_labels, fs=fs, scale=scale)


def _label_part(labels: List[str], indices: List[int], total: int) -> str:
    if len(indices) == total:
        return "All channels"
    return ", ".join(_ch_label(labels, idx) for idx in indices)


def _resolve_aux_channels(params: Dict[str, Any]) -> int:
    if params.get("Device") == "ActiCHamp":
        return int(params.get("Parameters", {}).get("NumberAUXChannels", 0) or 0)
    return 0


def _selected_recording_seconds(params: Dict[str, Any]) -> Optional[float]:
    parameters = params.get("Parameters", {}) if isinstance(params, dict) else {}
    for key in ("RecordingTime", "recording_time", "Duration", "duration"):
        value = coerce_number(parameters.get(key))
        if value is not None and value > 0:
            return float(value)
    return None


def _plot_window_status(placeholder: Optional["st.delta_generator.DeltaGenerator"], title: str, path: Optional[Path]) -> None:
    if placeholder is None:
        return
    try:
        suffix = f" ({path.name})" if path is not None else ""
        placeholder.caption(f"{title} is in a plot window{suffix}.")
    except Exception:
        return


def _reset_plot_window_open_state(*keys: str) -> None:
    for key in keys:
        st.session_state.pop(f"_plot_window_opened_{_safe_window_key(key)}", None)


def _plot_live_buffer(
    buffer: np.ndarray,
    fs: float,
    placeholder: "st.delta_generator.DeltaGenerator",
    channel_indices: Optional[List[int]] = None,
    interactive: bool = False,
    window_seconds: Optional[float] = None,
    sample_offset: int = 0,
    params: Optional[Dict[str, Any]] = None,
    scale: Optional[float] = None,
    plot_type: str = DEFAULT_LIVE_PLOT,
    render_inline: bool = True,
    vep_placeholder=None,
    vep_buffer: Optional[np.ndarray] = None,
    render_vep: bool = True,
) -> None:
    LOGGER.info("Selected live plot: %s", plot_type)
    if buffer.size == 0:
        return

    
    raw_buffer = buffer
    buffer, ch_labels = _eeg_view(buffer, params)
    if buffer.size == 0:
        return

    samples = buffer.shape[0]
    offset = max(0, int(sample_offset))
    time_axis = (offset + np.arange(samples)) / fs if fs > 0 else offset + np.arange(samples)
    if fs > 0:
        time_axis = np.round(time_axis, 3)  # positive elapsed seconds from the start of the buffer
    if window_seconds is not None and fs > 0:
        # Show elapsed time with "now" on the right, clipped to the most recent `window` seconds.
        # No padding: early on the view just spans the data captured so far.
        time_axis_max = float(time_axis[-1])
        time_axis_min = max(0.0, time_axis_max - float(window_seconds))
    else:
        time_axis_min = time_axis[0]
        time_axis_max = time_axis[-1]
    total_channels = buffer.shape[1]
    indices = _normalize_channel_indices(channel_indices, total_channels)
    if str((params or {}).get("Method", "")).lower() == "vep":
        param_block = (params or {}).get("Parameters", {}) or {}
        live_channel = int(param_block.get("LivePlotCH", 1) or 1) - 1
        if live_channel < 0 or live_channel >= total_channels:
            live_channel = indices[0] if indices else 0
        indices = [live_channel]
        trigger_idx = 32 if str((params or {}).get("Device", "")).lower() == "actichamp" else int(param_block.get("TriggerChannel", raw_buffer.shape[1])) - 1
        if 0 <= trigger_idx < raw_buffer.shape[1]:
            # Keep the trigger raw and append it only to the display view. It must not be
            # software-referenced or included in the EEG channel average below.
            buffer = np.column_stack([buffer, raw_buffer[:, trigger_idx]])
            ch_labels = [*ch_labels, "Trigger"]
            indices = [*indices, buffer.shape[1] - 1]
            total_channels = buffer.shape[1]
    display_buffer, display_time_axis = _decimate_for_display(buffer, time_axis, fs)
    if interactive and go is not None:
        palette = list(getattr(plt.cm, "tab10").colors) if hasattr(plt.cm, "tab10") else []
        traces = []
        for plot_idx, ch in enumerate(indices):
            color = palette[plot_idx % len(palette)] if palette else (0.2, 0.4, 0.8)
            rgb = tuple(int(max(0, min(255, round(float(val) * 255)))) for val in (list(color) + [0, 0, 0])[:3])
            traces.append(
                go.Scatter(
                    x=display_time_axis,
                    y=display_buffer[:, ch],
                    mode="lines",
                    line=dict(color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", width=1.5),
                    name=_ch_label(ch_labels, ch),
                )
            )
        label_part = _label_part(ch_labels, indices, total_channels)
        layout = go.Layout(
            height=320,
            margin=dict(l=50, r=10, t=40, b=50),
            xaxis=dict(
                title="Time (s)" if fs > 0 else "Samples",
                range=[time_axis_min, time_axis_max],
            ),
            yaxis=dict(title="Amplitude (uV)"),
            title=f"Live preview - {label_part}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
        )
        title = f"Live preview - {label_part}"
        fig = go.Figure(data=traces, layout=layout)
        path = _write_plotly_window(
            fig,
            title,
            "live_preview_signal",
            auto_refresh=not interactive,
            refresh_seconds=0.75,
        )
        _open_plot_window_once(path, "live_preview_signal")
        _plot_window_status(placeholder, title, path)
    else:
        # Draw the plot the operator chose (stacked / overlaid / FFT / single channel).
        fig = build_live_figure(
            plot_type,
            display_buffer,
            display_time_axis,
            (time_axis_min, time_axis_max),
            indices,
            ch_labels,
            fs=fs,
            scale=scale,
        )
        title = f"Live preview - {plot_type} - {_label_part(ch_labels, indices, total_channels)}"
        if render_inline and placeholder is not None:
            # Show it right in the page, so it does not depend on a pop-out window opening.
            try:
                placeholder.pyplot(fig, clear_figure=False)
            except Exception:
                # a caption at least tells the operator the stream is alive
                _plot_window_status(placeholder, title, None)
        else:
            # auto_refresh is the whole point of a streaming window: the file underneath is
            # rewritten on every chunk, but without this the open tab never reloads it.
            path = _write_matplotlib_window(
                fig, title, "live_preview_signal",
                auto_refresh=True,
                refresh_seconds=LIVE_REFRESH_SECONDS,
            )
            _open_plot_window_once(path, "live_preview_signal")
            _plot_window_status(placeholder, title, path)
        plt.close(fig)

    if (
        str((params or {}).get("Method", "")).lower() == "vep"
        and vep_placeholder is not None
        and render_vep
    ):
        average_signal, average_label, status = _live_vep_average(
            raw_buffer if vep_buffer is None else vep_buffer, fs, params
        )
        fig = _vep_average_figure(average_signal, fs, average_label, status)
        vep_placeholder.pyplot(fig, clear_figure=False)
        plt.close(fig)


def _plot_fft_spectrum(
    buffer: np.ndarray,
    fs: float,
    placeholder: Optional["st.delta_generator.DeltaGenerator"],
    channel_indices: Optional[List[int]] = None,
    interactive: bool = False,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    if placeholder is None or buffer.size == 0 or fs <= 0:
        return
    buffer, ch_labels = _eeg_view(buffer, params)
    if buffer.size == 0:
        return
    total_channels = buffer.shape[1]
    indices = _normalize_channel_indices(channel_indices, total_channels)
    bands = [
        ("Delta", 0.5, 4.0, "#6C91FF"),
        ("Theta", 4.0, 8.0, "#7ED957"),
        ("Alpha", 8.0, 13.0, "#FFB347"),
        ("Beta", 13.0, 30.0, "#FF6F61"),
        ("Gamma", 30.0, 50.0, "#9B59B6"),
    ]
    if interactive and go is not None:
        palette = list(getattr(plt.cm, "tab10").colors) if hasattr(plt.cm, "tab10") else []
        traces = []
        shapes = []
        max_freq = 60.0
        for plot_idx, ch in enumerate(indices):
            data = buffer[:, ch]
            if data.size < 4:
                continue
            detrended = data - np.mean(data)
            window = np.hanning(detrended.size)
            windowed = detrended * window
            spectrum = np.fft.rfft(windowed)
            freq = np.fft.rfftfreq(detrended.size, d=1.0 / fs)
            power = (np.abs(spectrum) ** 2) / (np.sum(window**2) * fs)
            power_db = 10 * np.log10(power + 1e-12)
            freq_mask = freq <= max_freq
            color = palette[plot_idx % len(palette)] if palette else (0.2, 0.4, 0.8)
            rgb = tuple(int(max(0, min(255, round(float(val) * 255)))) for val in (list(color) + [0, 0, 0])[:3])
            traces.append(
                go.Scatter(
                    x=freq[freq_mask],
                    y=power_db[freq_mask],
                    mode="lines",
                    line=dict(color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", width=1.4),
                    name=_ch_label(ch_labels, ch),
                )
            )
        for name, low, high, band_color in bands:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=low,
                    x1=high,
                    y0=0,
                    y1=1,
                    fillcolor=band_color,
                    opacity=0.1,
                    layer="below",
                    line=dict(width=0),
                )
            )
        label_part = "All channels" if len(indices) == total_channels else ", ".join(f"{ch + 1}" for ch in indices)
        layout = go.Layout(
            height=320,
            margin=dict(l=50, r=10, t=40, b=50),
            xaxis=dict(title="Frequency (Hz)", range=[0, max_freq]),
            yaxis=dict(title="Power (dB/Hz)"),
            title=f"FFT - {label_part}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
            shapes=shapes,
        )
        title = f"FFT - {label_part}"
        fig = go.Figure(data=traces, layout=layout)
        path = _write_plotly_window(
            fig,
            title,
            "live_preview_fft",
            auto_refresh=not interactive,
            refresh_seconds=0.75,
        )
        _open_plot_window_once(path, "live_preview_fft")
        _plot_window_status(placeholder, title, path)
    else:
        colors = plt.cm.tab10.colors
        fig, ax = plt.subplots(figsize=(10, 4))
        max_freq = None
        filled = False
        for plot_idx, ch in enumerate(indices):
            data = buffer[:, ch]
            if data.size < 4:
                continue
            detrended = data - np.mean(data)
            window = np.hanning(detrended.size)
            windowed = detrended * window
            spectrum = np.fft.rfft(windowed)
            freq = np.fft.rfftfreq(detrended.size, d=1.0 / fs)
            power = (np.abs(spectrum) ** 2) / (np.sum(window**2) * fs)
            power_db = 10 * np.log10(power + 1e-12)

            if max_freq is None:
                max_freq = min(60.0, freq.max())
            freq_mask = freq <= max_freq
            color = colors[plot_idx % len(colors)]
            ax.plot(freq[freq_mask], power_db[freq_mask], linewidth=1.0, color=color, label=_ch_label(ch_labels, ch))

            if not filled:
                for name, low, high, band_color in bands:
                    band_mask = (freq >= low) & (freq <= high)
                    if not np.any(band_mask):
                        continue
                    ax.fill_between(freq[band_mask], power_db[band_mask], color=band_color, alpha=0.15, label=name)
                filled = True

        if max_freq is None:
            plt.close(fig)
            return
        ax.set_xlim(0, max_freq)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB/Hz)")
        label_part = "All channels" if len(indices) == total_channels else ", ".join(f"{ch + 1}" for ch in indices)
        ax.set_title(f"FFT - {label_part}")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        title = f"FFT - {label_part}"
        path = _write_matplotlib_window(fig, title, "live_preview_fft")
        _open_plot_window_once(path, "live_preview_fft")
        _plot_window_status(placeholder, title, path)
        plt.close(fig)


def _plot_individual_channels(
    buffer: np.ndarray,
    fs: float,
    placeholders: Optional[List["st.delta_generator.DeltaGenerator"]],
    indices: List[int],
    interactive: bool = True,
    window_seconds: Optional[float] = None,
    sample_offset: int = 0,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    if buffer.size == 0:
        return
    buffer, ch_labels = _eeg_view(buffer, params)
    if buffer.size == 0:
        return
    placeholders = placeholders or []
    offset = max(0, int(sample_offset))
    time_axis = (offset + np.arange(buffer.shape[0])) / fs if fs > 0 else offset + np.arange(buffer.shape[0])
    if fs > 0:
        time_axis = np.round(time_axis, 3)  # positive elapsed seconds, newest on the right
    if window_seconds is not None and fs > 0:
        time_axis_max = float(time_axis[-1])
        time_axis_min = max(0.0, time_axis_max - float(window_seconds))
    else:
        time_axis_min = float(time_axis[0])
        time_axis_max = float(time_axis[-1])
    palette = list(getattr(plt.cm, "tab10").colors) if hasattr(plt.cm, "tab10") else []
    default_color = (0.2, 0.4, 0.8)
    for plot_idx, ch in enumerate(indices):
        placeholder = placeholders[plot_idx] if plot_idx < len(placeholders) else None
        color = palette[plot_idx % len(palette)] if palette else default_color
        base_color = color if len(color) >= 3 else (list(color) + list(default_color))[:3]
        rgb = tuple(int(max(0, min(255, round(float(val) * 255)))) for val in base_color)
        if go is None or not interactive:
            fig, ax = plt.subplots(figsize=(14, 3))
            ax.plot(time_axis, buffer[:, ch], color=color, linewidth=1.0)
            ax.set_xlim(time_axis_min, time_axis_max)
            ax.set_xlabel("Time (s) - newest on the right" if fs > 0 else "Samples")
            ax.set_ylabel("Amplitude (uV)")
            ax.set_title(_ch_label(ch_labels, ch))
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            title = _ch_label(ch_labels, ch)
            key = f"live_channel_{ch + 1}"
            path = _write_matplotlib_window(fig, title, key)
            _open_plot_window_once(path, key)
            _plot_window_status(placeholder, title, path)
            plt.close(fig)
        else:
            trace = go.Scatter(
                x=time_axis,
                y=buffer[:, ch],
                mode="lines",
                line=dict(width=1.2, color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"),
                name=_ch_label(ch_labels, ch),
            )
            layout = go.Layout(
                height=280,
                margin=dict(l=40, r=10, t=30, b=40),
                xaxis=dict(
                    title="Time (s)" if fs > 0 else "Samples",
                    range=[time_axis_min, time_axis_max],
                ),
                yaxis=dict(title="Amplitude (uV)"),
                template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
            )
            title = _ch_label(ch_labels, ch)
            key = f"live_channel_{ch + 1}"
            path = _write_plotly_window(go.Figure(data=[trace], layout=layout), title, key)
            _open_plot_window_once(path, key)
            _plot_window_status(placeholder, title, path)




def _as_2d_array(chunk: Any) -> np.ndarray:
    array = np.asarray(chunk, dtype=float)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    return array


class LiveViewDevice(DeviceInterface):
    """Device wrapper that mirrors prime/acquire results into a Streamlit live view."""

    def __init__(self, wrapped: DeviceInterface, live_view: "LiveViewService", fs: float) -> None:
        self._wrapped = wrapped
        self._live_view = live_view
        self._fs = fs

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)

    def connect(self) -> None:
        self._wrapped.connect()

    def acquire(self, duration_seconds: float, aux_channels: int = 0):
        total = float(duration_seconds or 0.0)
        max_step = max(0.1, float(getattr(self._live_view, "max_update_seconds", 0.5)))
        if total <= max_step:
            chunk = self._wrapped.acquire(duration_seconds, aux_channels)
            self._live_view.push(chunk, self._fs)
            return chunk

        chunks: List[np.ndarray] = []
        remaining = total
        while remaining > 1e-9:
            step = min(max_step, remaining)
            chunk = self._wrapped.acquire(step, aux_channels)
            array = _as_2d_array(chunk)
            if array.size:
                chunks.append(array)
            self._live_view.push(array, self._fs)
            remaining -= step
        if not chunks:
            return np.empty((0, 0))
        return np.vstack(chunks)

    def prime(self, duration_seconds: float, aux_channels: int = 0):
        chunk = self._wrapped.prime(duration_seconds, aux_channels)
        self._live_view.push(chunk, self._fs)
        return chunk

    def disconnect(self) -> None:
        self._wrapped.disconnect()


def _normalize_channel_indices(indices: Optional[List[int]], total_channels: int) -> List[int]:
    if total_channels <= 0:
        return []
    if not indices:
        return list(range(total_channels))
    deduped = []
    for idx in indices:
        clipped = int(np.clip(idx, 0, total_channels - 1))
        if clipped not in deduped:
            deduped.append(clipped)
    return deduped or list(range(total_channels))


class LiveViewService:
    def __init__(
        self,
        placeholder: Optional["st.delta_generator.DeltaGenerator"],
        window_seconds: float = LIVE_WINDOW_SECONDS,
        channel_indices: Optional[List[int]] = None,
        fft_placeholder: Optional["st.delta_generator.DeltaGenerator"] = None,
        vep_placeholder: Optional["st.delta_generator.DeltaGenerator"] = None,
        progress_placeholder: Optional["st.delta_generator.DeltaGenerator"] = None,
        total_seconds: Optional[float] = None,
        max_update_seconds: float = 0.5,
        scale: Optional[float] = None,
        plot_type: str = DEFAULT_LIVE_PLOT,
    ) -> None:
        self.placeholder = placeholder
        self.fft_placeholder = fft_placeholder
        self.vep_placeholder = vep_placeholder
        self.progress_placeholder = progress_placeholder
        self.window_seconds = max(1.0, float(window_seconds))
        self.channel_indices = channel_indices or []
        self.total_seconds = float(total_seconds or 0.0)
        self.max_update_seconds = max(0.1, float(max_update_seconds))
        self.buffer: np.ndarray = np.empty((0, 0))
        self.vep_buffer: np.ndarray = np.empty((0, 0))
        self.samples_seen = 0
        self.fs = 0.0
        self._last_vep_render_samples = -1
        self.params: Dict[str, Any] = {}
        self.scale: Optional[float] = scale
        self.plot_type: str = plot_type or DEFAULT_LIVE_PLOT
        self._last_signal_render = 0.0

    def wrap_device(self, device: DeviceInterface, params: Dict[str, Any]) -> LiveViewDevice:
        fs_value = coerce_number(params.get("Parameters", {}).get("fs"))
        fs = float(fs_value) if fs_value else 0.0
        self.fs = fs
        self.params = params  # needed to reference and label the live buffer
        self.reset(clear_progress=False)
        return LiveViewDevice(device, self, fs)

    def reset(self, *, clear_progress: bool = True) -> None:
        self.buffer = np.empty((0, 0))
        self.vep_buffer = np.empty((0, 0))
        self.samples_seen = 0
        self._last_vep_render_samples = -1
        self._last_signal_render = 0.0
        if self.placeholder is not None:
            self.placeholder.empty()
        if self.progress_placeholder is not None:
            if clear_progress:
                self.progress_placeholder.empty()
            elif self.total_seconds > 0:
                self.progress_placeholder.progress(0.0, text=f"Recording 0.0s / {self.total_seconds:.1f}s")

    def mark_complete(self) -> None:
        if self.progress_placeholder is not None and self.total_seconds > 0:
            self.progress_placeholder.progress(1.0, text=f"Recording {self.total_seconds:.1f}s / {self.total_seconds:.1f}s")
        if self.placeholder is not None and self.buffer.size:
            indices = _normalize_channel_indices(self.channel_indices, self._eeg_width())
            offset = max(0, self.samples_seen - self.buffer.shape[0])
            # Final frame: the same chosen plot, rendered inline.
            _plot_live_buffer(
                self.buffer,
                self.fs,
                self.placeholder,
                channel_indices=indices,
                window_seconds=self.window_seconds,
                sample_offset=offset,
                params=self.params,
                scale=self.scale,
                plot_type=self.plot_type,
                render_inline=True,
                vep_placeholder=self.vep_placeholder,
                vep_buffer=self.vep_buffer,
                render_vep=True,
            )

    def _update_progress(self, fs: float) -> None:
        if self.progress_placeholder is None:
            return
        if self.total_seconds > 0 and fs > 0:
            elapsed = min(self.total_seconds, self.samples_seen / float(fs))
            fraction = min(1.0, elapsed / self.total_seconds)
            self.progress_placeholder.progress(fraction, text=f"Recording {elapsed:.1f}s / {self.total_seconds:.1f}s")
        elif self.samples_seen:
            self.progress_placeholder.caption(f"Recorded {self.samples_seen:,} samples.")

    def push(self, chunk: Any, fs: float) -> None:
        if chunk is None:
            return
        array = _as_2d_array(chunk)
        if array.size == 0:
            return
        self.samples_seen += int(array.shape[0])
        self._update_progress(fs)
        if self.placeholder is None:
            return
        if self.buffer.size == 0:
            self.buffer = array
        else:
            self.buffer = np.vstack([self.buffer, array])
        if self.vep_buffer.size == 0:
            self.vep_buffer = array.copy()
        else:
            self.vep_buffer = np.vstack([self.vep_buffer, array])
        max_window = int(fs * self.window_seconds) if fs > 0 else self.buffer.shape[0]
        if max_window > 0 and self.buffer.shape[0] > max_window:
            self.buffer = self.buffer[-max_window:]
        now = time.monotonic()
        render_vep = (
            self._last_vep_render_samples < 0
            or self.samples_seen - self._last_vep_render_samples
            >= int(round(VEP_REFRESH_SECONDS * fs))
        )
        if (
            self._last_signal_render
            and now - self._last_signal_render < self.max_update_seconds
            and not render_vep
        ):
            return
        indices = _normalize_channel_indices(self.channel_indices, self._eeg_width())
        offset = max(0, self.samples_seen - self.buffer.shape[0])
        # Render the single plot the operator chose, inline in the page (no dependency on a
        # pop-out window opening). FFT and single-channel are plot types now, so there is no
        # separate FFT window during streaming.
        _plot_live_buffer(
            self.buffer,
            fs,
            self.placeholder,
            channel_indices=indices,
            window_seconds=self.window_seconds,
            sample_offset=offset,
            params=self.params,
            scale=self.scale,
            plot_type=self.plot_type,
            render_inline=True,
            vep_placeholder=self.vep_placeholder,
            vep_buffer=self.vep_buffer,
            render_vep=render_vep,
        )
        if render_vep:
            self._last_vep_render_samples = self.samples_seen
        self._last_signal_render = now

    def _eeg_width(self) -> int:
        """Number of EEG columns the plots will actually see (non-EEG columns are dropped)."""
        if self.buffer.size == 0:
            return 0
        param_block = self.params.get("Parameters", {}) or {}
        return max(1, min(eeg_channel_count(param_block, self.buffer.shape[1]), self.buffer.shape[1]))


def run_live_preview(
    params: Dict[str, Any],
    placeholder: "st.delta_generator.DeltaGenerator",
    fft_placeholder: Optional["st.delta_generator.DeltaGenerator"],
    channel_indices: Optional[List[int]],
    channel_placeholders: Optional[List["st.delta_generator.DeltaGenerator"]] = None,
    initial_buffer: Optional[np.ndarray] = None,
    duration: float = 10.0,
    window: float = LIVE_WINDOW_SECONDS,
    update_interval: float = 0.25,
    final_interactive: bool = True,
    scale: Optional[float] = None,
    plot_type: str = DEFAULT_LIVE_PLOT,
    vep_placeholder=None,
) -> np.ndarray:
    params = dict(params)
    params.pop("data", None)

    fs_value = coerce_number(params.get("Parameters", {}).get("fs"))
    fs = float(fs_value) if fs_value else 0.0
    if fs <= 0:
        raise ValueError("Live preview requires a valid sampling rate (fs) in Parameters.")
    aux_channels = _resolve_aux_channels(params)
    indices = _normalize_channel_indices(channel_indices, int(params.get("Parameters", {}).get("NumberEEGChannels") or 0) or (params.get("Channels") and len(params.get("Channels")) or 1))
    LOGGER.info(
        "run_live_preview starting (device=%s, fs=%.2f, window=%.2f, interval=%.2f, channels=%s, aux=%s)",
        params.get("Device"),
        fs,
        window,
        update_interval,
        indices,
        aux_channels,
    )
    device = DeviceFactory.create(params)
    device.connect()

    try:
        analysis_buffer = np.asarray(initial_buffer, dtype=float) if initial_buffer is not None else np.empty((0, 0))
        if analysis_buffer.size and analysis_buffer.ndim == 1:
            analysis_buffer = analysis_buffer[:, np.newaxis]
        samples_seen = int(analysis_buffer.shape[0]) if analysis_buffer.size else 0
        buffer = analysis_buffer.copy()
        max_window = int(math.ceil(fs * window)) if fs > 0 else None
        if buffer.size and max_window:
            buffer = buffer[-max_window:]
        sample_offset = max(0, samples_seen - buffer.shape[0])
        prime_chunk = np.asarray(device.prime(min(update_interval, duration), aux_channels), dtype=float)
        if prime_chunk.ndim == 1:
            prime_chunk = prime_chunk[:, np.newaxis]
        samples_seen += int(prime_chunk.shape[0]) if prime_chunk.size else 0
        if analysis_buffer.size == 0:
            analysis_buffer = prime_chunk
        else:
            analysis_buffer = np.vstack([analysis_buffer, prime_chunk])
        buffer = analysis_buffer.copy()
        if max_window and buffer.shape[0] > max_window:
            buffer = buffer[-max_window:]
        sample_offset = max(0, samples_seen - buffer.shape[0])
        # Render the chosen plot inline, refreshed each chunk.
        last_vep_render_samples = -1
        last_signal_render = 0.0

        def _draw(buf: np.ndarray, offset: int, *, force_vep: bool = False) -> None:
            nonlocal last_vep_render_samples, last_signal_render
            now = time.monotonic()
            render_vep = (
                force_vep
                or last_vep_render_samples < 0
                or samples_seen - last_vep_render_samples
                >= int(round(VEP_REFRESH_SECONDS * fs))
            )
            if (
                not force_vep
                and last_vep_render_samples >= 0
                and now - last_signal_render < max(0.1, update_interval)
                and not render_vep
            ):
                return
            _plot_live_buffer(
                buf, fs, placeholder,
                channel_indices=indices,
                window_seconds=window,
                sample_offset=offset,
                params=params,
                scale=scale,
                plot_type=plot_type,
                render_inline=True,
                vep_placeholder=vep_placeholder,
                vep_buffer=analysis_buffer,
                render_vep=render_vep,
            )
            if render_vep:
                last_vep_render_samples = samples_seen
            last_signal_render = now

        _draw(buffer, sample_offset, force_vep=True)
        start = time.time()
        while (time.time() - start) < duration:
            remaining = duration - (time.time() - start)
            chunk = np.asarray(device.acquire(min(update_interval, remaining), aux_channels), dtype=float)
            if chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]
            if chunk.size > 0:
                samples_seen += int(chunk.shape[0])
                if analysis_buffer.size == 0:
                    analysis_buffer = chunk
                else:
                    analysis_buffer = np.vstack([analysis_buffer, chunk])
                buffer = analysis_buffer.copy()
                max_window = int(math.ceil(fs * window)) if fs > 0 else buffer.shape[0]
                if buffer.shape[0] > max_window:
                    buffer = buffer[-max_window:]
                sample_offset = max(0, samples_seen - buffer.shape[0])
                _draw(buffer, sample_offset)
            else:
                time.sleep(update_interval)
        # Final frame of the chosen plot.
        _draw(buffer, max(0, samples_seen - buffer.shape[0]), force_vep=True)
        return analysis_buffer
    finally:
        device.disconnect()
        LOGGER.info("run_live_preview finished")


