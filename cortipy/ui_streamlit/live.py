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


def _ch_label(labels: List[str], idx: int) -> str:
    return labels[idx] if 0 <= idx < len(labels) else f"Ch {idx + 1}"


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
) -> None:
    if buffer.size == 0:
        return
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
    if interactive and go is not None:
        palette = list(getattr(plt.cm, "tab10").colors) if hasattr(plt.cm, "tab10") else []
        traces = []
        for plot_idx, ch in enumerate(indices):
            color = palette[plot_idx % len(palette)] if palette else (0.2, 0.4, 0.8)
            rgb = tuple(int(max(0, min(255, round(float(val) * 255)))) for val in (list(color) + [0, 0, 0])[:3])
            traces.append(
                go.Scatter(
                    x=time_axis,
                    y=buffer[:, ch],
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
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = plt.cm.tab10.colors
        for plot_idx, ch in enumerate(indices):
            color = colors[plot_idx % len(colors)]
            ax.plot(time_axis, buffer[:, ch], label=_ch_label(ch_labels, ch), color=color)
        ax.set_xlim(time_axis_min, time_axis_max)
        ax.set_xlabel("Time (s) - newest on the right" if fs > 0 else "Samples")
        ax.set_ylabel("Amplitude (uV)")
        label_part = _label_part(ch_labels, indices, total_channels)
        ax.set_title(f"Live preview - {label_part}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        title = f"Live preview - {label_part}"
        path = _write_matplotlib_window(fig, title, "live_preview_signal")
        _open_plot_window_once(path, "live_preview_signal")
        _plot_window_status(placeholder, title, path)
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
        window_seconds: float = 5.0,
        channel_indices: Optional[List[int]] = None,
        fft_placeholder: Optional["st.delta_generator.DeltaGenerator"] = None,
        progress_placeholder: Optional["st.delta_generator.DeltaGenerator"] = None,
        total_seconds: Optional[float] = None,
        max_update_seconds: float = 0.5,
        final_channel_windows: bool = False,
    ) -> None:
        self.placeholder = placeholder
        self.fft_placeholder = fft_placeholder
        self.progress_placeholder = progress_placeholder
        self.window_seconds = max(1.0, float(window_seconds))
        self.channel_indices = channel_indices or []
        self.total_seconds = float(total_seconds or 0.0)
        self.max_update_seconds = max(0.1, float(max_update_seconds))
        self.final_channel_windows = bool(final_channel_windows)
        self.buffer: np.ndarray = np.empty((0, 0))
        self.samples_seen = 0
        self.fs = 0.0
        self.params: Dict[str, Any] = {}

    def wrap_device(self, device: DeviceInterface, params: Dict[str, Any]) -> LiveViewDevice:
        fs_value = coerce_number(params.get("Parameters", {}).get("fs"))
        fs = float(fs_value) if fs_value else 0.0
        self.fs = fs
        self.params = params  # needed to reference and label the live buffer
        self.reset(clear_progress=False)
        return LiveViewDevice(device, self, fs)

    def reset(self, *, clear_progress: bool = True) -> None:
        self.buffer = np.empty((0, 0))
        self.samples_seen = 0
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
            _plot_live_buffer(
                self.buffer,
                self.fs,
                self.placeholder,
                channel_indices=indices,
                interactive=True,
                window_seconds=self.window_seconds,
                sample_offset=offset,
                params=self.params,
            )
            if self.fft_placeholder is not None:
                _plot_fft_spectrum(
                    self.buffer,
                    self.fs,
                    self.fft_placeholder,
                    channel_indices=indices,
                    interactive=True,
                    params=self.params,
                )
            if self.final_channel_windows:
                _plot_individual_channels(
                    self.buffer,
                    self.fs,
                    None,
                    indices,
                    interactive=True,
                    window_seconds=self.window_seconds,
                    sample_offset=offset,
                    params=self.params,
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
        max_window = int(fs * self.window_seconds) if fs > 0 else self.buffer.shape[0]
        if max_window > 0 and self.buffer.shape[0] > max_window:
            self.buffer = self.buffer[-max_window:]
        indices = _normalize_channel_indices(self.channel_indices, self._eeg_width())
        offset = max(0, self.samples_seen - self.buffer.shape[0])
        _plot_live_buffer(
            self.buffer,
            fs,
            self.placeholder,
            channel_indices=indices,
            window_seconds=self.window_seconds,
            sample_offset=offset,
            params=self.params,
        )
        if self.fft_placeholder is not None:
            _plot_fft_spectrum(
                self.buffer, fs, self.fft_placeholder, channel_indices=indices, params=self.params
            )

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
    window: float = 5.0,
    update_interval: float = 0.25,
    final_interactive: bool = True,
) -> np.ndarray:
    params = dict(params)
    params.pop("data", None)

    fs_value = coerce_number(params.get("Parameters", {}).get("fs"))
    fs = float(fs_value) if fs_value else 0.0
    if fs <= 0:
        raise ValueError("Live preview requires a valid sampling rate (fs) in Parameters.")
    channels_enabled = bool(channel_placeholders)

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
        buffer = np.asarray(initial_buffer, dtype=float) if initial_buffer is not None else np.empty((0, 0))
        if buffer.size and buffer.ndim == 1:
            buffer = buffer[:, np.newaxis]
        samples_seen = int(buffer.shape[0]) if buffer.size else 0
        max_window = int(math.ceil(fs * window)) if fs > 0 else None
        if buffer.size and max_window:
            buffer = buffer[-max_window:]
        sample_offset = max(0, samples_seen - buffer.shape[0])
        prime_chunk = np.asarray(device.prime(min(update_interval, duration), aux_channels), dtype=float)
        if prime_chunk.ndim == 1:
            prime_chunk = prime_chunk[:, np.newaxis]
        samples_seen += int(prime_chunk.shape[0]) if prime_chunk.size else 0
        if buffer.size == 0:
            buffer = prime_chunk
        else:
            buffer = np.vstack([buffer, prime_chunk])
        if max_window and buffer.shape[0] > max_window:
            buffer = buffer[-max_window:]
        sample_offset = max(0, samples_seen - buffer.shape[0])
        # Live view writes auto-refreshing pop-out windows; the page stays as a control surface.
        _plot_live_buffer(
            buffer,
            fs,
            placeholder,
            channel_indices=indices,
            interactive=False,
            window_seconds=window,
            sample_offset=sample_offset,
            params=params,
        )
        _plot_fft_spectrum(
            buffer, fs, fft_placeholder, channel_indices=indices, interactive=False, params=params
        )
        # To keep UI smooth, only show the aggregated view + FFT during streaming.
        start = time.time()
        while (time.time() - start) < duration:
            remaining = duration - (time.time() - start)
            chunk = np.asarray(device.acquire(min(update_interval, remaining), aux_channels), dtype=float)
            if chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]
            if chunk.size > 0:
                samples_seen += int(chunk.shape[0])
                if buffer.size == 0:
                    buffer = chunk
                else:
                    buffer = np.vstack([buffer, chunk])
                max_window = int(math.ceil(fs * window)) if fs > 0 else buffer.shape[0]
                if buffer.shape[0] > max_window:
                    buffer = buffer[-max_window:]
                sample_offset = max(0, samples_seen - buffer.shape[0])
                _plot_live_buffer(
                    buffer,
                    fs,
                    placeholder,
                    channel_indices=indices,
                    interactive=False,
                    window_seconds=window,
                    sample_offset=sample_offset,
                    params=params,
                )
                _plot_fft_spectrum(
                    buffer, fs, fft_placeholder, channel_indices=indices, interactive=False, params=params
                )
            else:
                time.sleep(update_interval)
        # After capture, replace auto-refreshing windows with editable final plot windows.
        if final_interactive:
            sample_offset = max(0, samples_seen - buffer.shape[0])
            _plot_live_buffer(
                buffer,
                fs,
                placeholder,
                channel_indices=indices,
                interactive=True,
                window_seconds=window,
                sample_offset=sample_offset,
                params=params,
            )
            _plot_fft_spectrum(
                buffer, fs, fft_placeholder, channel_indices=indices, interactive=True, params=params
            )
            if channels_enabled:
                _plot_individual_channels(
                    buffer,
                    fs,
                    channel_placeholders or [],
                    indices,
                    interactive=True,
                    window_seconds=window,
                    sample_offset=sample_offset,
                    params=params,
                )
        return buffer
    finally:
        device.disconnect()
        LOGGER.info("run_live_preview finished")


