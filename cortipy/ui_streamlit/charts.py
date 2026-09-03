"""Chart data model + rendering primitives for the Streamlit UI.

Extracted from apps/streamlit_app.py (modularization). Holds the self-contained
data model, downsampling, and the matplotlib renderer. The chart *builders*
(``_chart_from_*``) stay in the app for now since they are tangled with evaluation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from cortipy.ui_streamlit.fields import coerce_number
from cortipy.ui_streamlit.plot_windows import (
    render_matplotlib_window_launcher,
    render_plotly_window_launcher,
    safe_window_key,
)
from cortipy.shared.channels import channel_labels
from cortipy.shared.reference import apply_eeg_reference, eeg_channel_count

try:  # pragma: no cover - optional UI dependency
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


@dataclass(frozen=True)
class ChartSeries:
    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass
class ChartData:
    key: str
    title: str
    x_label: str
    y_label: str
    series: List[ChartSeries]
    description: Optional[str] = None


def downsample_series(x: np.ndarray, y: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = max(1, math.ceil(len(x) / max_points))
    return x[::step], y[::step]


def referenced_eeg_view(params: Dict[str, Any], data: np.ndarray) -> np.ndarray:
    """Return a copied EEG view with the selected analysis reference applied."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] == 0:
        return arr

    param_block = params.get("Parameters", {}) if params else {}
    referenced = apply_eeg_reference(arr, param_block) if "ReferenceChannel" in param_block else arr.copy()
    eeg_count = eeg_channel_count(param_block, referenced.shape[1])
    device = str(params.get("Device", "") if params else "").lower()
    if device == "unicorn":
        eeg_count = min(8, eeg_count)
    return referenced[:, :eeg_count]


def _chart_to_plotly(chart: ChartData) -> Optional["go.Figure"]:
    if go is None:
        return None
    fig = go.Figure()
    for series in chart.series:
        fig.add_trace(
            go.Scatter(
                x=series.x,
                y=series.y,
                mode="lines",
                line=dict(width=1.5),
                name=series.name,
            )
        )
    fig.update_layout(
        title=chart.title,
        xaxis_title=chart.x_label,
        yaxis_title=chart.y_label,
        hovermode="x unified",
        height=560,
        margin=dict(l=70, r=25, t=70, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    if chart.description:
        fig.add_annotation(
            x=0,
            y=0,
            xref="paper",
            yref="paper",
            text=chart.description,
            showarrow=False,
            align="left",
            font=dict(size=11, color="#6b7280"),
        )
    return fig


def render_chart(chart: ChartData) -> None:
    plotly_fig = _chart_to_plotly(chart)
    if plotly_fig is not None:
        render_plotly_window_launcher(
            plotly_fig,
            chart.title,
            f"chart_{safe_window_key(chart.key)}",
            button_label="Open chart",
        )
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    for series in chart.series:
        ax.plot(series.x, series.y, label=series.name)
    ax.set_title(chart.title)
    ax.set_xlabel(chart.x_label)
    ax.set_ylabel(chart.y_label)
    if len(chart.series) > 1:
        ax.legend(loc="best")
    if chart.description:
        ax.text(
            0.01,
            0.02,
            chart.description,
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
            ha="left",
        )
    render_matplotlib_window_launcher(
        fig,
        chart.title,
        f"chart_{safe_window_key(chart.key)}",
        button_label="Open chart",
    )
    plt.close(fig)


def _chart_from_raw_data(label: str, params: Dict[str, Any], data: np.ndarray, aggregate: bool = False) -> Optional[ChartData]:
    if data is None:
        return None
    arr = np.asarray(data, dtype=float)
    if arr.ndim < 2 or arr.shape[0] == 0:
        return None

    param_block = params.get("Parameters", {}) if params else {}
    fs_value = coerce_number(param_block.get("fs"))
    fs = float(fs_value) if fs_value else 0.0
    n_samples, n_channels = arr.shape[0], arr.shape[1]
    max_seconds = 10
    limit = min(n_samples, int(fs * max_seconds) if fs > 0 else n_samples)
    time_axis = np.arange(n_samples) / fs if fs > 0 else np.arange(n_samples)
    x_vals = time_axis[:limit]
    series: List[ChartSeries] = []

    if aggregate or n_channels == 1:
        y_vals = np.mean(arr[:limit], axis=1)
        x_ds, y_ds = downsample_series(x_vals, y_vals)
        series.append(ChartSeries(name=label, x=x_ds, y=y_ds))
    else:
        max_channels = min(4, n_channels)
        names = channel_labels(params, n_channels)
        for ch in range(max_channels):
            y_vals = arr[:limit, ch]
            x_ds, y_ds = downsample_series(x_vals, y_vals)
            series.append(ChartSeries(name=f"{label} – {names[ch]}", x=x_ds, y=y_ds))

    return ChartData(
        key="raw",
        title="EEG preview (reference applied)",
        x_label="Time (s)" if fs > 0 else "Sample",
        y_label="Amplitude (uV)",
        series=series,
        description="First 10 seconds; saved raw data remains unchanged" if fs > 0 else "Full buffer preview; saved raw data remains unchanged",
    )


def _chart_from_psd(label: str, params: Dict[str, Any], data: np.ndarray, aggregate: bool = False) -> Optional[ChartData]:
    if data is None:
        return None
    arr = np.asarray(data, dtype=float)
    if arr.ndim < 2 or arr.shape[0] == 0:
        return None

    param_block = params.get("Parameters", {}) if params else {}
    fs_value = coerce_number(param_block.get("fs"))
    fs = float(fs_value) if fs_value else 0.0
    if fs <= 0:
        return None

    max_seconds = 10
    limit = min(arr.shape[0], int(fs * max_seconds)) or arr.shape[0]
    segment = arr[:limit]

    spectrum = np.fft.rfft(segment, axis=0)
    psd = (1.0 / (limit * fs)) * np.abs(spectrum) ** 2
    if psd.shape[0] > 2:
        psd[1:-1] *= 2
    freq = np.fft.rfftfreq(limit, d=1.0 / fs)

    series: List[ChartSeries] = []
    if aggregate or psd.shape[1] == 1:
        y_vals = 10.0 * np.log10(np.maximum(psd.mean(axis=1), np.finfo(float).tiny))
        series.append(ChartSeries(name=label, x=freq, y=y_vals))
    else:
        max_channels = min(4, psd.shape[1])
        names = channel_labels(params, psd.shape[1])
        for ch in range(max_channels):
            y_vals = 10.0 * np.log10(np.maximum(psd[:, ch], np.finfo(float).tiny))
            series.append(ChartSeries(name=f"{label} – {names[ch]}", x=freq, y=y_vals))

    return ChartData(
        key="psd",
        title="Power Spectral Density",
        x_label="Frequency (Hz)",
        y_label="Power (dB/Hz)",
        series=series,
        description="Welch-style PSD from first 10 seconds",
    )


def _chart_from_alpha_power(label: str, alpha_eval: Dict[str, Any], aggregate: bool = False, params: Optional[Dict[str, Any]] = None) -> Optional[ChartData]:
    if not alpha_eval:
        return None
    try:
        time_axis = np.asarray(alpha_eval.get("time"))
        power = np.asarray(alpha_eval.get("dBpsdx"))
    except Exception:
        return None
    if time_axis.ndim != 1 or power.ndim != 2 or power.shape[1] != time_axis.shape[0]:
        return None

    series: List[ChartSeries] = []
    if aggregate or power.shape[0] == 1:
        series.append(ChartSeries(name=label, x=time_axis, y=power.mean(axis=0)))
    else:
        max_channels = min(4, power.shape[0])
        names = channel_labels(params, power.shape[0])
        for idx in range(max_channels):
            series.append(ChartSeries(name=f"{label} – {names[idx]}", x=time_axis, y=power[idx]))

    return ChartData(
        key="alpha_power",
        title="Alpha band power",
        x_label=alpha_eval.get("timeUnit", "Time"),
        y_label=alpha_eval.get("dBpsdxUnit", "Power"),
        series=series,
    )


def _chart_from_eval_psd(label: str, psd_eval: Dict[str, Any], aggregate: bool = False, params: Optional[Dict[str, Any]] = None) -> Optional[ChartData]:
    if not psd_eval:
        return None
    try:
        freq = np.asarray(psd_eval.get("freq"))
        psd_values = psd_eval.get("dBpsdx") or psd_eval.get("psdx")
        if psd_values is None:
            return None
        psd_array = np.asarray(psd_values)
    except Exception:
        return None

    if freq.ndim != 1 or psd_array.size == 0:
        return None

    if psd_array.ndim == 3:
        psd_array = psd_array.mean(axis=2)
    if psd_array.ndim == 2 and psd_array.shape[0] == freq.shape[0] and psd_array.shape[1] != freq.shape[0]:
        psd_array = psd_array.T
    elif psd_array.ndim == 1:
        psd_array = psd_array[None, :]

    if psd_array.ndim != 2 or psd_array.shape[1] != freq.shape[0]:
        return None

    series: List[ChartSeries] = []
    if aggregate or psd_array.shape[0] == 1:
        series.append(ChartSeries(name=label, x=freq, y=psd_array.mean(axis=0)))
    else:
        max_channels = min(4, psd_array.shape[0])
        names = channel_labels(params, psd_array.shape[0])
        for idx in range(max_channels):
            series.append(ChartSeries(name=f"{label} – {names[idx]}", x=freq, y=psd_array[idx]))

    return ChartData(
        key="eval_psd",
        title="PSD (evaluation)",
        x_label=psd_eval.get("freqUnit", "Frequency (Hz)"),
        y_label=psd_eval.get("dBpsdxUnit") or psd_eval.get("psdxUnit") or "Power",
        series=series,
    )


def _chart_from_eval_fft(label: str, fft_eval: Dict[str, Any], aggregate: bool = False, params: Optional[Dict[str, Any]] = None) -> Optional[ChartData]:
    if not fft_eval:
        return None
    try:
        freq = np.asarray(fft_eval.get("freq"))
        xdft = np.asarray(fft_eval.get("xdft"))
    except Exception:
        return None
    if freq.ndim != 1 or xdft.size == 0:
        return None
    if xdft.ndim == 1:
        xdft = xdft[:, None]
    if xdft.ndim == 3:
        xdft = np.mean(xdft, axis=2)
    if xdft.shape[0] != freq.shape[0]:
        xdft = xdft.T if xdft.shape[1] == freq.shape[0] else xdft
    if xdft.shape[0] != freq.shape[0]:
        return None

    magnitude = np.abs(xdft)
    series: List[ChartSeries] = []
    if aggregate or magnitude.shape[1] == 1:
        series.append(ChartSeries(name=label, x=freq, y=magnitude.mean(axis=1)))
    else:
        max_channels = min(4, magnitude.shape[1])
        names = channel_labels(params, magnitude.shape[1])
        for idx in range(max_channels):
            series.append(ChartSeries(name=f"{label} – {names[idx]}", x=freq, y=magnitude[:, idx]))

    return ChartData(
        key="eval_fft",
        title="FFT (evaluation)",
        x_label=fft_eval.get("freqUnit", "Frequency (Hz)"),
        y_label=fft_eval.get("xdftUnit", "Amplitude"),
        series=series,
    )


def _chart_from_average_signals(label: str, avg_eval: Dict[str, Any], aggregate: bool = False, params: Optional[Dict[str, Any]] = None) -> Optional[ChartData]:
    if not avg_eval:
        return None
    try:
        time_axis = np.asarray(avg_eval.get("time"))
        voltage = np.asarray(avg_eval.get("voltage"))
    except Exception:
        return None
    if time_axis.ndim != 1 or voltage.ndim != 2 or voltage.shape[0] != time_axis.shape[0]:
        return None

    series: List[ChartSeries] = []
    if aggregate or voltage.shape[1] == 1:
        series.append(ChartSeries(name=label, x=time_axis, y=voltage.mean(axis=1)))
    else:
        max_channels = min(4, voltage.shape[1])
        names = channel_labels(params, voltage.shape[1])
        for idx in range(max_channels):
            series.append(ChartSeries(name=f"{label} – {names[idx]}", x=time_axis, y=voltage[:, idx]))

    return ChartData(
        key="avg_signals",
        title="Average signals (evaluation)",
        x_label=avg_eval.get("timeUnit", "Time"),
        y_label=avg_eval.get("voltageUnit", "Amplitude"),
        series=series,
    )


def _chart_from_metric_vector(label: str, name: str, values: Any, unit: str = "Value") -> Optional[ChartData]:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size == 0:
        return None
    x_axis = np.arange(1, arr.size + 1)
    return ChartData(
        key=f"metric_{name}",
        title=f"{label} – {name}",
        x_label="Index / Channel",
        y_label=unit,
        series=[ChartSeries(name=name, x=x_axis, y=arr)],
    )
