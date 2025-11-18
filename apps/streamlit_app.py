"""Streamlit UI for configuring and running cortipy sessions."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
import html
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
import streamlit as st
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

try:
    import mne  # type: ignore
except Exception:  # pragma: no cover
    mne = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortipy import MeasurementPipeline
from cortipy.core.pipeline import PipelineHooks
from cortipy.ui import SaveManager, normalize_params

DEFAULT_SAVE_DIR = Path.cwd() / "cortipy_runs"
SCHEMA_DIR = ROOT / "ParameterJSON"
SUPPORTED_EXTRA_DEVICES = ["LSL", "Offline", "Dummy"]
DEVICE_CONFIG_SCHEMA = {
    "UNICORN": [
        {
            "name": "UNICORNPort",
            "label": "UNICORN Port / Address",
            "kind": "text",
            "placeholder": "COM7 or /dev/tty.Unicorn-DevB",
            "help": "Enter the virtual COM port (USB/Bluetooth serial) exposed by the UNICORN.",
            "default": "",
            "aliases": ["UnicornPort", "UNICORNAddress", "UnicornAddress"],
        },
        {
            "name": "UNICORNDeviceName",
            "label": "Device Name (optional)",
            "kind": "text",
            "placeholder": "EEG-Headset-01",
            "help": "Friendly name stored alongside the recording (appears in logs).",
            "default": "",
            "aliases": ["UnicornDeviceName"],
        },
        {
            "name": "UnicornTimeout",
            "label": "Connection timeout (s)",
            "kind": "number",
            "default": 5.0,
            "min": 0.5,
            "max": 30.0,
            "step": 0.5,
            "help": "Maximum time to wait for the UNICORN stream handshake.",
            "aliases": ["UNICORNTimeout"],
        },
    ]
}
DEVICE_FIELD_ALIASES: Dict[str, List[str]] = {}
for fields in DEVICE_CONFIG_SCHEMA.values():
    for field in fields:
        DEVICE_FIELD_ALIASES[field["name"]] = field.get("aliases", [])


@dataclass(frozen=True)
class FieldSchema:
    name: str
    kind: str
    options: List[str]
    tooltip: str


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


def device_default_values(device: str) -> Dict[str, Any]:
    return {field["name"]: field.get("default") for field in DEVICE_CONFIG_SCHEMA.get(device, [])}


def _load_schema_file(path: Path) -> tuple[str, List[FieldSchema]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    (root_key, fields_raw), *_ = data.items()
    fields = [
        FieldSchema(
            name=entry["name"],
            kind=entry["type"],
            options=[str(opt) for opt in entry.get("options", [])],
            tooltip=entry.get("tooltip", "").strip(),
        )
        for entry in fields_raw
    ]
    return root_key, fields


GENERAL_SCHEMA = _load_schema_file(SCHEMA_DIR / "GeneralParams.json")[1]
METHOD_SCHEMAS: Dict[str, List[FieldSchema]] = {}
for schema_path in SCHEMA_DIR.glob("*.json"):
    if schema_path.name in {"GeneralParams.json", "electrodes.json"}:
        continue
    key, fields = _load_schema_file(schema_path)
    METHOD_SCHEMAS[key] = fields

ELECTRODE_LIBRARY: Dict[str, List[str]] = json.loads((SCHEMA_DIR / "electrodes.json").read_text(encoding="utf-8"))
ELECTRODE_RUBRICS = list(ELECTRODE_LIBRARY.keys())
ELECTRODE_MODELS = sorted({model for models in ELECTRODE_LIBRARY.values() for model in models})

DEVICE_DEFAULT_CHANNELS = {
    "ActiCHamp": 32,
    "UNICORN": 8,
    "BIOPACK": 16,
    "LSL": 8,
    "Offline": 8,
    "Dummy": 8,
}
DEVICE_EXTRA_LABELS = {
    "ActiCHamp": ["GND"],
    "UNICORN": ["GND", "Ref"],
}

INT_FIELD_NAMES = {
    "NumberEEGChannels",
    "NumberAUXChannels",
    "ReferenceChannel",
    "TriggerChannel",
    "LivePlotCH",
    "ChannelIpsi",
    "ChannelContra",
    "TestSubjectNo",
    "RecordingTime",
    "TriggerTime",
    "RepeatMeasCount",
}
INT_FIELD_PREFIXES = ("Number",)
INT_FIELD_SUFFIXES = ("Channel", "Channels", "Trials", "Count", "No")

TEN_TWENTY_32 = [
    "Fp1",
    "Fpz",
    "Fp2",
    "AF3",
    "AFz",
    "AF4",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO3",
    "PO4",
    "Oz",
]
UNICORN_8 = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
BIOPACK_16 = TEN_TWENTY_32[:16]

DEVICE_POSITION_DEFAULTS = {
    "ActiCHamp": TEN_TWENTY_32,
    "UNICORN": UNICORN_8,
    "BIOPACK": BIOPACK_16,
    "LSL": UNICORN_8,
    "Offline": UNICORN_8,
    "Dummy": UNICORN_8,
}


def _normalize_position_label(label: str) -> str:
    return re.sub(r"\s+", "", str(label or "").strip()).upper()


_STANDARD_POSITION_ORDER: List[str] = []
for positions in DEVICE_POSITION_DEFAULTS.values():
    for pos in positions:
        normalized = _normalize_position_label(pos)
        if normalized and normalized not in _STANDARD_POSITION_ORDER:
            _STANDARD_POSITION_ORDER.append(normalized)
for extras in DEVICE_EXTRA_LABELS.values():
    for label in extras:
        normalized = _normalize_position_label(label)
        if normalized and normalized not in _STANDARD_POSITION_ORDER:
            _STANDARD_POSITION_ORDER.append(normalized)
if not _STANDARD_POSITION_ORDER:
    _STANDARD_POSITION_ORDER = [f"CH{idx+1}" for idx in range(32)]
_POSITION_ANGLE_LOOKUP = {label: idx for idx, label in enumerate(_STANDARD_POSITION_ORDER)}

METHOD_FULL_NAMES: Dict[str, str] = {
    "Alpha": "Alpha Relaxation",
    "ASSR": "Auditory Steady-State Response",
    "BCI": "SSVEP Brain-Computer Interface",
    "BERA": "Brainstem Evoked Response Audiometry",
    "P300": "Visual Oddball P300",
    "SSVEP": "Steady-State Visual Evoked Potential",
    "VEP": "Transient Visual Evoked Potential",
}
METHOD_DESCRIPTIONS: Dict[str, str] = {
    "Alpha": "Eyes-closed relaxation run to monitor 8–12 Hz activity.",
    "ASSR": "Amplitude-modulated tones to probe auditory entrainment.",
    "BCI": "Frequency-coded checkerboards for real-time BCI control.",
    "BERA": "Click trains capturing early brainstem responses.",
    "P300": "Oddball stimuli evoking the P300 component.",
    "SSVEP": "Continuous flicker to follow steady-state responses.",
    "VEP": "Transient pattern reversal for latency tracking.",
}

PARTICIPANT_CARD_STYLE = """
<style>
.participant-card {
    background: linear-gradient(135deg, #f3f6ff 0%, #fff7f0 100%);
    border-radius: 16px;
    border: 1px solid #dbe2ef;
    padding: 0.85rem 1rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    margin-bottom: 0.75rem;
}
.participant-card .avatar {
    font-size: 48px;
    text-align: center;
    margin-bottom: 0.4rem;
}
.participant-card .field {
    display: flex;
    gap: 0.65rem;
    margin-bottom: 0.35rem;
    align-items: baseline;
}
.participant-card .field:last-child {
    margin-bottom: 0;
}
.participant-card .icon {
    font-size: 1.1rem;
}
.participant-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 0.05rem;
}
.participant-card .value {
    font-size: 0.95rem;
    color: #111827;
    font-weight: 600;
}
.participant-card .notes {
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: #374151;
    padding-top: 0.4rem;
    border-top: 1px solid rgba(255, 255, 255, 0.6);
}
</style>
"""


def _load_standard_xy() -> Dict[str, np.ndarray]:
    if mne is None:
        return {}
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
    except Exception:
        return {}
    ch_pos = montage.get_positions().get("ch_pos", {})
    xy: Dict[str, np.ndarray] = {}
    for name, coords in ch_pos.items():
        xy[_normalize_position_label(name)] = np.asarray(coords[:2], dtype=float)
    return xy


_STANDARD_1020_XY = _load_standard_xy()
if _STANDARD_1020_XY:
    _STANDARD_XY_SCALE = 0.95 / max(float(np.linalg.norm(val)) for val in _STANDARD_1020_XY.values())
else:
    _STANDARD_XY_SCALE = 1.0

PARTICIPANT_DEFAULT = {
    "Code": "",
    "Initials": "",
    "Age": None,
    "Gender": "Unspecified",
    "DominantHand": "Right",
    "Notes": "",
}
GENDER_OPTIONS = ["Unspecified", "Female", "Male", "Diverse"]
HANDEDNESS_OPTIONS = ["Right", "Left", "Ambidextrous"]


def default_values(fields: Iterable[FieldSchema]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for field in fields:
        if field.kind == "dropdown":
            defaults[field.name] = field.options[0] if field.options else ""
        elif field.kind == "numeric":
            defaults[field.name] = None
        else:
            defaults[field.name] = ""
    return defaults


def resolve_choice(options: List[str], current: Optional[str]) -> str:
    if current in options:
        return current
    return options[0] if options else ""


def is_integer_field(name: str) -> bool:
    if name in INT_FIELD_NAMES:
        return True
    lower = name.lower()
    if any(lower.startswith(prefix.lower()) for prefix in INT_FIELD_PREFIXES):
        return True
    if any(lower.endswith(suffix.lower()) for suffix in INT_FIELD_SUFFIXES):
        return True
    return False


def render_numeric_input(target, field: FieldSchema, current: Any, key: str):
    as_number = coerce_number(current)
    if is_integer_field(field.name):
        default = int(as_number) if as_number is not None else 0
        value = target.number_input(
            field.name,
            value=default,
            step=1,
            format="%d",
            help=field.tooltip or None,
            key=key,
        )
        return int(value)
    default = float(as_number) if as_number is not None else 0.0
    return target.number_input(
        field.name,
        value=default,
        step=0.1,
        format="%.3f",
        help=field.tooltip or None,
        key=key,
    )


def _load_npz_array(source: Union[Path, Any]) -> Optional[np.ndarray]:
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        npz = np.load(source)
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to load NPZ data: {exc}")
        return None
    try:
        if isinstance(npz, np.ndarray):
            return np.asarray(npz)
        files = list(getattr(npz, "files", []))
        if not files:
            st.error("NPZ archive is empty.")
            return None
        key = "data" if "data" in files else files[0]
        return np.asarray(npz[key])
    finally:
        if hasattr(npz, "close"):
            npz.close()


def _downsample_series(x: np.ndarray, y: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = max(1, math.ceil(len(x) / max_points))
    return x[::step], y[::step]


def _chart_from_raw_data(label: str, params: Dict[str, Any], data: np.ndarray, aggregate: bool = False) -> Optional[ChartData]:
    if data is None:
        return None
    arr = np.asarray(data)
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
        x_ds, y_ds = _downsample_series(x_vals, y_vals)
        series.append(ChartSeries(name=label, x=x_ds, y=y_ds))
    else:
        max_channels = min(4, n_channels)
        for ch in range(max_channels):
            y_vals = arr[:limit, ch]
            x_ds, y_ds = _downsample_series(x_vals, y_vals)
            series.append(ChartSeries(name=f"{label} – Ch {ch + 1}", x=x_ds, y=y_ds))

    return ChartData(
        key="raw",
        title="Raw EEG preview",
        x_label="Time (s)" if fs > 0 else "Sample",
        y_label="Amplitude (uV)",
        series=series,
        description="First 10 seconds" if fs > 0 else "Full buffer preview",
    )


def _chart_from_psd(label: str, params: Dict[str, Any], data: np.ndarray, aggregate: bool = False) -> Optional[ChartData]:
    if data is None:
        return None
    arr = np.asarray(data)
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
        for ch in range(max_channels):
            y_vals = 10.0 * np.log10(np.maximum(psd[:, ch], np.finfo(float).tiny))
            series.append(ChartSeries(name=f"{label} – Ch {ch + 1}", x=freq, y=y_vals))

    return ChartData(
        key="psd",
        title="Power Spectral Density",
        x_label="Frequency (Hz)",
        y_label="Power (dB/Hz)",
        series=series,
        description="Welch-style PSD from first 10 seconds",
    )


def _chart_from_alpha_power(label: str, alpha_eval: Dict[str, Any], aggregate: bool = False) -> Optional[ChartData]:
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
        for idx in range(max_channels):
            series.append(ChartSeries(name=f"{label} – Ch {idx + 1}", x=time_axis, y=power[idx]))

    return ChartData(
        key="alpha_power",
        title="Alpha band power",
        x_label=alpha_eval.get("timeUnit", "Time"),
        y_label=alpha_eval.get("dBpsdxUnit", "Power"),
        series=series,
    )


def _chart_from_eval_psd(label: str, psd_eval: Dict[str, Any], aggregate: bool = False) -> Optional[ChartData]:
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
        for idx in range(max_channels):
            series.append(ChartSeries(name=f"{label} – Ch {idx + 1}", x=freq, y=psd_array[idx]))

    return ChartData(
        key="eval_psd",
        title="PSD (evaluation)",
        x_label=psd_eval.get("freqUnit", "Frequency (Hz)"),
        y_label=psd_eval.get("dBpsdxUnit") or psd_eval.get("psdxUnit") or "Power",
        series=series,
    )


def collect_chart_data(label: str, params: Dict[str, Any], data: Optional[np.ndarray], aggregate: bool = False) -> Dict[str, ChartData]:
    charts: Dict[str, ChartData] = {}
    if data is not None:
        raw_chart = _chart_from_raw_data(label, params, data, aggregate=aggregate)
        if raw_chart:
            charts[raw_chart.key] = raw_chart
        psd_chart = _chart_from_psd(label, params, data, aggregate=aggregate)
        if psd_chart:
            charts[psd_chart.key] = psd_chart

    evaluation = params.get("Evaluation") if params else None
    if isinstance(evaluation, dict):
        alpha_chart = _chart_from_alpha_power(label, evaluation.get("alphaPower", {}), aggregate=aggregate)
        if alpha_chart:
            charts[alpha_chart.key] = alpha_chart
        eval_psd_chart = _chart_from_eval_psd(label, evaluation.get("PSD", {}), aggregate=aggregate)
        if eval_psd_chart:
            charts[eval_psd_chart.key] = eval_psd_chart

    return charts


def render_chart(chart: ChartData) -> None:
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
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_chart_section(label: str, params: Dict[str, Any], data: Optional[np.ndarray], aggregate: bool = False) -> None:
    charts = collect_chart_data(label, params or {}, data, aggregate=aggregate)
    if not charts:
        st.info("No charts available for this session yet.")
        return
    st.subheader(f"Charts – {label}")
    for key in sorted(charts.keys()):
        render_chart(charts[key])


def render_comparison_charts(payloads: List[tuple[str, Dict[str, Any], Optional[np.ndarray]]]) -> None:
    merged: Dict[str, ChartData] = {}
    for label, params, data in payloads:
        charts = collect_chart_data(label, params or {}, data, aggregate=True)
        for key, chart in charts.items():
            if key not in merged:
                merged[key] = ChartData(
                    key=key,
                    title=f"{chart.title} (comparison)",
                    x_label=chart.x_label,
                    y_label=chart.y_label,
                    series=list(chart.series),
                    description=chart.description,
                )
            else:
                merged[key].series.extend(chart.series)

    if not merged:
        st.info("No comparable charts for the selected sessions.")
        return

    st.subheader("Comparison")
    for key in sorted(merged.keys()):
        render_chart(merged[key])


def load_params_into_state(params: Dict[str, Any], data_override: Optional[np.ndarray] = None) -> None:
    general = st.session_state["general_form"]
    method = params.get("Method") or general.get("Method") or "Alpha"
    device = params.get("Device") or general.get("Device") or "LSL"
    general["Method"] = method
    general["Device"] = device

    parameters = params.get("Parameters", {})
    for field in GENERAL_SCHEMA:
        if field.name in {"Method", "Device"}:
            continue
        if field.name in parameters:
            general[field.name] = parameters[field.name]

    schema = METHOD_SCHEMAS.get(method, [])
    method_state = st.session_state["method_forms"].setdefault(method, default_values(schema))
    for field in schema:
        if field.name in parameters:
            method_state[field.name] = parameters[field.name]

    if "device_forms" not in st.session_state:
        st.session_state["device_forms"] = {name: device_default_values(name) for name in DEVICE_CONFIG_SCHEMA}
    device_fields = DEVICE_CONFIG_SCHEMA.get(device, [])
    if device_fields:
        device_state = st.session_state["device_forms"].setdefault(device, device_default_values(device))
        for field in device_fields:
            value = parameters.get(field["name"]) or params.get(field["name"])
            if value in (None, "", []):
                for alias in field.get("aliases", []):
                    alias_value = parameters.get(alias) or params.get(alias)
                    if alias_value not in (None, "", []):
                        value = alias_value
                        break
            if value not in (None, "", []):
                device_state[field["name"]] = value

    if params.get("Channels"):
        imported_rows = []
        for ch in params["Channels"]:
            channel_name = ch.get("Channel") or ch.get("Label") or ch.get("name")
            if not channel_name:
                continue
            imported_rows.append(
                {
                    "Channel": channel_name,
                    "Position": ch.get("Position") or channel_name.replace(" ", ""),
                    "Rubrik": ch.get("Rubrik") or ch.get("Rubric") or (ELECTRODE_RUBRICS[0] if ELECTRODE_RUBRICS else ""),
                    "Model": ch.get("Model") or (ELECTRODE_MODELS[0] if ELECTRODE_MODELS else ""),
                    "Impedance": coerce_number(ch.get("Impedance")) or 0.0,
                    "Active": bool(ch.get("Active", True)),
                }
            )
        st.session_state["channel_tables"][device] = ensure_channel_rows(device, imported_rows)

    metadata = params.get("Metadata") or {}
    participant_meta = metadata.get("Participant")
    if participant_meta:
        st.session_state["participant"].update(participant_meta)

    data_array = data_override
    if data_array is None and params.get("data") is not None:
        data_array = np.asarray(params["data"])
    st.session_state["imported_data"] = data_array
    st.session_state["use_imported_data"] = bool(data_array)
    st.session_state["imported_params_raw"] = params


def ensure_state() -> None:
    if "general_form" not in st.session_state:
        st.session_state["general_form"] = default_values(GENERAL_SCHEMA)
        # Prefer Alpha as initial method if available.
        if "Method" in st.session_state["general_form"]:
            st.session_state["general_form"]["Method"] = st.session_state["general_form"]["Method"] or "Alpha"
    if "method_forms" not in st.session_state:
        st.session_state["method_forms"] = {name: default_values(fields) for name, fields in METHOD_SCHEMAS.items()}
    if "device_forms" not in st.session_state:
        st.session_state["device_forms"] = {device: device_default_values(device) for device in DEVICE_CONFIG_SCHEMA}
    if "participant" not in st.session_state:
        st.session_state["participant"] = dict(PARTICIPANT_DEFAULT)
    if "channel_tables" not in st.session_state:
        st.session_state["channel_tables"] = {}
    if "imported_data" not in st.session_state:
        st.session_state["imported_data"] = None
    if "imported_params_raw" not in st.session_state:
        st.session_state["imported_params_raw"] = None
    if "use_imported_data" not in st.session_state:
        st.session_state["use_imported_data"] = False


def coerce_number(value: Any) -> Optional[float | int]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number
    return None


def parse_numeric_list(text: str) -> Optional[List[float | int]]:
    tokens = [tok for tok in re.split(r"[,\s;]+", text.strip()) if tok]
    if len(tokens) <= 1:
        return None
    parsed: List[float | int] = []
    for token in tokens:
        number = coerce_number(token)
        if number is None:
            return None
        parsed.append(number)
    return parsed


def convert_value(field: FieldSchema, value: Any) -> Any:
    if value in ("", None):
        return None
    if field.kind == "numeric":
        number = coerce_number(value)
        if number is None:
            return None
        return int(number) if is_integer_field(field.name) else number
    if field.kind == "dropdown":
        casted = coerce_number(value)
        return casted if casted is not None else value
    if field.kind == "edit":
        text = str(value).strip()
        if not text:
            return None
        numeric_list = parse_numeric_list(text)
        return numeric_list if numeric_list is not None else text
    return value


def _position_angle(label: str, fallback_idx: int) -> float:
    total = len(_STANDARD_POSITION_ORDER) or 1
    key = _normalize_position_label(label)
    idx = _POSITION_ANGLE_LOOKUP.get(key)
    if idx is None:
        idx = fallback_idx % total
    return 2 * math.pi * (idx / total)


def render_config_snapshot(method: str, device: str, general: Dict[str, Any]) -> None:
    method = method or "—"
    device = device or "—"
    full_name = METHOD_FULL_NAMES.get(method)
    desc = METHOD_DESCRIPTIONS.get(method)
    params_block = general if "Parameters" not in general else general.get("Parameters", {})
    fs_value = coerce_number(params_block.get("fs") or general.get("fs"))
    channels = coerce_number(params_block.get("NumberEEGChannels") or general.get("NumberEEGChannels"))
    duration = coerce_number(params_block.get("RecordingTime") or general.get("RecordingTime"))

    method_label = f"{method} – {full_name}" if full_name else method
    lines = [f"**Method:** {method_label}"]
    if desc:
        lines.append(desc)
    lines.append(f"**Device:** {device}")

    fs_txt = f"{fs_value} Hz" if fs_value is not None else "—"
    ch_txt = f"{int(channels)}" if channels is not None else "—"
    dur_txt = f"{duration} s" if duration is not None else "— s"

    stats_md = f"fs: {fs_txt}  \nChannels: {ch_txt}  \nRecording time: {dur_txt}"
    st.markdown("\n\n".join(lines + [stats_md]))


def _electrode_map_figure(device: str, rows: List[Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_aspect("equal")
    ax.axis("off")
    head_radius = 1.05
    ax.set_facecolor("#fbfbfd")
    ax.add_patch(Circle((0, 0), head_radius, facecolor="#f8fafc", edgecolor="#90a4ae", linewidth=1.2))
    ax.add_patch(Circle((0, 0), 0.35, fill=False, linestyle="--", linewidth=1.0, edgecolor="#b0bec5"))
    ax.add_patch(
        Polygon(
            [(0.0, head_radius), (0.08, head_radius + 0.18), (-0.08, head_radius + 0.18)],
            closed=True,
            facecolor="#ffe0b2",
            edgecolor="#fb8c00",
            linewidth=1.0,
        )
    )
    ax.add_patch(Rectangle((-head_radius - 0.03, -0.25), 0.12, 0.5, facecolor="#f5f5f5", edgecolor="#b0bec5", linewidth=1.0))
    ax.add_patch(Rectangle((head_radius - 0.09, -0.25), 0.12, 0.5, facecolor="#f5f5f5", edgecolor="#b0bec5", linewidth=1.0))

    extras = {name.lower() for name in DEVICE_EXTRA_LABELS.get(device, [])}
    box_width, box_height = 0.22, 0.14
    for idx, row in enumerate(rows):
        position_label = row.get("Position") or row.get("Channel") or f"Ch {idx + 1}"
        normalized = _normalize_position_label(position_label)
        coords = _STANDARD_1020_XY.get(normalized)
        if coords is not None:
            x = float(coords[0]) * _STANDARD_XY_SCALE
            y = float(coords[1]) * _STANDARD_XY_SCALE
        else:
            angle = _position_angle(position_label, idx)
            radius = 0.85 if row.get("Active") else 0.65
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
        ch_label = row.get("Channel") or position_label
        lower_name = str(ch_label).lower()
        is_active = bool(row.get("Active"))
        is_reference = "ref" in lower_name
        is_ground = "gnd" in lower_name or "ground" in lower_name
        if is_reference:
            fill_color = "#1e88e5"
        elif is_ground:
            fill_color = "#263238"
        elif is_active or lower_name in extras:
            fill_color = "#43a047"
        else:
            fill_color = "#ffd54f"
        edge_color = "#0f172a" if is_reference or is_ground else "#37474f"
        rect = FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            facecolor=fill_color,
            edgecolor=edge_color,
            linewidth=3,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y + box_height * 0.15,
            position_label,
            ha="center",
            va="center",
            fontsize=9,
            color="white" if fill_color in {"#1e88e5", "#263238", "#43a047"} else "#1f2937",
            weight="bold",
            zorder=4,
        )
        ax.text(
            x,
            y - box_height * 0.25,
            f"{idx + 1}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if fill_color in {"#1e88e5", "#263238"} else "#424242",
            zorder=4,
        )
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(f"{device} electrode map", fontsize=12)
    return fig


def _render_participant_card(participant: Dict[str, Any]) -> None:
    st.markdown(PARTICIPANT_CARD_STYLE, unsafe_allow_html=True)
    info_rows = [
        ("🆔", "Code", participant.get("Code") or "—"),
        ("🔤", "Initials", participant.get("Initials") or "—"),
        ("🎂", "Age", participant.get("Age") or "—"),
        ("⚧", "Gender", participant.get("Gender") or "—"),
        ("✋", "Dominant hand", participant.get("DominantHand") or "—"),
    ]
    info_html = "".join(
        f"<div class='field'><div class='icon'>{icon}</div>"
        f"<div><div class='label'>{label}</div><div class='value'>{html.escape(str(value))}</div></div></div>"
        for icon, label, value in info_rows
    )
    notes_text = str(participant.get("Notes") or "").strip()
    notes_html = ""
    if notes_text:
        notes_html = f"<div class='notes'>📝 {html.escape(notes_text)}</div>"
    card_html = f"<div class='participant-card'><div class='avatar'>🧑</div>{info_html}{notes_html}</div>"
    st.markdown(card_html, unsafe_allow_html=True)


def render_general_form() -> Dict[str, Any]:
    general = st.session_state["general_form"]
    method_field = next(field for field in GENERAL_SCHEMA if field.name == "Method")
    device_field = next(field for field in GENERAL_SCHEMA if field.name == "Device")

    with st.expander("Session configuration", expanded=True):
        form_col, viz_col = st.columns((3, 2))
        left, right = form_col.columns(2)
        method_options = method_field.options or sorted(METHOD_SCHEMAS.keys())
        method_value = resolve_choice(method_options, general.get("Method"))
        method = left.selectbox(
            "Method",
            options=method_options,
            index=method_options.index(method_value),
            help=method_field.tooltip or None,
        )
        device_options = sorted(set((device_field.options or []) + SUPPORTED_EXTRA_DEVICES))
        device_value = resolve_choice(device_options, general.get("Device"))
        device = right.selectbox(
            "Device",
            options=device_options,
            index=device_options.index(device_value),
            help=device_field.tooltip or None,
        )
        method_full = METHOD_FULL_NAMES.get(method)
        description = METHOD_DESCRIPTIONS.get(method)
        if method_full or description:
            caption_parts = [method_full or ""]
            if description:
                caption_parts.append(description)
            left.caption(" · ".join(part for part in caption_parts if part))
        general["Method"] = method
        general["Device"] = device

        other_fields = [field for field in GENERAL_SCHEMA if field.name not in {"Method", "Device"}]
        cols = form_col.columns(2)
        for idx, field in enumerate(other_fields):
            target = cols[idx % 2]
            key = f"general_{field.name}"
            current = general.get(field.name)
            if field.kind == "dropdown":
                options = field.options or [""]
                resolved = resolve_choice(options, current)
                value = target.selectbox(
                    field.name,
                    options=options,
                    index=options.index(resolved),
                    help=field.tooltip or None,
                    key=key,
                )
            elif field.kind == "numeric":
                value = render_numeric_input(target, field, current, key)
            else:
                value = target.text_input(field.name, value=current or "", help=field.tooltip or None, key=key)
            general[field.name] = value

        with viz_col:
            spacer_col, snap_col = viz_col.columns([1, 5])
            with snap_col:
                st.caption("Configuration snapshot")
                render_config_snapshot(general.get("Method", ""), general.get("Device", ""), general)
    return dict(general)


def render_device_config(device: str) -> Dict[str, Any]:
    schema = DEVICE_CONFIG_SCHEMA.get(device)
    if not schema:
        return {}
    device_forms = st.session_state.setdefault("device_forms", {})
    form_state = device_forms.setdefault(device, device_default_values(device))

    with st.expander(f"{device} device settings", expanded=True):
        cols = st.columns(2)
        for idx, field in enumerate(schema):
            target = cols[idx % 2]
            key = f"device_{device}_{field['name']}"
            current = form_state.get(field["name"], field.get("default"))
            if field["kind"] == "number":
                fallback = field.get("default", 0.0)
                numeric = coerce_number(current)
                value_default = float(numeric if numeric is not None else fallback or 0.0)
                kwargs: Dict[str, Any] = {
                    "value": value_default,
                    "step": float(field.get("step", 0.5)),
                    "help": field.get("help"),
                    "key": key,
                }
                if field.get("min") is not None:
                    kwargs["min_value"] = float(field["min"])
                if field.get("max") is not None:
                    kwargs["max_value"] = float(field["max"])
                value = target.number_input(field["label"], **kwargs)
            else:
                value = target.text_input(
                    field["label"],
                    value=str(current or ""),
                    help=field.get("help"),
                    placeholder=field.get("placeholder"),
                    key=key,
                )
                value = value.strip()
            form_state[field["name"]] = value
    return dict(form_state)


def render_method_form(method: str) -> Dict[str, Any]:
    schema = METHOD_SCHEMAS.get(method, [])
    method_state = st.session_state["method_forms"].setdefault(method, default_values(schema))
    if not schema:
        st.info(f"No dedicated parameter schema found for {method}.")
        return {}

    with st.expander(f"{method} parameters", expanded=True):
        cols = st.columns(2)
        for idx, field in enumerate(schema):
            target = cols[idx % 2]
            key = f"{method}_{field.name}"
            current = method_state.get(field.name)
            if field.kind == "dropdown":
                options = field.options or [""]
                resolved = resolve_choice(options, current)
                value = target.selectbox(field.name, options=options, index=options.index(resolved), help=field.tooltip or None, key=key)
            elif field.kind == "numeric":
                value = render_numeric_input(target, field, current, key)
            else:
                value = target.text_input(field.name, value=current or "", help=field.tooltip or None, key=key)
            method_state[field.name] = value
    return dict(method_state)


def ensure_channel_rows(device: str, existing: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    base_count = DEVICE_DEFAULT_CHANNELS.get(device, 8)
    extras = DEVICE_EXTRA_LABELS.get(device, [])
    existing = existing or []
    existing_map = {
        (row.get("Channel") or row.get("Label")): dict(row)
        for row in existing
        if row.get("Channel") or row.get("Label")
    }
    default_positions = DEVICE_POSITION_DEFAULTS.get(device, [])

    def base_row(label: str, is_extra: bool, index: Optional[int] = None) -> Dict[str, Any]:
        default_rubric = ELECTRODE_RUBRICS[0] if ELECTRODE_RUBRICS else ""
        default_model = ELECTRODE_LIBRARY.get(default_rubric, ["Unknown"])[0] if ELECTRODE_RUBRICS else ""
        position_label = label.replace(" ", "")
        if index is not None and index < len(default_positions):
            position_label = default_positions[index]
        row = {
            "Channel": label,
            "Position": position_label,
            "Rubrik": default_rubric,
            "Model": default_model,
            "Impedance": 0.0,
            "Active": True if is_extra else False,
        }
        return row

    rows: List[Dict[str, Any]] = []
    for label in extras:
        row = existing_map.get(label, base_row(label, True))
        row["Active"] = True
        rows.append(row)

    for idx in range(base_count):
        label = f"Ch {idx + 1}"
        row = existing_map.get(label, base_row(label, False, idx))
        rows.append(row)

    return rows


def render_channel_editor(device: str) -> List[Dict[str, Any]]:
    channel_state = st.session_state["channel_tables"]
    rows = ensure_channel_rows(device, channel_state.get(device))
    with st.expander(f"Electrodes ({device})", expanded=True):
        table_col, map_col = st.columns((2, 1))
        with table_col:
            edited = st.data_editor(
                rows,
                num_rows="fixed",
                hide_index=True,
                key=f"channels_{device}",
                column_config={
                    "Channel": st.column_config.TextColumn("Channel", disabled=True, width="small"),
                    "Position": st.column_config.TextColumn(
                        "Electrode / Position",
                        help="10-20 label or custom montage description",
                        width="medium",
                    ),
                    "Rubrik": st.column_config.SelectboxColumn(
                        "Electrode type (Rubrik)",
                        options=ELECTRODE_RUBRICS,
                        width="medium",
                    ),
                    "Model": st.column_config.SelectboxColumn("Model", options=ELECTRODE_MODELS, width="large"),
                    "Impedance": st.column_config.NumberColumn(
                        "Impedance (kΩ)",
                        min_value=0.0,
                        step=0.5,
                        format="%.1f",
                    ),
                    "Active": st.column_config.CheckboxColumn("Use channel"),
                },
            )
        with map_col:
            st.caption("Montage preview")
            st.markdown("<div style='margin-top: 0.75rem'></div>", unsafe_allow_html=True)
            fig = _electrode_map_figure(device, edited)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Active selections glow; reference/ground remain pinned.")
        extras = set(DEVICE_EXTRA_LABELS.get(device, []))
        for row in edited:
            if row["Channel"] in extras:
                row["Active"] = True
        channel_state[device] = edited
    return edited


def render_participant_form() -> Dict[str, Any]:
    participant = st.session_state["participant"]
    with st.expander("Participant / proband information", expanded=True):
        form_col, viz_col = st.columns((4, 1))
        with form_col:
            cols = form_col.columns(2)
            participant["Code"] = cols[0].text_input(
                "Participant code", value=participant.get("Code", ""), placeholder="e.g. VEP_023"
            )
            participant["Initials"] = cols[1].text_input("Initials", value=participant.get("Initials", ""))
            cols = form_col.columns(3)
            age_number = coerce_number(participant.get("Age"))
            age_default = int(age_number) if isinstance(age_number, (int, float)) and age_number > 0 else 0
            participant["Age"] = cols[0].number_input("Age", min_value=0, max_value=110, value=age_default)
            gender_value = resolve_choice(GENDER_OPTIONS, participant.get("Gender"))
            hand_value = resolve_choice(HANDEDNESS_OPTIONS, participant.get("DominantHand"))
            participant["Gender"] = cols[1].selectbox(
                "Gender", options=GENDER_OPTIONS, index=GENDER_OPTIONS.index(gender_value)
            )
            participant["DominantHand"] = cols[2].selectbox(
                "Dominant hand", options=HANDEDNESS_OPTIONS, index=HANDEDNESS_OPTIONS.index(hand_value)
            )
            participant["Notes"] = form_col.text_area("Session notes", value=participant.get("Notes", ""), height=80)
        with viz_col:
            spacer_col, snap_col = viz_col.columns([1, 5])
            with snap_col:
                st.caption("Participant snapshot")
                _render_participant_card(participant)
    return dict(participant)


def build_channels(device: str) -> List[Dict[str, Any]]:
    rows = st.session_state["channel_tables"].get(device) or ensure_channel_rows(device)
    extras = set(DEVICE_EXTRA_LABELS.get(device, []))
    active_rows: List[Dict[str, Any]] = []
    for row in rows:
        channel_name = row.get("Channel") or row.get("Label")
        if not channel_name:
            continue
        is_active = bool(row.get("Active")) or channel_name in extras
        entry = {
            "Channel": channel_name,
            "Position": row.get("Position") or channel_name.replace(" ", ""),
            "Rubrik": row.get("Rubrik") or row.get("Rubric") or (ELECTRODE_RUBRICS[0] if ELECTRODE_RUBRICS else ""),
            "Model": row.get("Model") or (ELECTRODE_MODELS[0] if ELECTRODE_MODELS else ""),
            "Impedance": coerce_number(row.get("Impedance")),
            "Active": bool(is_active),
        }
        active_rows.append(entry)
    return active_rows


def build_metadata(participant: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {key: value for key, value in participant.items() if value not in ("", None, 0)}
    return {"Participant": cleaned} if cleaned else {}


def assemble_params(
    general: Dict[str, Any],
    method_values: Dict[str, Any],
    participant: Dict[str, Any],
    device_specific: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "Method": general["Method"],
        "Device": general["Device"],
        "Parameters": {},
    }
    for field in GENERAL_SCHEMA:
        if field.name in {"Method", "Device"}:
            continue
        value = convert_value(field, general.get(field.name))
        if value not in (None, "", []):
            params["Parameters"][field.name] = value
    for field in METHOD_SCHEMAS.get(params["Method"], []):
        value = convert_value(field, method_values.get(field.name))
        if value not in (None, "", []):
            params["Parameters"][field.name] = value
    if device_specific:
        for key, value in device_specific.items():
            if value in (None, "", []):
                continue
            params["Parameters"][key] = value
            for alias in DEVICE_FIELD_ALIASES.get(key, []):
                params["Parameters"][alias] = value

    channels = build_channels(params["Device"])
    if channels:
        params["Channels"] = channels

    metadata = build_metadata(participant)
    if metadata:
        params["Metadata"] = metadata

    return params


def validate_params(params: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    method = params.get("Method")
    device = params.get("Device") or ""
    parameters = params.get("Parameters", {})
    if not method:
        issues.append("Method must be selected.")
    if not device:
        issues.append("Device must be selected.")
    if "fs" not in parameters or not parameters["fs"]:
        issues.append("Sampling rate (fs) is required.")
    if parameters.get("RecordingTime") in (0, None):
        issues.append("RecordingTime should be greater than zero.")
    if (device.lower() if device else "") not in {"actichamp", "unicorn", "lsl", "offline", "dummy"}:
        issues.append(f"Device '{device}' is not yet supported by the Python pipeline.")
    if device and device.lower() == "unicorn":
        port = (
            parameters.get("UNICORNPort")
            or parameters.get("UNICORNAddress")
            or parameters.get("UnicornPort")
            or parameters.get("UnicornAddress")
        )
        if not port:
            issues.append("UNICORN configuration requires a serial port / address.")
    return issues


def params_to_json(params: Dict[str, Any]) -> str:
    def default(obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(params, indent=2, default=default)


@st.cache_data
def list_saved_sessions(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    return sorted([path for path in base_dir.iterdir() if path.is_dir()], reverse=True)


def run_pipeline_once(params: Dict[str, Any], save_dir: Path) -> tuple[Dict[str, Any], Optional[Path]]:
    saver = SaveManager(save_dir)
    provider_called = {"done": False}
    captured: Dict[str, Any] = {}

    def provider(_: Optional[Dict[str, Any]]):
        if provider_called["done"]:
            return None
        provider_called["done"] = True
        return params

    def save_and_capture(run_params: Dict[str, Any]) -> None:
        captured["params"] = run_params
        saver(run_params)
        captured["path"] = getattr(saver, "last_target_dir", None)

    hooks = PipelineHooks(params_provider=provider, save_callback=save_and_capture, should_continue=lambda _: False)
    MeasurementPipeline(hooks=hooks).run()
    return captured.get("params", params), captured.get("path")


def _load_session_contents(session_dir: Path) -> tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    params_path = session_dir / "params.json"
    data_path = session_dir / "data.npz"
    params_content: Optional[Dict[str, Any]] = None
    data_array: Optional[np.ndarray] = None
    if params_path.exists():
        try:
            params_content = json.loads(params_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to load params.json from {session_dir.name}: {exc}")
    else:
        st.warning(f"{params_path.name} missing in {session_dir.name}.")
    if data_path.exists():
        data_array = _load_npz_array(data_path)
    return params_content, data_array


def render_saved_sessions(base_dir: Path) -> tuple[Optional[tuple[str, Dict[str, Any], Optional[np.ndarray]]], List[tuple[str, Dict[str, Any], Optional[np.ndarray]]]]:
    st.subheader("Saved sessions")
    refresh = st.button("Refresh list")
    if refresh:
        list_saved_sessions.clear()
    sessions = list_saved_sessions(base_dir)
    if not sessions:
        st.info("No sessions saved yet.")
        return None, []

    primary = st.selectbox("Session directory", options=sessions, format_func=lambda p: p.name)
    compare_selection = st.multiselect(
        "Sessions to compare / overlay",
        options=sessions,
        default=[primary] if primary else [],
        format_func=lambda p: p.name,
    )

    params_content: Optional[Dict[str, Any]] = None
    data_array: Optional[np.ndarray] = None
    if primary:
        params_content, data_array = _load_session_contents(primary)

    cols = st.columns(2)
    with cols[0]:
        if params_content:
            st.caption("params.json")
            st.json(params_content)
            if st.button("Load session into editor", key=f"load_session_{primary.name}"):
                load_params_into_state(normalize_params(params_content), data_array)
                st.session_state["last_results"] = {
                    "label": primary.name,
                    "params": normalize_params(params_content),
                    "data": data_array,
                }
                st.success("Session loaded. Review settings before running.")
        else:
            st.warning("params.json missing.")
    with cols[1]:
        if data_array is not None:
            st.caption("data.npz (arrays and shapes)")
            st.write({"data": data_array.shape})
        else:
            st.info("data.npz missing.")

    primary_payload: Optional[tuple[str, Dict[str, Any], Optional[np.ndarray]]] = None
    chart_button = st.button("Show charts for selected session", key="show_primary_charts")
    if chart_button and params_content:
        primary_payload = (primary.name, normalize_params(params_content), data_array)

    compare_payloads: List[tuple[str, Dict[str, Any], Optional[np.ndarray]]] = []
    compare_button = st.button("Compare selected sessions", key="compare_sessions")
    if compare_button:
        for path in compare_selection:
            params_loaded, data_loaded = _load_session_contents(path)
            if params_loaded:
                compare_payloads.append((path.name, normalize_params(params_loaded), data_loaded))
    return primary_payload, compare_payloads


def handle_upload() -> None:
    with st.sidebar.expander("Import config / Params", expanded=False):
        uploaded = st.file_uploader(
            "Load JSON/TOML config or params.json",
            type=["json", "toml", "tml"],
            key="config_uploader",
        )
        data_upload = st.file_uploader(
            "Attach data file (.npz)",
            type=["npz"],
            key="data_uploader",
        )

        if uploaded:
            config_bytes = uploaded.read()
            suffix = Path(uploaded.name).suffix.lower()
            try:
                if suffix in {".toml", ".tml"}:
                    config = tomllib.loads(config_bytes.decode("utf-8"))
                else:
                    config = json.loads(config_bytes.decode("utf-8"))
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to parse uploaded config: {exc}")
            else:
                params = normalize_params(config)
                load_params_into_state(params)
                st.success(f"Imported parameters from '{uploaded.name}'.")

        if data_upload:
            data_array = _load_npz_array(data_upload)
            if data_array is not None:
                st.session_state["imported_data"] = data_array
                st.session_state["use_imported_data"] = True
                st.success(f"Attached data from '{data_upload.name}'.")


def main() -> None:
    st.set_page_config(page_title="cortipy UI", layout="wide")
    st.markdown(
        """
<style>
div[data-testid="stExpander"] > details {
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background-color: #ffffff;
}
div[data-testid="stExpander"] > details > summary {
    background-color: #eef2ff;
}
div[data-testid="stExpander"] > details > summary:hover {
    background-color: #e0e7ff;
}
div[data-testid="stExpander"] > details > div[role="group"] {
    padding-top: 0.5rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    st.title("cortipy – EEG Measurement UI")
    ensure_state()

    sidebar = st.sidebar
    sidebar.header("Run controls")
    handle_upload()
    default_save = sidebar.text_input("Save directory", value=str(DEFAULT_SAVE_DIR))
    simulate = sidebar.checkbox("Simulate run (no device)", value=False)
    imported_data = st.session_state.get("imported_data")
    default_use_imported = st.session_state.get("use_imported_data", False) or bool(imported_data)
    use_imported_data = sidebar.checkbox(
        "Use imported data for offline replay",
        value=default_use_imported and imported_data is not None,
        disabled=imported_data is None,
    )
    st.session_state["use_imported_data"] = use_imported_data and imported_data is not None
    start_button = sidebar.button("Start measurement")

    session_tab, electrodes_tab, preview_tab, charts_tab, saved_tab = st.tabs(
        ["Session configuration", "Electrodes", "Preview", "Charts", "Saved sessions"]
    )

    with session_tab:
        general_values = render_general_form()
        device_values = render_device_config(general_values["Device"])
        method_values = render_method_form(general_values["Method"])
        participant_values = render_participant_form()

    with electrodes_tab:
        render_channel_editor(general_values["Device"])

    assembled_params = assemble_params(general_values, method_values, participant_values, device_values)
    validation_issues = validate_params(assembled_params)

    with preview_tab:
        if validation_issues:
            st.warning(" • ".join(validation_issues))
        st.json(assembled_params)
        st.download_button(
            "Download params.json",
            data=params_to_json(assembled_params),
            file_name=f"{assembled_params['Method']}_params.json",
            mime="application/json",
        )

    with charts_tab:
        current_results = st.session_state.get("last_results")
        chart_label: Optional[str] = None
        chart_params: Optional[Dict[str, Any]] = None
        chart_data: Optional[np.ndarray] = None

        if current_results:
            chart_label = current_results.get("label", "Last run")
            chart_params = current_results.get("params") or assembled_params
            chart_data = current_results.get("data")
            st.caption("Showing charts from the last run or loaded session.")
        elif imported_data is not None:
            chart_label = "Imported data"
            chart_params = assembled_params
            chart_data = imported_data
            st.caption("Using imported data with current parameters.")

        if chart_label and chart_params is not None:
            render_chart_section(chart_label, chart_params, chart_data)
        else:
            st.info("Run a measurement or load a saved session to see charts.")

    with saved_tab:
        primary_payload, compare_payloads = render_saved_sessions(Path(default_save).expanduser())
        if primary_payload:
            label, params_loaded, data_loaded = primary_payload
            render_chart_section(label, params_loaded, data_loaded)
        if compare_payloads:
            render_comparison_charts(compare_payloads)

    if start_button:
        if validation_issues:
            st.error("Please fix the highlighted configuration issues before starting a run.")
            return
        save_dir = Path(default_save).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        params_to_run = dict(assembled_params)
        params_to_run.pop("Evaluation", None)
        params_to_run.pop("data", None)
        if simulate:
            fs = int(assembled_params["Parameters"].get("fs", 250))
            n_channels = int(assembled_params["Parameters"].get("NumberEEGChannels", len(assembled_params.get("Channels", [])) or 8))
            params_to_run["data"] = np.zeros((fs, n_channels))
            saved_path = SaveManager(save_dir)(params_to_run)
            st.session_state["last_results"] = {
                "label": getattr(saved_path, "name", "Simulated run"),
                "params": params_to_run,
                "data": params_to_run.get("data"),
            }
            st.success("Simulated data saved. Charts available in the Charts tab.")
        else:
            imported_data = st.session_state.get("imported_data")
            use_imported = st.session_state.get("use_imported_data", False)
            if use_imported and imported_data is not None:
                params_to_run["Device"] = "Offline"
                params_to_run["data"] = imported_data
            elif use_imported and imported_data is None:
                st.error("No imported data attached. Upload a data file or select a saved session first.")
                return
            try:
                run_params, saved_path = run_pipeline_once(params_to_run, save_dir)
                st.session_state["last_results"] = {
                    "label": getattr(saved_path, "name", "Last run"),
                    "params": run_params,
                    "data": run_params.get("data"),
                }
                st.success("Measurement finished and saved. Charts available in the Charts tab.")
            except Exception as exc:  # pragma: no cover
                st.error(f"Measurement failed: {exc}")


if __name__ == "__main__":
    main()
