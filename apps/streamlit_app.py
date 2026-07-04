"""Streamlit UI for configuring and running cortipy sessions."""

from __future__ import annotations

import json
import copy
import logging
import sys
import time
from dataclasses import dataclass
import html
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
import streamlit as st
try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None
try:
    from streamlit_plotly_events import plotly_events
except Exception:  # pragma: no cover
    plotly_events = None
try:
    from serial.tools import list_ports
except Exception:  # pragma: no cover
    list_ports = None
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:  # pragma: no cover
    add_script_run_ctx = None
    get_script_run_ctx = None

try:
    import mne  # type: ignore
except Exception:  # pragma: no cover
    mne = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "streamlit_app.log"


def _configure_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_attached = any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == LOG_PATH
        for handler in root_logger.handlers
    )
    if not file_attached:
        file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    stream_attached = any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers)
    if not stream_attached:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    logger = logging.getLogger(__name__)

    def _hook(exc_type, exc, tb):
        logger.error("Unhandled exception in Streamlit app", exc_info=(exc_type, exc, tb))
        return sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook
    return logger


LOGGER = _configure_logging()


def _tail_log(path: Path, n: int = 60) -> List[str]:
    """Return the last ``n`` lines of the app log (empty list if unreadable)."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception:
        return []

from cortipy import MeasurementPipeline  # noqa: E402
from cortipy.core.pipeline import PipelineHooks  # noqa: E402
from cortipy.devices import DeviceFactory, DeviceInterface  # noqa: E402
from cortipy.ui import SaveManager, normalize_params  # noqa: E402
from cortipy.ui_streamlit.styles import inject_global_styles  # noqa: E402

DEFAULT_SAVE_DIR = Path.cwd() / "cortipy_runs"
SCHEMA_DIR = ROOT / "apps" / "assets" / "ParameterJSON"
SUPPORTED_EXTRA_DEVICES = ["LSL", "Offline", "Dummy"]
VIEW_OPTIONS = ["Session configuration", "Electrodes", "Live preview", "Preview", "Charts", "Saved sessions"]
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


# Field schema + value coercion live in cortipy.ui_streamlit.fields (modularization).
from cortipy.ui_streamlit.fields import (  # noqa: E402
    FieldSchema,
    default_values,
    resolve_choice,
    is_integer_field,
    coerce_number,
    convert_value,
)


# Chart data model + renderer live in cortipy.ui_streamlit.charts (modularization).
from cortipy.ui_streamlit.charts import (  # noqa: E402
    ChartSeries,
    ChartData,
    downsample_series as _downsample_series,
    render_chart,
    _chart_from_raw_data,
    _chart_from_psd,
    _chart_from_alpha_power,
    _chart_from_eval_psd,
    _chart_from_eval_fft,
    _chart_from_average_signals,
    _chart_from_metric_vector,
)
from cortipy.ui_streamlit.electrodes import (  # noqa: E402
    actichamp_channel_count as _actichamp_channel_count,
    bump_channel_editor_revision as _bump_channel_editor_revision,
    map_impedances_to_channels as _map_impedances_to_channels,
    has_measured_impedance as _has_measured_impedance,
    impedance_range_kohm as _impedance_range_kohm,
)


@dataclass(frozen=True)
class SidebarControls:
    default_save: str
    simulate: bool
    live_view_enabled: bool
    live_view_window: int
    start_button: bool


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

# Device / montage / field configuration lives in cortipy.ui_streamlit.constants (modularization).
from cortipy.ui_streamlit.constants import (  # noqa: E402
    DEVICE_DEFAULT_CHANNELS,
    DEVICE_EXTRA_LABELS,
    DEVICE_FS_OPTIONS,
    DEVICE_POSITION_DEFAULTS,
    normalize_position_label as _normalize_position_label,
    STANDARD_POSITION_ORDER as _STANDARD_POSITION_ORDER,
    POSITION_ANGLE_LOOKUP as _POSITION_ANGLE_LOOKUP,
)

# 10-20 scalp coordinates + lookup live in cortipy.ui_streamlit.coords (first modularization step).
from cortipy.ui_streamlit.coords import (  # noqa: E402
    TEN_TWENTY_COORDS,
    channel_default_coords as _channel_default_coords,
)

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
:root {
    --card-bg: linear-gradient(135deg, #f3f6ff 0%, #fff7f0 100%);
    --card-border: #dbe2ef;
    --card-shadow: rgba(15, 23, 42, 0.08);
    --label-color: #6b7280;
    --value-color: #111827;
    --notes-color: #374151;
    --divider-color: rgba(255, 255, 255, 0.6);
}
@media (prefers-color-scheme: dark) {
    :root {
        --card-bg: linear-gradient(135deg, #111827 0%, #0b1220 100%);
        --card-border: #1f2937;
        --card-shadow: rgba(0, 0, 0, 0.4);
        --label-color: #9ca3af;
        --value-color: #e5e7eb;
        --notes-color: #cbd5e1;
        --divider-color: rgba(255, 255, 255, 0.15);
        --expander-bg: #0f172a;
        --expander-border: #1f2937;
        --expander-summary: #111827;
        --expander-summary-hover: #152238;
        --expander-text: #e5e7eb;
    }
}
.snapshot-panel {
    background: var(--card-bg);
    color: var(--value-color);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    box-shadow: 0 3px 10px var(--card-shadow);
}
.snapshot-panel .title {
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.snapshot-panel .desc {
    color: var(--label-color);
    margin-bottom: 0.4rem;
}
.snapshot-panel .line {
    margin-bottom: 0.1rem;
}
.snapshot-panel .stats {
    margin-top: 0.35rem;
    color: var(--label-color);
}
.participant-card {
    background: var(--card-bg);
    border-radius: 16px;
    border: 1px solid var(--card-border);
    padding: 0.85rem 1rem;
    box-shadow: 0 4px 12px var(--card-shadow);
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
    color: var(--label-color);
    margin-bottom: 0.05rem;
}
.participant-card .value {
    font-size: 0.95rem;
    color: var(--value-color);
    font-weight: 600;
}
.participant-card .notes {
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: var(--notes-color);
    padding-top: 0.4rem;
    border-top: 1px solid var(--divider-color);
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


MANUAL_UNICORN_PORT_OPTION = "__manual_unicorn_port__"


def serial_port_options() -> List[tuple[str, str]]:
    if list_ports is None:
        return []
    seen = set()
    options: List[tuple[str, str]] = []
    for info in list_ports.comports():
        device = getattr(info, "device", None) or getattr(info, "name", None)
        if not device or device in seen:
            continue
        description = getattr(info, "description", "") or getattr(info, "product", "")
        manufacturer = getattr(info, "manufacturer", "")
        serial_no = getattr(info, "serial_number", "") or getattr(info, "serial", "")
        vid = getattr(info, "vid", None)
        pid = getattr(info, "pid", None)
        usb_id = f"{vid:04X}:{pid:04X}" if isinstance(vid, int) and isinstance(pid, int) else ""
        details_parts = []
        if description and description != device:
            details_parts.append(description)
        if manufacturer:
            details_parts.append(manufacturer)
        if usb_id:
            details_parts.append(usb_id)
        if serial_no:
            details_parts.append(f"SN {serial_no}")
        details = ", ".join(details_parts)
        label = f"{device} – {details}" if details else device
        options.append((device, label))
        seen.add(device)
    return sorted(options, key=lambda item: item[0])


def render_unicorn_port_input(target, field: Dict[str, Any], current: Any, key: str) -> str:
    port_entries = serial_port_options()
    labels = {port: label for port, label in port_entries}
    options: List[str] = [port for port, _ in port_entries]

    current_str = str(current) if current not in (None, "") else ""
    if current_str and current_str not in options:
        options.append(current_str)
        labels[current_str] = f"{current_str} (saved)"

    options.append(MANUAL_UNICORN_PORT_OPTION)
    default_choice = current_str if current_str in options else options[0]

    selection = target.selectbox(
        field["label"],
        options=options,
        index=options.index(default_choice) if options else 0,
        format_func=lambda value: "Manual entry" if value == MANUAL_UNICORN_PORT_OPTION else labels.get(value, value),
        help=field.get("help"),
        key=key,
    )

    if selection == MANUAL_UNICORN_PORT_OPTION:
        manual_value = target.text_input(
            "Custom UNICORN port",
            value=current_str if current_str and current_str not in labels else "",
            help="Enter the UNICORN serial/Bluetooth port manually.",
            key=f"{key}_manual",
        )
        return manual_value.strip()
    return selection


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


def _autoevaluate_if_needed(params: Dict[str, Any], data: Optional[np.ndarray]) -> Dict[str, Any]:
    # If Evaluation already exists (even empty), never auto-run evaluators.
    if "Evaluation" in params:
        return params.get("Evaluation") or {}

    if data is None:
        return {}

    data_array = np.asarray(data)
    if data_array.size == 0 or (data_array.ndim >= 2 and data_array.shape[1] == 0):
        LOGGER.warning("Skipping auto-evaluation: empty data buffer", extra={"shape": data_array.shape})
        return {}

    evaluation: Dict[str, Any] = {}
    method_name = str(params.get("Method", "")).lower()

    evaluator_map = {
        "alpha": "AlphaEvaluator",
        "ssvep": "SsvepEvaluator",
        "assr": "AssrEvaluator",
        "vep": "VepEvaluator",
        "p300": "P300Evaluator",
        "bera": "BeraEvaluator",
    }
    evaluator_name = evaluator_map.get(method_name)
    if not evaluator_name:
        return evaluation

    try:  # pragma: no cover - runtime convenience
        from cortipy.core.context import ModuleContext

        if evaluator_name == "AlphaEvaluator":
            from cortipy.evaluation.alpha import AlphaEvaluator as EvalCls
        elif evaluator_name == "SsvepEvaluator":
            from cortipy.evaluation.ssvep import SsvepEvaluator as EvalCls
        elif evaluator_name == "AssrEvaluator":
            from cortipy.evaluation.assr import AssrEvaluator as EvalCls
        elif evaluator_name == "VepEvaluator":
            from cortipy.evaluation.vep import VepEvaluator as EvalCls
        elif evaluator_name == "P300Evaluator":
            from cortipy.evaluation.p300 import P300Evaluator as EvalCls
        elif evaluator_name == "BeraEvaluator":
            from cortipy.evaluation.bera import BeraEvaluator as EvalCls
        else:
            return evaluation

        temp_params = dict(params)
        temp_params["Parameters"] = dict(params.get("Parameters", {}))
        temp_params["data"] = data
        ctx = ModuleContext(temp_params)
        EvalCls(show_plots=False).evaluate(ctx)
        evaluation = ctx.params.get("Evaluation") or {}
        params["Evaluation"] = evaluation
    except Exception as exc:
        LOGGER.warning("Evaluation generation failed for method %s: %s", method_name, exc)
    return evaluation


def _render_evaluation_figures(params: Dict[str, Any], data: Optional[np.ndarray]) -> List[plt.Figure]:
    if data is None:
        return []
    method_name = str(params.get("Method", "")).lower()
    evaluator_map = {
        "alpha": "AlphaEvaluator",
        "ssvep": "SsvepEvaluator",
        "assr": "AssrEvaluator",
        "vep": "VepEvaluator",
        "p300": "P300Evaluator",
        "bera": "BeraEvaluator",
    }
    evaluator_name = evaluator_map.get(method_name)
    if not evaluator_name:
        return []

    try:  # pragma: no cover - UI rendering only
        from cortipy.core.context import ModuleContext

        if evaluator_name == "AlphaEvaluator":
            from cortipy.evaluation.alpha import AlphaEvaluator as EvalCls
        elif evaluator_name == "SsvepEvaluator":
            from cortipy.evaluation.ssvep import SsvepEvaluator as EvalCls
        elif evaluator_name == "AssrEvaluator":
            from cortipy.evaluation.assr import AssrEvaluator as EvalCls
        elif evaluator_name == "VepEvaluator":
            from cortipy.evaluation.vep import VepEvaluator as EvalCls
        elif evaluator_name == "P300Evaluator":
            from cortipy.evaluation.p300 import P300Evaluator as EvalCls
        elif evaluator_name == "BeraEvaluator":
            from cortipy.evaluation.bera import BeraEvaluator as EvalCls
        else:
            return []

        temp_params = copy.deepcopy(params)
        temp_params["data"] = np.asarray(data)
        ctx = ModuleContext(temp_params)
        EvalCls(show_plots=True).evaluate(ctx)
        fig_nums = list(plt.get_fignums())
        LOGGER.debug(
            "Evaluation figures captured",
            extra={"method": method_name, "fig_nums": fig_nums},
        )
        figures = [plt.figure(num) for num in fig_nums]
        return figures
    except Exception as exc:
        LOGGER.warning("Evaluation figure rendering failed for %s: %s", method_name, exc)
    return []


def collect_chart_data(label: str, params: Dict[str, Any], data: Optional[np.ndarray], aggregate: bool = False) -> Dict[str, ChartData]:
    charts: Dict[str, ChartData] = {}
    method_name = str(params.get("Method", "")).lower()
    if data is not None:
        raw_chart = _chart_from_raw_data(label, params, data, aggregate=aggregate)
        if raw_chart:
            charts[raw_chart.key] = raw_chart
        psd_chart = _chart_from_psd(label, params, data, aggregate=aggregate)
        if psd_chart:
            charts[psd_chart.key] = psd_chart

    evaluation = _autoevaluate_if_needed(params, data)
    if isinstance(evaluation, dict):
        alpha_chart = _chart_from_alpha_power(label, evaluation.get("alphaPower", {}), aggregate=aggregate)
        if alpha_chart:
            charts[alpha_chart.key] = alpha_chart
        eval_psd_chart = _chart_from_eval_psd(label, evaluation.get("PSD", {}), aggregate=aggregate)
        if eval_psd_chart:
            charts[eval_psd_chart.key] = eval_psd_chart
        fft_chart = _chart_from_eval_fft(label, evaluation.get("fft", {}), aggregate=aggregate)
        if fft_chart:
            charts[fft_chart.key] = fft_chart
        avg_chart = _chart_from_average_signals(label, evaluation.get("average_signals", {}), aggregate=aggregate)
        if avg_chart:
            charts[avg_chart.key] = avg_chart

        # Peak metrics for VEP-like evaluations
        for peak_name in ("P100", "N75", "N135"):
            peak_block = evaluation.get(peak_name)
            if isinstance(peak_block, dict):
                peak_vals = peak_block.get("peak_values")
                peak_times = peak_block.get("peak_times")
                val_chart = _chart_from_metric_vector(label, f"{peak_name} amplitude", peak_vals, unit="Amplitude (uV)")
                if val_chart:
                    charts.setdefault(f"metric_{peak_name}_amp", val_chart)
                time_chart = _chart_from_metric_vector(
                    label,
                    f"{peak_name} latency",
                    peak_times,
                    unit="Time (ms)",
                )
                if time_chart:
                    charts.setdefault(f"metric_{peak_name}_latency", time_chart)

        # Scalar-within-dict metrics (e.g., tRes.h)
        t_res = evaluation.get("tRes")
        if isinstance(t_res, dict) and "h" in t_res:
            tres_chart = _chart_from_metric_vector(label, "tRes", t_res.get("h"), unit="tRes h")
            if tres_chart:
                charts.setdefault("metric_tRes", tres_chart)

        # Generic metric vectors (e.g., SNR, Fsp, RN values, rho)
        metric_map = {
            "SNR": "SNR",
            "SNR_dB": "SNR (dB)",
            "R_AM": "R_AM",
            "R_AM_dB": "R_AM (dB)",
            "Fsp": "Fsp",
            "Fmp": "Fmp",
            "RN_elberlingDon": "Residual noise (Elberling)",
            "RN_eclipse": "Residual noise (Eclipse)",
            "rho": "Correlation",
            "f_CCA": "F_CCA",
            "T2circ": "T2 circ",
            "p_values": "p-values",
            "SNR_2_45Hz": "SNR 2-45 Hz",
            "SNR_max": "SNR max",
            "amp_V": "Amplitude",
            "SNR_time": "SNR time",
            "SNR_Peak": "SNR peak",
            "RN_micV": "Residual noise (µV)",
        }
        for key, label_name in metric_map.items():
            metric_chart = _chart_from_metric_vector(label, label_name, evaluation.get(key), unit=label_name)
            if metric_chart:
                charts.setdefault(metric_chart.key, metric_chart)

    eval_keys = list(evaluation.keys()) if isinstance(evaluation, dict) else None
    chart_keys = sorted(charts.keys()) if charts else []
    LOGGER.info(
        "Chart collection complete label=%s method=%s data_present=%s eval_keys=%s charts=%s",
        label,
        method_name,
        bool(data is not None),
        eval_keys,
        chart_keys,
    )

    return charts


def _resolve_aux_channels(params: Dict[str, Any]) -> int:
    if params.get("Device") == "ActiCHamp":
        return int(params.get("Parameters", {}).get("NumberAUXChannels", 0) or 0)
    return 0


def _plot_live_buffer(
    buffer: np.ndarray,
    fs: float,
    placeholder: "st.delta_generator.DeltaGenerator",
    channel_indices: Optional[List[int]] = None,
    interactive: bool = False,
    window_seconds: Optional[float] = None,
) -> None:
    if buffer.size == 0:
        return

    samples = buffer.shape[0]
    time_axis = np.arange(samples) / fs if fs > 0 else np.arange(samples)
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
                    name=f"Ch {ch + 1}",
                )
            )
        label_part = "All channels" if len(indices) == total_channels else ", ".join(f"{ch + 1}" for ch in indices)
        layout = go.Layout(
            height=320,
            margin=dict(l=50, r=10, t=40, b=50),
            xaxis=dict(
                title="Time (s)" if fs > 0 else "Samples",
                range=[time_axis_min, time_axis_max],
            ),
            yaxis=dict(title="Amplitude (uV)"),
            title=f"Live preview – {label_part}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
        )
        placeholder.plotly_chart(go.Figure(data=traces, layout=layout), width="stretch")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = plt.cm.tab10.colors
        for plot_idx, ch in enumerate(indices):
            color = colors[plot_idx % len(colors)]
            ax.plot(time_axis, buffer[:, ch], label=f"Ch {ch + 1}", color=color)
        ax.set_xlim(time_axis_min, time_axis_max)
        ax.set_xlabel("Time (s) — newest on the right" if fs > 0 else "Samples")
        ax.set_ylabel("Amplitude (uV)")
        label_part = "All channels" if len(indices) == total_channels else ", ".join(f"{ch + 1}" for ch in indices)
        ax.set_title(f"Live preview – {label_part}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        placeholder.pyplot(fig)
        plt.close(fig)


def _plot_fft_spectrum(
    buffer: np.ndarray,
    fs: float,
    placeholder: Optional["st.delta_generator.DeltaGenerator"],
    channel_indices: Optional[List[int]] = None,
    interactive: bool = False,
) -> None:
    if placeholder is None or buffer.size == 0 or fs <= 0:
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
                    name=f"Ch {ch + 1}",
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
            title=f"FFT – {label_part}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
            shapes=shapes,
        )
        placeholder.plotly_chart(go.Figure(data=traces, layout=layout), width="stretch")
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
            ax.plot(freq[freq_mask], power_db[freq_mask], linewidth=1.0, color=color, label=f"Ch {ch + 1}")

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
        ax.set_title(f"FFT – {label_part}")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        placeholder.pyplot(fig)
        plt.close(fig)


def _plot_individual_channels(
    buffer: np.ndarray,
    fs: float,
    placeholders: List["st.delta_generator.DeltaGenerator"],
    indices: List[int],
    interactive: bool = True,
    window_seconds: Optional[float] = None,
) -> None:
    if not placeholders or buffer.size == 0:
        return
    time_axis = np.arange(buffer.shape[0]) / fs if fs > 0 else np.arange(buffer.shape[0])
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
        if plot_idx >= len(placeholders):
            break
        placeholder = placeholders[plot_idx]
        color = palette[plot_idx % len(palette)] if palette else default_color
        base_color = color if len(color) >= 3 else (list(color) + list(default_color))[:3]
        rgb = tuple(int(max(0, min(255, round(float(val) * 255)))) for val in base_color)
        if go is None or not interactive:
            fig, ax = plt.subplots(figsize=(14, 3))
            ax.plot(time_axis, buffer[:, ch], color=color, linewidth=1.0)
            ax.set_xlim(time_axis_min, time_axis_max)
            ax.set_xlabel("Time (s) — newest on the right" if fs > 0 else "Samples")
            ax.set_ylabel("Amplitude (uV)")
            ax.set_title(f"Channel {ch + 1}")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            placeholder.pyplot(fig)
            plt.close(fig)
        else:
            trace = go.Scatter(
                x=time_axis,
                y=buffer[:, ch],
                mode="lines",
                line=dict(width=1.2, color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"),
                name=f"Ch {ch + 1}",
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
            placeholder.plotly_chart(go.Figure(data=[trace], layout=layout), width="stretch")


def _plot_topography(rows: List[Dict[str, Any]], placeholder: "st.delta_generator.DeltaGenerator") -> None:
    if placeholder is None:
        return
    if not rows:
        placeholder.info("No channels configured.")
        return

    labels = [row.get("Position") or row.get("Channel") or "" for row in rows]
    impedances = [coerce_number(row.get("Impedance")) for row in rows]
    total = len(labels)
    angles = np.linspace(0, 2 * np.pi, max(total, 8), endpoint=False)

    coords: List[tuple[str, float, float, Optional[float]]] = []
    fallback_idx = 0
    for idx, label in enumerate(labels):
        clean = label.replace(" ", "")
        custom_x = coerce_number(rows[idx].get("PosX"))
        custom_y = coerce_number(rows[idx].get("PosY"))
        xy = None
        if custom_x is not None and custom_y is not None:
            xy = (float(custom_x), float(custom_y))
        if xy is None:
            xy = TEN_TWENTY_COORDS.get(clean)
        if xy is None:
            angle = angles[fallback_idx % len(angles)]
            xy = (0.8 * np.cos(angle), 0.8 * np.sin(angle))
            fallback_idx += 1
        coords.append((label or f"Ch {idx+1}", xy[0], xy[1], impedances[idx]))

    fig, ax = plt.subplots(figsize=(5, 5))
    head = plt.Circle((0, 0), 1.05, edgecolor="black", facecolor="none", linewidth=1.5)
    ax.add_patch(head)
    nose = np.array([[0.0, 1.05], [-0.08, 1.15], [0.08, 1.15]])
    ax.plot(nose[:, 0], nose[:, 1], color="black", linewidth=1.2)
    ax.plot([-1.05, -1.2, -1.05], [0.15, 0.0, -0.15], color="black", linewidth=1.0)
    ax.plot([1.05, 1.2, 1.05], [0.15, 0.0, -0.15], color="black", linewidth=1.0)

    impedance_values = [val for val in impedances if val is not None]
    vmin, vmax = (min(impedance_values), max(impedance_values)) if impedance_values else (0.0, 1.0)
    if vmax == vmin:
        vmax = vmin + 1.0
    cmap = plt.cm.plasma

    for label, x, y, imp in coords:
        color = "#2d6cdf"
        if imp is not None:
            norm = (float(imp) - vmin) / (vmax - vmin)
            color = cmap(np.clip(norm, 0, 1))
        ax.scatter(x, y, s=160, color=color, edgecolors="white", linewidth=1.0, zorder=3)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="white", weight="bold", zorder=4)

    if impedance_values:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Impedance (kΩ)")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")
    ax.set_title("Scalp topography (positions + impedances)")
    fig.tight_layout()
    placeholder.pyplot(fig)
    plt.close(fig)


def _plotly_topography(rows: List[Dict[str, Any]]) -> Optional["go.Figure"]:
    if go is None:
        return None
    labels = [row.get("Position") or row.get("Channel") or "" for row in rows]
    impedances = [coerce_number(row.get("Impedance")) for row in rows]
    xs: List[float] = []
    ys: List[float] = []
    texts: List[str] = []
    colors: List[float] = []
    for idx, label in enumerate(labels):
        clean = label.replace(" ", "")
        custom_x = coerce_number(rows[idx].get("PosX"))
        custom_y = coerce_number(rows[idx].get("PosY"))
        xy = None
        if custom_x is not None and custom_y is not None:
            xy = (float(custom_x), float(custom_y))
        if xy is None:
            xy = TEN_TWENTY_COORDS.get(clean, (0.0, 0.0))
        xs.append(xy[0])
        ys.append(xy[1])
        texts.append(f"{label}<br>Imp: {impedances[idx] if impedances[idx] is not None else '—'} kΩ")
        colors.append(impedances[idx] if impedances[idx] is not None else 0.0)

    scatter = go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=[row.get("Channel") or f"Ch {i+1}" for i, row in enumerate(rows)],
        textposition="middle center",
        marker=dict(
            size=18,
            color=colors,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Imp (kΩ)"),
            line=dict(color="white", width=1),
        ),
        hoverinfo="text",
        hovertext=texts,
    )

    # Invisible dense grid to capture click positions everywhere inside the head.
    grid_x = []
    grid_y = []
    for gx in np.linspace(-1.1, 1.1, 30):
        for gy in np.linspace(-1.1, 1.1, 30):
            if gx**2 + gy**2 <= 1.25**2:
                grid_x.append(gx)
                grid_y.append(gy)
    grid = go.Scatter(
        x=grid_x,
        y=grid_y,
        mode="markers",
        marker=dict(size=16, opacity=0.01, color="rgba(0,0,0,0.05)"),
        hoverinfo="none",
        showlegend=False,
        name="click-target",
    )

    fig = go.Figure()
    fig.add_trace(grid)
    fig.add_trace(scatter)
    fig.update_layout(
        width=500,
        height=500,
        xaxis=dict(visible=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, range=[-1.3, 1.3]),
        title="Drag or click to reposition a channel",
        margin=dict(l=10, r=10, t=40, b=10),
        clickmode="event+select",
        dragmode="pan",
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True, scaleanchor="x", scaleratio=1)
    fig.add_shape(type="circle", x0=-1.05, y0=-1.05, x1=1.05, y1=1.05, line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=-1.05, y0=0.15, x1=-1.2, y1=0.0, line=dict(color="black"))
    fig.add_shape(type="line", x0=-1.2, y0=0.0, x1=-1.05, y1=-0.15, line=dict(color="black"))
    fig.add_shape(type="line", x0=1.05, y0=0.15, x1=1.2, y1=0.0, line=dict(color="black"))
    fig.add_shape(type="line", x0=1.2, y0=0.0, x1=1.05, y1=-0.15, line=dict(color="black"))
    fig.add_shape(type="path", path="M -0.08 1.05 L 0 1.15 L 0.08 1.05", line=dict(color="black", width=2))
    return fig


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
        chunk = self._wrapped.acquire(duration_seconds, aux_channels)
        self._live_view.push(chunk, self._fs)
        return chunk

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
        placeholder: "st.delta_generator.DeltaGenerator",
        window_seconds: float = 5.0,
        channel_indices: Optional[List[int]] = None,
        fft_placeholder: Optional["st.delta_generator.DeltaGenerator"] = None,
    ) -> None:
        self.placeholder = placeholder
        self.fft_placeholder = fft_placeholder
        self.window_seconds = max(1.0, float(window_seconds))
        self.channel_indices = channel_indices or []
        self.buffer: np.ndarray = np.empty((0, 0))

    def wrap_device(self, device: DeviceInterface, params: Dict[str, Any]) -> LiveViewDevice:
        fs_value = coerce_number(params.get("Parameters", {}).get("fs"))
        fs = float(fs_value) if fs_value else 0.0
        self.reset()
        return LiveViewDevice(device, self, fs)

    def reset(self) -> None:
        self.buffer = np.empty((0, 0))
        if self.placeholder is not None:
            self.placeholder.empty()

    def push(self, chunk: Any, fs: float) -> None:
        if chunk is None or self.placeholder is None:
            return
        array = np.asarray(chunk, dtype=float)
        if array.ndim == 1:
            array = array[:, np.newaxis]
        if array.size == 0:
            return
        if self.buffer.size == 0:
            self.buffer = array
        else:
            self.buffer = np.vstack([self.buffer, array])
        max_window = int(fs * self.window_seconds) if fs > 0 else self.buffer.shape[0]
        if max_window > 0 and self.buffer.shape[0] > max_window:
            self.buffer = self.buffer[-max_window:]
        indices = _normalize_channel_indices(self.channel_indices, self.buffer.shape[1])
        _plot_live_buffer(self.buffer, fs, self.placeholder, channel_indices=indices)
        if self.fft_placeholder is not None:
            _plot_fft_spectrum(self.buffer, fs, self.fft_placeholder, channel_indices=indices)


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
        max_window = int(math.ceil(fs * window)) if fs > 0 else None
        if buffer.size and max_window:
            buffer = buffer[-max_window:]
        prime_chunk = np.asarray(device.prime(min(update_interval, duration), aux_channels), dtype=float)
        if prime_chunk.ndim == 1:
            prime_chunk = prime_chunk[:, np.newaxis]
        if buffer.size == 0:
            buffer = prime_chunk
        else:
            buffer = np.vstack([buffer, prime_chunk])
        if max_window and buffer.shape[0] > max_window:
            buffer = buffer[-max_window:]
        # Live view uses lightweight (matplotlib) rendering to avoid flicker; interactive Plotly drawn after loop.
        _plot_live_buffer(buffer, fs, placeholder, channel_indices=indices, interactive=False, window_seconds=window)
        _plot_fft_spectrum(buffer, fs, fft_placeholder, channel_indices=indices, interactive=False)
        # To keep UI smooth, only show the aggregated view + FFT during streaming.
        start = time.time()
        while (time.time() - start) < duration:
            remaining = duration - (time.time() - start)
            chunk = np.asarray(device.acquire(min(update_interval, remaining), aux_channels), dtype=float)
            if chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]
            if chunk.size > 0:
                if buffer.size == 0:
                    buffer = chunk
                else:
                    buffer = np.vstack([buffer, chunk])
                max_window = int(math.ceil(fs * window)) if fs > 0 else buffer.shape[0]
                if buffer.shape[0] > max_window:
                    buffer = buffer[-max_window:]
                _plot_live_buffer(buffer, fs, placeholder, channel_indices=indices, interactive=False, window_seconds=window)
                _plot_fft_spectrum(buffer, fs, fft_placeholder, channel_indices=indices, interactive=False)
            else:
                time.sleep(update_interval)
        # After capture, replace with interactive Plotly charts if available.
        if final_interactive:
            _plot_live_buffer(buffer, fs, placeholder, channel_indices=indices, interactive=True, window_seconds=window)
            _plot_fft_spectrum(buffer, fs, fft_placeholder, channel_indices=indices, interactive=True)
            if channels_enabled:
                _plot_individual_channels(buffer, fs, channel_placeholders or [], indices, interactive=True, window_seconds=window)
        return buffer
    finally:
        device.disconnect()
        LOGGER.info("run_live_preview finished")


def render_chart_section(label: str, params: Dict[str, Any], data: Optional[np.ndarray], aggregate: bool = False) -> None:
    eval_figures = _render_evaluation_figures(params, data)
    LOGGER.debug(
        "Render chart section",
        extra={
            "label": label,
            "method": params.get("Method"),
            "eval_figures": len(eval_figures),
        },
    )
    charts = collect_chart_data(label, params or {}, data, aggregate=aggregate)
    if not charts:
        st.info("No charts available for this session yet.")
        return
    st.subheader(f"Charts – {label}")
    if eval_figures:
        st.caption("Evaluation plots")
        for fig in eval_figures:
            st.pyplot(fig, clear_figure=False)
        plt.close("all")
    for key in sorted(charts.keys()):
        render_chart(charts[key])


def _safe_stat_value(func, array: np.ndarray) -> Optional[float]:
    try:
        value = float(func(array))
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _summarize_session_stats(label: str, params: Dict[str, Any], data: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr[:, None]

    n_samples, n_channels = arr.shape[0], arr.shape[1]
    flat = arr.reshape(-1)
    fs_value = coerce_number(params.get("Parameters", {}).get("fs") if params else None)
    fs = float(fs_value) if fs_value else 0.0
    duration = (n_samples / fs) if fs > 0 else None

    return {
        "Session": label,
        "Samples": n_samples,
        "Channels": n_channels,
        "Duration (s)": round(duration, 2) if duration is not None and math.isfinite(duration) else None,
        "Median": _safe_stat_value(np.nanmedian, flat),
        "Mean": _safe_stat_value(np.nanmean, flat),
        "Std": _safe_stat_value(np.nanstd, flat),
        "P25": _safe_stat_value(lambda a: np.nanpercentile(a, 25), flat),
        "P75": _safe_stat_value(lambda a: np.nanpercentile(a, 75), flat),
        "Min": _safe_stat_value(np.nanmin, flat),
        "Max": _safe_stat_value(np.nanmax, flat),
    }


def _summarize_channel_stats(label: str, params: Dict[str, Any], data: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    if data is None:
        return []
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr[:, None]

    n_channels = arr.shape[1]
    labels: List[str] = []
    for entry in params.get("Channels") or []:
        name = entry.get("Channel") or entry.get("Label")
        if name:
            labels.append(str(name))
    if len(labels) < n_channels:
        labels.extend([f"Ch {idx + 1}" for idx in range(len(labels), n_channels)])

    rows: List[Dict[str, Any]] = []
    for idx in range(n_channels):
        ch_data = arr[:, idx]
        rows.append(
            {
                "Session": label,
                "Channel": labels[idx] if idx < len(labels) else f"Ch {idx + 1}",
                "Median": _safe_stat_value(np.nanmedian, ch_data),
                "Mean": _safe_stat_value(np.nanmean, ch_data),
                "Std": _safe_stat_value(np.nanstd, ch_data),
                "P25": _safe_stat_value(lambda a: np.nanpercentile(a, 25), ch_data),
                "P75": _safe_stat_value(lambda a: np.nanpercentile(a, 75), ch_data),
                "Min": _safe_stat_value(np.nanmin, ch_data),
                "Max": _safe_stat_value(np.nanmax, ch_data),
            }
        )
    return rows


def render_comparison_charts(payloads: List[tuple[str, Dict[str, Any], Optional[np.ndarray]]]) -> None:
    session_stats: List[Dict[str, Any]] = []
    channel_stats: List[Dict[str, Any]] = []
    merged: Dict[str, ChartData] = {}
    for label, params, data in payloads:
        stats_row = _summarize_session_stats(label, params or {}, data)
        if stats_row:
            session_stats.append(stats_row)
        channel_stats.extend(_summarize_channel_stats(label, params or {}, data))
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

    if session_stats:
        st.subheader("Comparison report")
        summary_columns = [
            "Session",
            "Samples",
            "Channels",
            "Duration (s)",
            "Median",
            "Mean",
            "Std",
            "P25",
            "P75",
            "Min",
            "Max",
        ]
        df = pd.DataFrame(session_stats)
        df = df[[col for col in summary_columns if col in df.columns]]
        st.dataframe(df)

    if channel_stats:
        with st.expander("Per-channel statistics"):
            channel_columns = [
                "Session",
                "Channel",
                "Median",
                "Mean",
                "Std",
                "P25",
                "P75",
                "Min",
                "Max",
            ]
            channel_df = pd.DataFrame(channel_stats)
            channel_df = channel_df[[col for col in channel_columns if col in channel_df.columns]]
            st.dataframe(channel_df)

    if not merged:
        st.info("No comparable charts for the selected sessions.")
        return

    st.subheader("Comparison overlays")
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
    if device.lower() == "unicorn":
        general["fs"] = "250"

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
                    "PosX": coerce_number(ch.get("PosX")),
                    "PosY": coerce_number(ch.get("PosY")),
                    "Active": bool(ch.get("Active", True)),
                }
            )
        st.session_state["channel_tables"][device] = ensure_channel_rows(device, imported_rows)

    metadata = params.get("Metadata") or {}
    participant_meta = metadata.get("Participant")
    if participant_meta:
        st.session_state["participant"].update(participant_meta)

    # The loaded values live in the backing dicts above. Clear the stale *widget* state so the
    # widgets re-initialise from those dicts on the next run: Streamlit ignores value=/index=
    # once a keyed widget has been instantiated, so without this import/load silently no-ops.
    for f in GENERAL_SCHEMA:
        st.session_state.pop(f"general_{f.name}", None)
    for f in METHOD_SCHEMAS.get(method, []):
        st.session_state.pop(f"{method}_{f.name}", None)
    for f in DEVICE_CONFIG_SCHEMA.get(device, []):
        st.session_state.pop(f"device_{device}_{f['name']}", None)
    _bump_channel_editor_revision(device)

    data_array = data_override
    if data_array is None and params.get("data") is not None:
        data_array = np.asarray(params["data"])
    st.session_state["imported_data"] = data_array
    has_data = False
    if data_array is not None:
        try:
            has_data = np.asarray(data_array).size > 0
        except Exception:
            has_data = True
    st.session_state["use_imported_data"] = has_data
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
    if "_last_device_selection" not in st.session_state:
        st.session_state["_last_device_selection"] = None
    if "_actichamp_impedance_loaded" not in st.session_state:
        st.session_state["_actichamp_impedance_loaded"] = False
    if "_actichamp_impedance_timestamp" not in st.session_state:
        st.session_state["_actichamp_impedance_timestamp"] = None
    if "_actichamp_impedance_status" not in st.session_state:
        st.session_state["_actichamp_impedance_status"] = None
    if "_live_view_active" not in st.session_state:
        st.session_state["_live_view_active"] = False
    if "_channel_editor_revision" not in st.session_state:
        st.session_state["_channel_editor_revision"] = {}


def get_live_view_placeholder():
    placeholder = st.session_state.get("_live_view_placeholder")
    if placeholder is None:
        placeholder = st.empty()
        st.session_state["_live_view_placeholder"] = placeholder
    return placeholder


def _render_live_preview_frame(
    placeholder: "st.delta_generator.DeltaGenerator",
    fft_placeholder: Optional["st.delta_generator.DeltaGenerator"],
    channel_placeholders: Optional[List["st.delta_generator.DeltaGenerator"]],
) -> None:
    buffer = st.session_state.get("_live_preview_buffer")
    meta = st.session_state.get("_live_preview_meta") or {}
    if buffer is None:
        return
    buf = np.asarray(buffer)
    fs = float(meta.get("fs") or 0.0)
    indices = meta.get("indices")
    _plot_live_buffer(buf, fs, placeholder, channel_indices=indices)
    _plot_fft_spectrum(buf, fs, fft_placeholder, channel_indices=indices)
    if channel_placeholders:
        _plot_individual_channels(buf, fs, channel_placeholders, indices or [])


def _fetch_actichamp_impedances(fs_value: Any) -> None:
    if st.session_state.get("_actichamp_impedance_loaded"):
        return

    fs = coerce_number(fs_value)
    if fs is None or fs <= 0:
        st.session_state["_actichamp_impedance_status"] = "Set a valid sampling rate (fs) to read ActiCHamp impedances."
        return

    channel_tables = st.session_state.setdefault("channel_tables", {})
    rows = ensure_channel_rows("ActiCHamp", channel_tables.get("ActiCHamp"))
    channel_tables["ActiCHamp"] = rows
    channel_count = _actichamp_channel_count(rows)

    params = {
        "Device": "ActiCHamp",
        "Parameters": {"fs": float(fs), "NumberEEGChannels": channel_count},
    }

    LOGGER.info("Attempting ActiCHamp impedance read (fs=%s, channels=%s)", fs, channel_count)
    values: List[float] = []

    try:
        with st.spinner("Checking ActiCHamp impedances..."):
            device = DeviceFactory.create(params)
            try:
                reader = getattr(device, "read_impedances", None)
                values = reader(wait_seconds=7.0, settle_seconds=1.5) if callable(reader) else []
            finally:
                device.disconnect()
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("ActiCHamp impedance read failed")
        st.warning(f"ActiCHamp impedance read failed: {exc}")
        st.session_state["_actichamp_impedance_status"] = f"ActiCHamp impedance read failed: {exc}"
        return

    if not values:
        LOGGER.warning("ActiCHamp impedance read returned no values.")
        st.session_state["_actichamp_impedance_status"] = "ActiCHamp impedance read returned no values."
        return

    if not _has_measured_impedance(values):
        LOGGER.warning("ActiCHamp impedance read returned only zero or unavailable values: %s", values)
        st.session_state["_actichamp_impedance_status"] = (
            "ActiCHamp impedance read returned only zero/unavailable values. "
            "Check that the producer is in impedance mode and electrodes are connected."
        )
        return

    LOGGER.info("Loaded %d ActiCHamp impedance values", len(values))
    channel_tables["ActiCHamp"] = _map_impedances_to_channels(rows, values)
    _bump_channel_editor_revision("ActiCHamp")
    st.session_state["_actichamp_impedance_loaded"] = True
    range_kohm = _impedance_range_kohm(values)
    if range_kohm is None:
        st.session_state["_actichamp_impedance_status"] = f"Loaded {len(values)} impedance values."
    else:
        low, high = range_kohm
        st.session_state["_actichamp_impedance_status"] = (
            f"Loaded {len(values)} impedance values ({low:.1f}-{high:.1f} kΩ)."
        )
    st.session_state["_actichamp_impedance_timestamp"] = time.time()


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
    fs_txt = f"{fs_value} Hz" if fs_value is not None else "—"
    ch_txt = f"{int(channels)}" if channels is not None else "—"
    dur_txt = f"{duration} s" if duration is not None else "— s"

    html_block = textwrap.dedent(
        f"""
        <div class="snapshot-panel">
            <div class="title">Method: {html.escape(method_label)}</div>
            {'<div class="desc">' + html.escape(desc) + '</div>' if desc else ''}
            <div class="line"><strong>Device:</strong> {html.escape(device)}</div>
            <div class="stats">fs: {fs_txt}<br>Channels: {ch_txt}<br>Recording time: {dur_txt}</div>
        </div>
        """
    )
    st.markdown(html_block, unsafe_allow_html=True)


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
    device_is_unicorn = device.lower() == "unicorn"
    cols = form_col.columns(2)
    for idx, field in enumerate(other_fields):
        target = cols[idx % 2]
        key = f"general_{field.name}"
        current = general.get(field.name)
        if field.name == "fs" and device_is_unicorn:
            locked_value = "250"
            if st.session_state.get(key) != locked_value:
                st.session_state[key] = locked_value
            options = [locked_value] + [opt for opt in (field.options or []) if opt != locked_value]
            value = target.selectbox(
                field.name,
                options=options,
                index=options.index(locked_value),
                help="UNICORN sampling rate is fixed to 250 Hz.",
                key=key,
                disabled=True,
            )
        elif field.name == "fs":
            options = DEVICE_FS_OPTIONS.get(device, field.options or [""]) or [""]
            if st.session_state.get(key) not in options:
                st.session_state.pop(key, None)  # previous device's rate may be unavailable here
            resolved = resolve_choice(options, current)
            fs_help = field.tooltip or None
            if device == "ActiCHamp":
                fs_help = "ActiCHamp-supported rates. An unsupported rate is clamped by the hardware and makes the recording run longer than RecordingTime."
            value = target.selectbox(
                field.name,
                options=options,
                index=options.index(resolved),
                help=fs_help,
                key=key,
            )
        elif field.kind == "dropdown":
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
    previous_device = st.session_state.get("_last_device_selection")
    device_changed = device != previous_device
    if device_changed:
        st.session_state["_last_device_selection"] = device
        st.session_state["_actichamp_impedance_loaded"] = False
        st.session_state["_actichamp_impedance_status"] = None
        st.session_state["_actichamp_impedance_timestamp"] = None

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
            if device == "UNICORN" and field["name"] == "UNICORNPort":
                value = render_unicorn_port_input(target, field, current, key)
            elif field["kind"] == "number":
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
        ncols = 3 if len(schema) > 4 else 2  # denser grid for long method forms (less scrolling)
        cols = st.columns(ncols)
        for idx, field in enumerate(schema):
            target = cols[idx % ncols]
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
        pos_x, pos_y = _channel_default_coords(position_label)
        row = {
            "Channel": label,
            "Position": position_label,
            "Rubrik": default_rubric,
            "Model": default_model,
            "Impedance": 0.0,
            "Active": True if is_extra else False,
            "PosX": pos_x,
            "PosY": pos_y,
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
    editor_revision = int(st.session_state.setdefault("_channel_editor_revision", {}).get(device, 0))
    with st.expander(f"Electrodes ({device})", expanded=True):
        st.caption(
            "Toggle the channels you intend to record, set the 10-20 name (Position), adjust coordinates, and choose "
            "hardware. Ground/Reference rows stay enabled automatically."
        )
        if device == "ActiCHamp":
            fs_value = st.session_state.get("general_form", {}).get("fs")
            status = st.session_state.get("_actichamp_impedance_status")
            last_ts = st.session_state.get("_actichamp_impedance_timestamp")
            btn_cols = st.columns([1, 1, 2])
            if btn_cols[0].button("Read ActiCHamp impedances", key="actichamp_impedance_button"):
                st.session_state["_actichamp_impedance_loaded"] = False
                _fetch_actichamp_impedances(fs_value)
                rows = st.session_state["channel_tables"].get("ActiCHamp", rows)
                editor_revision = int(st.session_state["_channel_editor_revision"].get(device, 0))
            if btn_cols[1].button("Clear impedances", key="actichamp_impedance_clear"):
                st.session_state["_actichamp_impedance_loaded"] = False
                st.session_state["_actichamp_impedance_timestamp"] = None
                st.session_state["_actichamp_impedance_status"] = "Cleared previously loaded impedances."
                for row in rows:
                    row["Impedance"] = 0.0
                channel_state["ActiCHamp"] = rows
                _bump_channel_editor_revision("ActiCHamp")
                editor_revision = int(st.session_state["_channel_editor_revision"].get(device, 0))
            with btn_cols[2]:
                if status:
                    st.info(status)
                if last_ts:
                    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts))
                    st.caption(f"Last read: {ts_str}")

        # Quick setup: fill the standard montage / bulk-toggle active channels without hand-editing.
        extras_set = set(DEVICE_EXTRA_LABELS.get(device, []))
        qs = st.columns(3)
        if qs[0].button("Fill standard positions", key=f"fill_pos_{device}",
                        help="Set each channel's Position from the device's standard montage and activate it."):
            preset = DEVICE_POSITION_DEFAULTS.get(device, [])
            eeg_idx = 0
            for row in rows:
                if row["Channel"] in extras_set:
                    continue
                if eeg_idx < len(preset):
                    row["Position"] = preset[eeg_idx]
                    row["PosX"], row["PosY"] = _channel_default_coords(preset[eeg_idx])
                row["Active"] = True
                eeg_idx += 1
            channel_state[device] = rows
            _bump_channel_editor_revision(device)
            st.rerun()
        if qs[1].button("Activate all", key=f"activate_all_{device}"):
            for row in rows:
                row["Active"] = True
            channel_state[device] = rows
            _bump_channel_editor_revision(device)
            st.rerun()
        if qs[2].button("Deactivate EEG", key=f"deactivate_eeg_{device}",
                        help="Turn off all EEG channels (Ground/Reference stay on)."):
            for row in rows:
                if row["Channel"] not in extras_set:
                    row["Active"] = False
            channel_state[device] = rows
            _bump_channel_editor_revision(device)
            st.rerun()

        # Reference electrode (#8): designate which EEG channel is the reference. ReferenceChannel is
        # a 1-based index over the EEG channels (excluding Ground/Reference extras), matching analysis.
        method = st.session_state.get("general_form", {}).get("Method")
        method_has_ref = any(f.name == "ReferenceChannel" for f in METHOD_SCHEMAS.get(method, []))
        eeg_rows = [row for row in rows if row["Channel"] not in extras_set]
        if method and method_has_ref and eeg_rows:
            ref_labels = [f"{i + 1}: {row.get('Position') or row['Channel']}" for i, row in enumerate(eeg_rows)]
            current_ref = st.session_state.get("method_forms", {}).get(method, {}).get("ReferenceChannel")
            try:
                cur_idx = int(current_ref) - 1
            except (TypeError, ValueError):
                cur_idx = 0
            cur_idx = cur_idx if 0 <= cur_idx < len(ref_labels) else 0
            choice = st.selectbox(
                "Reference electrode",
                options=list(range(len(ref_labels))),
                format_func=lambda i: ref_labels[i],
                index=cur_idx,
                key=f"ref_electrode_{device}",
                help="Channel used as the reference (ReferenceChannel) during analysis.",
            )
            new_ref = choice + 1
            if str(new_ref) != str(current_ref):
                st.session_state.setdefault("method_forms", {}).setdefault(method, {})["ReferenceChannel"] = new_ref
                st.session_state.pop(f"{method}_ReferenceChannel", None)  # keep the method form in sync

        table_col, map_col = st.columns((2, 1))
        with table_col:
            edited = st.data_editor(
                rows,
                num_rows="fixed",
                hide_index=True,
                key=f"channels_{device}_{editor_revision}",
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
                    "PosX": st.column_config.NumberColumn(
                        "Pos X",
                        help="Custom X coordinate for scalp plot (-1.5 to 1.5). Leave blank to use defaults.",
                        min_value=-1.5,
                        max_value=1.5,
                        step=0.05,
                        format="%.2f",
                    ),
                    "PosY": st.column_config.NumberColumn(
                        "Pos Y",
                        help="Custom Y coordinate for scalp plot (-1.5 to 1.5). Leave blank to use defaults.",
                        min_value=-1.5,
                        max_value=1.5,
                        step=0.05,
                        format="%.2f",
                    ),
                    "Active": st.column_config.CheckboxColumn("Use channel"),
                },
            )
            extras = set(DEVICE_EXTRA_LABELS.get(device, []))
            for row in edited:
                if row["Channel"] in extras:
                    row["Active"] = True
                if not row.get("Position"):
                    row["Position"] = row["Channel"].replace(" ", "")
                if row.get("PosX") in ("", None) or row.get("PosY") in ("", None):
                    pos_x, pos_y = _channel_default_coords(row["Position"])
                    row["PosX"] = pos_x
                    row["PosY"] = pos_y
                # Keep Model consistent with the chosen electrode type (Rubrik) using electrodes.json:
                # if the model isn't valid for that type, snap it to the first valid one.
                models_for_type = ELECTRODE_LIBRARY.get(row.get("Rubrik") or "", [])
                if models_for_type and row.get("Model") not in models_for_type:
                    row["Model"] = models_for_type[0]
            channel_state[device] = edited

        with map_col:
            st.caption("Scalp map")
            enable_click_placement = st.checkbox(
                "Enable click placement",
                value=False,
                key=f"enable_topo_click_{device}",
                help="Loads an optional custom Streamlit component for placing electrodes by clicking the map.",
            )
            if enable_click_placement and plotly_events is not None and go is not None:
                channel_labels = [row.get("Channel") or f"Ch {idx+1}" for idx, row in enumerate(edited)]
                target = st.selectbox("Channel to place", options=channel_labels, key=f"topo_target_{device}")
                fig = _plotly_topography(edited)
                st.caption("Pick a channel, then click on the head map to place it.")
                events = plotly_events(
                    fig,
                    click_event=True,
                    select_event=True,
                    override_height=520,
                    override_width=520,
                    key=f"topo_events_{device}",
                )
                if events:
                    evt = events[0]
                    x_new = evt.get("x")
                    y_new = evt.get("y")
                    if x_new is not None and y_new is not None:
                        idx = channel_labels.index(target)
                        edited[idx]["PosX"] = float(x_new)
                        edited[idx]["PosY"] = float(y_new)
                        channel_state[device] = edited
                        st.session_state["_topo_last_message"] = f"Updated {target} to ({x_new:.2f}, {y_new:.2f})."
                        st.rerun()
                if msg := st.session_state.get("_topo_last_message"):
                    st.info(msg)
            elif enable_click_placement:
                st.info("Install optional deps `plotly` and `streamlit-plotly-events` for click placement.")
            else:
                st.caption("Click placement is off. Use the coordinate columns to edit positions, or enable it here.")
            _plot_topography(edited, st.empty())
    return edited


def render_live_preview_tab(params: Dict[str, Any], validation_issues: List[str]) -> None:
    st.subheader("Live preview (beta)")
    st.caption(
        "Stream a short window from the configured device. This uses the current session settings and renders up to "
        "four channels in real time."
    )

    if validation_issues:
        st.warning("Fix configuration issues in the Session tab before starting a live preview.")
        return

    parameters = params.get("Parameters", {})
    fs_value = coerce_number(parameters.get("fs"))
    fs = float(fs_value) if fs_value else 0.0
    channel_count = len(params.get("Channels", [])) or int(
        parameters.get("NumberEEGChannels") or DEVICE_DEFAULT_CHANNELS.get(params.get("Device"), 8)
    )
    channel_count = max(channel_count, 1)
    channel_labels = [row.get("Channel") or f"Ch {idx+1}" for idx, row in enumerate(params.get("Channels", []))] or [
        f"Ch {idx+1}" for idx in range(channel_count)
    ]
    options = ["All"] + channel_labels
    selection = st.multiselect("Channels to preview", options=options, default=["All"])
    if not selection or "All" in selection:
        selected_indices = list(range(channel_count))
    else:
        selected_indices = [channel_labels.index(label) for label in selection if label in channel_labels]

    unlimited = st.checkbox("Run indefinitely (manual stop)", value=False)
    duration = float("inf") if unlimited else float(
        st.slider("Preview duration (s)", min_value=2, max_value=120, value=10, step=1)
    )
    window = st.slider(
        "Rolling window (s)",
        min_value=1,
        max_value=120,
        value=min(5, int(duration) if math.isfinite(duration) else 5),
        step=1,
    )
    interval = st.slider("Update interval (s)", min_value=0.05, max_value=1.0, value=0.25, step=0.05)

    placeholder = st.empty()
    fft_placeholder = st.empty()
    st.caption("Per-channel views")
    per_channel_container = st.container()
    per_channel_placeholders = [per_channel_container.empty() for _ in selected_indices]
    state = st.session_state
    active = state.get("_live_preview_active", False)
    start_clicked = st.button("Start live preview", type="primary", disabled=active)
    stop_clicked = st.button("Stop live preview", type="secondary", disabled=not active)

    if start_clicked:
        state["_live_preview_active"] = True
        state["_live_preview_unlimited"] = bool(unlimited)
        state["_live_preview_args"] = {
            "selected_indices": selected_indices,
            "duration": float(duration),
            "window": float(window),
            "interval": float(interval),
        }
        state["_live_preview_last_buffer"] = None
        st.rerun()

    if stop_clicked:
        state["_live_preview_active"] = False
        st.rerun()

    if state.get("_live_preview_active") and state.get("_live_preview_args"):
        args = state["_live_preview_args"]
        chunk_duration = args["duration"] if math.isfinite(args["duration"]) else args["window"]
        try:
            buffer = run_live_preview(
                params,
                placeholder,
                fft_placeholder,
                args["selected_indices"],
                per_channel_placeholders,
                initial_buffer=state.get("_live_preview_last_buffer"),
                duration=float(chunk_duration),
                window=float(args["window"]),
                update_interval=float(args["interval"]),
                final_interactive=not state.get("_live_preview_unlimited", False),
            )
            state["_live_preview_last_buffer"] = buffer
            if state.get("_live_preview_unlimited", False) and state.get("_live_preview_active", False):
                st.rerun()
            else:
                state["_live_preview_active"] = False
                st.success(f"Captured {buffer.shape[0]} samples over {args['duration']} seconds.")
        except Exception as exc:  # pragma: no cover
            state["_live_preview_active"] = False
            st.error(f"Live preview failed: {exc}")
    if (
        not state.get("_live_preview_active")
        and state.get("_live_preview_last_buffer") is not None
        and state.get("_live_preview_unlimited", False)
        and state.get("_live_preview_args")
    ):
        buffer = state["_live_preview_last_buffer"]
        indices = _normalize_channel_indices(state["_live_preview_args"].get("selected_indices"), channel_count)
        _plot_live_buffer(buffer, fs, placeholder, channel_indices=indices, interactive=True)
        _plot_fft_spectrum(buffer, fs, fft_placeholder, channel_indices=indices, interactive=True)
        _plot_individual_channels(buffer, fs, per_channel_placeholders, indices, interactive=True)
        st.success(f"Stopped live preview after capturing {buffer.shape[0]} samples.")


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
        pos_x = coerce_number(row.get("PosX"))
        pos_y = coerce_number(row.get("PosY"))
        if pos_x is not None and pos_y is not None:
            entry["PosX"] = float(pos_x)
            entry["PosY"] = float(pos_y)
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

    if params["Device"].lower() == "unicorn":
        params["Parameters"]["fs"] = 250

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


def current_params_snapshot() -> Optional[Dict[str, Any]]:
    """Assemble the current editor state into a params dict for export (None if incomplete)."""
    general = dict(st.session_state.get("general_form", {}) or {})
    if not general.get("Method") or not general.get("Device"):
        return None
    method = general["Method"]
    device = general["Device"]
    method_values = dict(st.session_state.get("method_forms", {}).get(method, {}) or {})
    device_values = dict(st.session_state.get("device_forms", {}).get(device, {}) or {})
    participant = dict(st.session_state.get("participant", {}) or {})
    try:
        return assemble_params(general, method_values, participant, device_values)
    except Exception:  # pragma: no cover - defensive: never break the sidebar over export
        LOGGER.exception("Failed to assemble params snapshot for export")
        return None


def verify_device_connection(params: Dict[str, Any]) -> tuple[bool, str]:
    """Connect to the configured device and confirm it streams a short window.

    Returns (ok, message). Used as a pre-flight 'first-connect confirm' so a recording
    isn't started against a device that hasn't actually come up / is still settling.
    """
    device = DeviceFactory.create(params)
    aux = 0
    if params.get("Device") == "ActiCHamp":
        aux = int(params.get("Parameters", {}).get("NumberAUXChannels", 0) or 0)
    try:
        device.connect()
        probe_s = 0.5
        sample = np.asarray(device.acquire(probe_s, aux), dtype=float)
        if sample.size == 0 or sample.shape[0] == 0:
            return False, "Connected, but no samples were received (device may still be settling)."
        eff_fs = sample.shape[0] / probe_s
        return True, f"Streaming — {sample.shape[1]} channels, ~{eff_fs:.0f} Hz over {probe_s:.1f}s."
    finally:
        try:
            device.disconnect()
        except Exception:  # pragma: no cover - best-effort cleanup
            LOGGER.debug("Device disconnect after verify raised", exc_info=True)


def export_recording(data: Any, params: Dict[str, Any], out_dir: Path, container: str, raw_format: str) -> Path:
    """Export a recording as BIDS or SBIDS with a Parquet/EDF raw layer. Returns the output folder."""
    from cortipy.shared.bids import BIDSLoader, BIDSLoadResult, _coerce_to_raw_array

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Recording data must be 2-D (samples x channels).")
    pblock = params.get("Parameters", {}) if isinstance(params, dict) else {}
    fs = float(pblock.get("fs") or 250) or 250.0
    chans = params.get("Channels") or []
    ch_names = [(c.get("Position") or c.get("Channel")) for c in chans] or None
    if ch_names and len(ch_names) != arr.shape[1]:
        ch_names = None  # fall back to Ch1..N when the montage doesn't match the data width
    part = (params.get("Metadata") or {}).get("Participant") or {}
    subject = (str(part.get("Code") or "01").replace(" ", "") or "01")
    task = str(params.get("Method") or "task").lower()
    fmt = raw_format.lower()
    out_dir = Path(out_dir)

    if container.upper() == "BIDS":
        root = out_dir / "bids_export"
        BIDSLoader(root).to_bids(
            arr, sampling_rate=fs, ch_names=ch_names, subject=subject, task=task,
            format=fmt, overwrite=True, dataset_description={"Name": params.get("Method") or "CortiPy export"},
        )
        return root

    from cortipy.shared.dataset import CortiDataset  # SBIDS path
    raw = _coerce_to_raw_array(arr, fs, ch_names, None)
    result = BIDSLoadResult(
        raw=raw, data=arr, sampling_rate=fs, events=None, channels=None,
        metadata={"params": params}, source_path=Path(f"sub-{subject}_{task}_scalpdata"), ancillary_files=[],
    )
    out = out_dir / "sbids_export" / f"sbids_meta_sub-{subject}.jsonld"
    out.parent.mkdir(parents=True, exist_ok=True)
    CortiDataset(result).to_sbids(out, export_format=fmt)
    return out.parent


@st.cache_data
def list_saved_sessions(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    return sorted([path for path in base_dir.iterdir() if path.is_dir()], reverse=True)


def run_pipeline_once(
    params: Dict[str, Any],
    save_dir: Path,
    live_view: Optional[LiveViewService] = None,
) -> tuple[Dict[str, Any], Optional[Path]]:
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

    def attach_context(context):
        if live_view is not None:
            context.attach_service("live_view", live_view)

    hooks = PipelineHooks(
        params_provider=provider,
        save_callback=save_and_capture,
        should_continue=lambda _: False,
        context_hook=attach_context,
    )
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
                st.session_state["_flash"] = "Session loaded. Review settings before running."
                st.rerun()
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
        normalized = normalize_params(params_content)
        primary_payload = (primary.name, normalized, data_array)
        st.session_state["last_results"] = {
            "label": primary.name,
            "params": normalized,
            "data": data_array,
        }

    compare_payloads: List[tuple[str, Dict[str, Any], Optional[np.ndarray]]] = []
    compare_button = st.button("Compare selected sessions", key="compare_sessions")
    if compare_button:
        for path in compare_selection:
            params_loaded, data_loaded = _load_session_contents(path)
            if params_loaded:
                compare_payloads.append((path.name, normalize_params(params_loaded), data_loaded))
    return primary_payload, compare_payloads


def handle_upload(target) -> None:
    with target.expander("Import config / params", expanded=False):
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
                st.session_state["_flash"] = f"Imported parameters from '{uploaded.name}'."
                st.rerun()

        if data_upload:
            data_array = _load_npz_array(data_upload)
            if data_array is not None:
                st.session_state["imported_data"] = data_array
                st.session_state["use_imported_data"] = True
                st.success(f"Attached data from '{data_upload.name}'.")


def render_sidebar_controls() -> SidebarControls:
    sidebar = st.sidebar
    sidebar.title("Controls")

    with sidebar.container(border=True):
        st.subheader("Measurement")
        default_save = st.text_input(
            "Experiment / save directory",
            value=str(DEFAULT_SAVE_DIR),
            help="Folder where run outputs are written. Point it at an existing experiment to continue it.",
        )
        # Continuity: if this experiment folder already holds sessions, offer to resume its settings
        # so a new subject can be recorded next day without re-entering every parameter.
        exp_dir = Path(default_save).expanduser()
        prior_sessions = list_saved_sessions(exp_dir) if exp_dir.exists() else []
        if prior_sessions:
            latest = prior_sessions[0]
            st.caption(f"📁 {len(prior_sessions)} prior session(s) — latest: {latest.name}")
            if st.button(
                "🔁 Continue experiment (load latest settings)",
                key="continue_experiment",
                width="stretch",
                help="Load the most recent session's parameters (not its data) so you can record the next subject.",
            ):
                params_content, _ = _load_session_contents(latest)
                if params_content:
                    load_params_into_state(normalize_params(params_content))
                    st.session_state["imported_data"] = None
                    st.session_state["use_imported_data"] = False
                    st.session_state["_flash"] = (
                        f"Loaded settings from {latest.name}. Update the participant, then start the recording."
                    )
                    st.rerun()
                else:
                    st.warning("Latest session has no params.json to resume from.")
        simulate = st.toggle(
            "Simulate run",
            value=False,
            help="Save generated zero data without connecting to hardware.",
        )

    with sidebar.container(border=True):
        st.subheader("Configuration & data")
        snapshot = current_params_snapshot()
        st.download_button(
            "⬇ Export settings (params.json)",
            data=params_to_json(snapshot) if snapshot else "{}",
            file_name=f"{(snapshot or {}).get('Method', 'cortipy')}_params.json",
            mime="application/json",
            disabled=snapshot is None,
            width="stretch",
            help="Download the current method/device/electrode configuration to reuse later.",
        )

        # Explicit recording export: container (BIDS/SBIDS) x raw format (Parquet/EDF).
        last = st.session_state.get("last_results") or {}
        can_export = last.get("data") is not None
        st.markdown("**Export recording**")
        exp_cols = st.columns(2)
        export_container = exp_cols[0].selectbox(
            "Container", ["SBIDS", "BIDS"], key="export_container", disabled=not can_export
        )
        export_fmt = exp_cols[1].selectbox(
            "Raw format", ["Parquet", "EDF"], key="export_raw_format", disabled=not can_export
        )
        if st.button(
            f"⬇ Export as {export_container} + {export_fmt}",
            disabled=not can_export,
            width="stretch",
            key="export_recording_btn",
        ):
            try:
                out = export_recording(
                    last.get("data"), last.get("params") or {}, Path(default_save).expanduser(),
                    export_container, export_fmt,
                )
                st.success(f"Exported {export_container} + {export_fmt} → {out}")
            except Exception as exc:  # pragma: no cover - surfaced to the user
                LOGGER.exception("Recording export failed")
                st.error(f"Export failed: {exc}")
        if not can_export:
            st.caption("Run or load a session first to enable recording export.")

        handle_upload(st)
        imported_data = st.session_state.get("imported_data")
        default_use_imported = st.session_state.get("use_imported_data", False) or bool(imported_data)
        use_imported_data = st.checkbox(
            "Use imported data for offline replay",
            value=default_use_imported and imported_data is not None,
            disabled=imported_data is None,
        )
        st.session_state["use_imported_data"] = use_imported_data and imported_data is not None

    with sidebar.container(border=True):
        st.subheader("Live view")
        live_view_enabled = st.toggle("During measurement", value=True)
        live_view_window = st.slider(
            "Window (s)",
            min_value=1,
            max_value=60,
            value=5,
            disabled=not live_view_enabled,
        )

    with sidebar.container(border=True):
        st.subheader("Run")
        if st.button("🔌 Test device connection", width="stretch",
                     help="Connect and confirm the device is streaming before starting a recording."):
            snap = current_params_snapshot()
            if snap is None:
                st.session_state["_device_check"] = ("warn", "Choose a method and device first.")
            else:
                with st.spinner("Connecting…"):
                    try:
                        ok, msg = verify_device_connection(snap)
                        st.session_state["_device_check"] = ("ok" if ok else "err", msg)
                    except Exception as exc:  # pragma: no cover - surfaced to the user
                        LOGGER.exception("Device connection test failed")
                        st.session_state["_device_check"] = ("err", str(exc))
        check = st.session_state.get("_device_check")
        if check:
            kind, msg = check
            {"ok": st.success, "warn": st.warning}.get(kind, st.error)(
                {"ok": "✅ ", "warn": "", "err": "❌ "}.get(kind, "") + msg
            )
        start_button = st.button("Start measurement", type="primary", width="stretch")
        st.caption("Tip: run **Test device connection** first, then Start. Use **Simulate run** above to try the flow without hardware.")

    with sidebar.expander("🩺 Diagnostics (logs)", expanded=False):
        st.caption(f"Log file: {LOG_PATH}")
        st.button("Refresh", key="refresh_logs")  # click triggers a rerun -> re-reads the log
        log_lines = _tail_log(LOG_PATH, 60)
        errs = [ln for ln in log_lines if "[ERROR]" in ln]
        warns = [ln for ln in log_lines if "[WARNING]" in ln]
        if errs:
            st.error("Recent errors:\n\n" + "\n".join(errs[-4:]))
        elif warns:
            st.warning("Recent warnings:\n\n" + "\n".join(warns[-4:]))
        st.code("\n".join(log_lines) if log_lines else "(log is empty)", language="log")

    return SidebarControls(
        default_save=default_save,
        simulate=simulate,
        live_view_enabled=live_view_enabled,
        live_view_window=int(live_view_window),
        start_button=start_button,
    )


def render_footer() -> None:
    st.divider()
    st.caption(
        "Research use only: CortiPy is not a medical device and must not be used for diagnosis or patient care. "
        "Validate latency/trigger behavior and device compatibility in your lab before clinical evaluation."
    )
    st.caption(
        "Protect privacy: avoid uploading or storing identifiable participant data. "
        "Keep run folders on secured systems and use anonymized or synthetic data when sharing."
    )


def render_workflow_progress(slot, params: Dict[str, Any], validation_issues: List[str]) -> None:
    """Compact top-of-page checklist guiding the user through a measurement workflow."""
    parameters = params.get("Parameters", {}) if isinstance(params, dict) else {}
    participant = (params.get("Metadata", {}) or {}).get("Participant", {}) or {}
    steps = [
        ("Method & device", bool(params.get("Method") and params.get("Device"))),
        ("Sampling rate", bool(parameters.get("fs"))),
        ("Electrodes", len(params.get("Channels", []) or []) > 0),
        ("Participant", bool(participant.get("Code"))),
    ]
    done = sum(1 for _, ok in steps if ok)
    ready = not validation_issues
    with slot:
        caption = "Ready to run — press Start measurement" if ready else "Complete the required fields to run"
        st.progress(done / len(steps), text=f"Setup {done}/{len(steps)} · {caption}")
        cols = st.columns(len(steps))
        for col, (label, ok) in zip(cols, steps):
            col.markdown(f"{'✅' if ok else '⬜'} {label}")


def main() -> None:
    st.set_page_config(page_title="cortipy UI", layout="wide")
    inject_global_styles(st)
    st.title("cortipy – EEG Measurement UI")
    ensure_state()

    flash = st.session_state.pop("_flash", None)
    if flash:
        st.success(flash)

    controls = render_sidebar_controls()
    default_save = controls.default_save
    simulate = controls.simulate
    live_view_enabled = controls.live_view_enabled
    live_view_window = controls.live_view_window
    start_button = controls.start_button
    imported_data = st.session_state.get("imported_data")

    page = st.segmented_control(
        "Section",
        VIEW_OPTIONS,
        default="Session configuration",
        key="active_view",
        label_visibility="collapsed",
        width="stretch",
    )
    if page is None:
        page = "Session configuration"

    # Elevated run-status + live-view region: measurement feedback renders here at the top of the
    # page (vivid start/end via st.status) instead of at the bottom where the run block executes.
    run_region = st.container()
    progress_slot = st.container()

    general_values = dict(st.session_state["general_form"])
    device_values = dict(st.session_state.setdefault("device_forms", {}).get(general_values.get("Device"), {}))
    method_values = dict(
        st.session_state["method_forms"].get(
            general_values.get("Method"),
            default_values(METHOD_SCHEMAS.get(general_values.get("Method"), [])),
        )
    )
    participant_values = dict(st.session_state["participant"])

    if page == "Session configuration":
        general_values = render_general_form()
        device_values = render_device_config(general_values["Device"])
        method_values = render_method_form(general_values["Method"])
        participant_values = render_participant_form()

    if page == "Electrodes":
        render_channel_editor(general_values["Device"])

    assembled_params = assemble_params(general_values, method_values, participant_values, device_values)
    validation_issues = validate_params(assembled_params)
    render_workflow_progress(progress_slot, assembled_params, validation_issues)

    if page == "Live preview":
        render_live_preview_tab(assembled_params, validation_issues)

    if page == "Preview":
        if validation_issues:
            st.warning(" • ".join(validation_issues))
        st.json(assembled_params)
        st.download_button(
            "Download params.json",
            data=params_to_json(assembled_params),
            file_name=f"{assembled_params['Method']}_params.json",
            mime="application/json",
        )

    if page == "Charts":
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

    if page == "Saved sessions":
        primary_payload, compare_payloads = render_saved_sessions(Path(default_save).expanduser())
        if primary_payload:
            label, params_loaded, data_loaded = primary_payload
            render_chart_section(label, params_loaded, data_loaded)
        if compare_payloads:
            render_comparison_charts(compare_payloads)

    if start_button:
        if validation_issues:
            issues_md = " • " + "\n • ".join(validation_issues)
            st.error(f"Please fix these configuration issues before starting a run:\n{issues_md}")
            render_footer()
            return
        channels_for_run = len(assembled_params.get("Channels", [])) or int(
            assembled_params.get("Parameters", {}).get("NumberEEGChannels") or 0
        )
        if channels_for_run <= 0:
            channels_for_run = DEVICE_DEFAULT_CHANNELS.get(assembled_params.get("Device"), 8)
        selection = st.session_state.get("live_preview_channel_selection") or ["All"]
        if "All" in selection or not selection:
            selected_indices = list(range(max(1, channels_for_run)))
        else:
            channel_labels = [row.get("Channel") or f"Ch {idx+1}" for idx, row in enumerate(assembled_params.get("Channels", []))] or [
                f"Ch {idx+1}" for idx in range(max(1, channels_for_run))
            ]
            selected_indices = [channel_labels.index(label) for label in selection if label in channel_labels]
        if not selected_indices:
            selected_indices = list(range(max(1, channels_for_run)))
        save_dir = Path(default_save).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        params_to_run = dict(assembled_params)
        params_to_run.pop("Evaluation", None)
        params_to_run.pop("data", None)
        # Position the live view inside the elevated top region (recreated each run).
        st.session_state["_live_view_placeholder"] = None
        with run_region, st.status("🔴 Measurement running…", expanded=True) as run_status:
            live_view_service: Optional[LiveViewService] = None
            if live_view_enabled:
                live_view_service = LiveViewService(
                    get_live_view_placeholder(),
                    window_seconds=float(live_view_window),
                    channel_indices=selected_indices,
                    fft_placeholder=st.session_state.get("_live_preview_fft_placeholder"),
                )
                st.session_state["_live_view_active"] = True
                st.session_state["_live_view_banner"] = "Measurement running: live EEG and FFT updating below."
            try:
                if simulate:
                    fs = int(assembled_params["Parameters"].get("fs", 250))
                    n_channels = max(1, int(
                        assembled_params["Parameters"].get("NumberEEGChannels", len(assembled_params.get("Channels", [])) or 8)
                    ))
                    params_to_run["data"] = np.zeros((fs, n_channels))
                    saved_path = SaveManager(save_dir)(params_to_run)
                    st.session_state["last_results"] = {
                        "label": getattr(saved_path, "name", "Simulated run"),
                        "params": params_to_run,
                        "data": params_to_run.get("data"),
                    }
                    run_status.update(label="✅ Simulated data saved — open the Charts tab", state="complete")
                else:
                    imported_data = st.session_state.get("imported_data")
                    use_imported = st.session_state.get("use_imported_data", False)
                    if use_imported and imported_data is None:
                        run_status.update(label="❌ No imported data attached", state="error")
                        st.error("Upload a data file or select a saved session first.")
                    else:
                        if use_imported and imported_data is not None:
                            params_to_run["Device"] = "Offline"
                            params_to_run["data"] = imported_data
                        run_params, saved_path = run_pipeline_once(params_to_run, save_dir, live_view=live_view_service)
                        st.session_state["last_results"] = {
                            "label": getattr(saved_path, "name", "Last run"),
                            "params": run_params,
                            "data": run_params.get("data"),
                        }
                        run_status.update(label="✅ Measurement finished and saved — open the Charts tab", state="complete")
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Measurement failed")
                run_status.update(label="❌ Measurement failed", state="error")
                st.error(f"Measurement failed: {exc}")
            finally:
                st.session_state["_live_view_active"] = False
                st.session_state["_live_view_banner"] = None
                if live_view_service is not None:
                    live_view_service.reset()

    render_footer()


if __name__ == "__main__":
    if hasattr(st, "runtime") and st.runtime.exists():
        main()  # Already inside a Streamlit runtime (e.g., `streamlit run ...`)
    else:  # Allow launching via `python apps/streamlit_app.py` (common on Windows)
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
