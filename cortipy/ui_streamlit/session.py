"""Streamlit UI for configuring and running cortipy sessions."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
import html
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore
try:
    import mne  # type: ignore
except Exception:  # pragma: no cover
    mne = None

ROOT = Path(__file__).resolve().parents[2]
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

DEFAULT_SAVE_DIR = Path.cwd() / "cortipy_runs"
SCHEMA_DIR = ROOT / "apps" / "assets" / "ParameterJSON"


# Field schema + value coercion live in cortipy.ui_streamlit.fields (modularization).
from cortipy.ui_streamlit.fields import (  # noqa: E402
    FieldSchema,
    default_values,
    resolve_choice,
    is_integer_field,
    coerce_number,
    convert_value,
)


from cortipy.ui_streamlit.plot_windows import (  # noqa: E402
    render_matplotlib_window_launcher as _render_matplotlib_window_launcher,
    render_plotly_window_launcher as _render_plotly_window_launcher,
)
from cortipy.ui_streamlit.data_import import (  # noqa: E402
    load_edf_array as _load_edf_array,
    load_npz_array as _load_npz_array,
    load_parquet_array as _load_parquet_array,
    load_uploaded_data_array as _load_uploaded_data_array,
    params_from_jsonld_doc as _params_from_jsonld_doc_base,
)
from cortipy.ui_streamlit.live import (  # noqa: E402
    LiveViewService,
    _as_2d_array,
    _normalize_channel_indices,
    _plot_fft_spectrum,
    _plot_individual_channels,
    _plot_live_buffer,
    _reset_plot_window_open_state,
    _resolve_aux_channels,
    _selected_recording_seconds,
    run_live_preview,
)
from cortipy.ui_streamlit.electrodes import (  # noqa: E402
    actichamp_channel_count as _actichamp_channel_count,
    bump_channel_editor_revision as _bump_channel_editor_revision,
    map_impedances_to_channels as _map_impedances_to_channels,
    has_measured_impedance as _has_measured_impedance,
    impedance_range_kohm as _impedance_range_kohm,
)
from cortipy.ui_streamlit.workflow import (  # noqa: E402
    has_active_eeg_channels as _has_active_eeg_channels,
)


@dataclass(frozen=True)
class SidebarControls:
    default_save: str
    active_dataset_dir: Optional[str]
    simulate: bool
    live_view_enabled: bool
    live_view_window: int
    start_button: bool


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


def _valid_electrode_rubrik(value: Any) -> str:
    rubric = resolve_choice(ELECTRODE_RUBRICS, value)
    if rubric:
        return str(rubric)
    return ELECTRODE_RUBRICS[0] if ELECTRODE_RUBRICS else ""


def _model_for_rubrik(rubrik: Any, current: Any = None) -> str:
    rubric_key = _valid_electrode_rubrik(rubrik)
    models = ELECTRODE_LIBRARY.get(rubric_key, [])
    current_text = str(current or "").strip()
    if current_text in models:
        return current_text
    if models:
        return models[0]
    return current_text or (ELECTRODE_MODELS[0] if ELECTRODE_MODELS else "")

# Device / montage / field configuration lives in cortipy.ui_streamlit.constants (modularization).
from cortipy.ui_streamlit.constants import (  # noqa: E402
    DEVICE_CONFIG_SCHEMA,
    DEVICE_DEFAULT_CHANNELS,
    DEVICE_FIELD_ALIASES,
    DEVICE_EXTRA_LABELS,
    DEVICE_FS_OPTIONS,
    DEVICE_POSITION_DEFAULTS,
    SUPPORTED_EXTRA_DEVICES,
    VIEW_OPTIONS,
    device_default_values,
    normalize_position_label as _normalize_position_label,
    STANDARD_POSITION_ORDER as _STANDARD_POSITION_ORDER,
    POSITION_ANGLE_LOOKUP as _POSITION_ANGLE_LOOKUP,
)

APP_VIEW_OPTIONS = list(VIEW_OPTIONS)
if "Workflow" not in APP_VIEW_OPTIONS:
    APP_VIEW_OPTIONS.insert(0, "Workflow")

NAVIGATION_STATE_VERSION = "workflow_home_v1"


def set_active_view(page: str) -> None:
    if page not in APP_VIEW_OPTIONS:
        return
    st.session_state["active_view"] = page
    st.session_state["active_view_selector"] = page


def ensure_navigation_state() -> None:
    if st.session_state.get("_navigation_state_version") != NAVIGATION_STATE_VERSION:
        set_active_view("Workflow")
        st.session_state["_navigation_state_version"] = NAVIGATION_STATE_VERSION
        return

    selector_page = st.session_state.get("active_view_selector")
    active_page = st.session_state.get("active_view")
    if selector_page in APP_VIEW_OPTIONS:
        st.session_state["active_view"] = selector_page
    elif active_page in APP_VIEW_OPTIONS:
        st.session_state["active_view_selector"] = active_page
    else:
        set_active_view("Workflow")

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
    "Alpha": "Eyes-closed relaxation run to monitor 8-12 Hz activity.",
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
    }
}
.snapshot-panel {
    background: var(--card-bg);
    color: var(--value-color);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 0.95rem 1.05rem;
    box-shadow: 0 3px 10px var(--card-shadow);
}
.snapshot-panel .title {
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.snapshot-panel .desc {
    color: var(--label-color);
    margin-bottom: 0.55rem;
}
.snapshot-panel .line {
    margin-bottom: 0.2rem;
}
.snapshot-panel .stats {
    margin-top: 0.55rem;
    color: var(--label-color);
}
.participant-card {
    background: var(--card-bg);
    border-radius: 16px;
    border: 1px solid var(--card-border);
    padding: 0.95rem 1.05rem;
    box-shadow: 0 4px 12px var(--card-shadow);
    margin-bottom: 0.75rem;
}
.participant-card .summary-head {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 0.45rem;
    margin-bottom: 0.75rem;
    font-weight: 700;
    color: var(--value-color);
}
.participant-card .avatar {
    font-size: 22px;
    line-height: 1;
}
.participant-card .fields {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    column-gap: 0.9rem;
    row-gap: 0.55rem;
}
.participant-card .field {
    display: flex;
    gap: 0.65rem;
    margin-bottom: 0;
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
    margin-bottom: 0.12rem;
}
.participant-card .value {
    font-size: 0.95rem;
    color: var(--value-color);
    font-weight: 600;
}
.participant-card .notes {
    margin-top: 0.72rem;
    font-size: 0.85rem;
    color: var(--notes-color);
    padding-top: 0.62rem;
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



def _params_from_jsonld_doc(doc: Dict[str, Any], file_name: str = "import.jsonld") -> Optional[Dict[str, Any]]:
    return _params_from_jsonld_doc_base(
        doc,
        file_name,
        method_names=METHOD_SCHEMAS.keys(),
        electrode_library=ELECTRODE_LIBRARY,
        default_method="Alpha" if "Alpha" in METHOD_SCHEMAS else next(iter(METHOD_SCHEMAS), ""),
    )


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
        cbar.set_label("Impedance (kOhm)")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")
    ax.set_title("Scalp topography (positions + impedances)")
    fig.tight_layout()
    if hasattr(placeholder, "container"):
        plotly_fig = _plotly_topography(rows)
        if plotly_fig is not None:
            _render_plotly_window_launcher(
                plotly_fig,
                "Scalp topography",
                "scalp_topography",
                target=placeholder,
                button_label="Open scalp map",
            )
        else:
            _render_matplotlib_window_launcher(
                fig,
                "Scalp topography",
                "scalp_topography",
                target=placeholder,
                button_label="Open scalp map",
            )
    else:
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
        texts.append(f"{label}<br>Imp: {impedances[idx] if impedances[idx] is not None else 'N/A'} kOhm")
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
            colorbar=dict(title="Imp (kOhm)"),
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
            rubric = _valid_electrode_rubrik(ch.get("Rubrik") or ch.get("Rubric"))
            imported_rows.append(
                {
                    "Channel": channel_name,
                    "Position": ch.get("Position") or channel_name.replace(" ", ""),
                    "Rubrik": rubric,
                    "Model": _model_for_rubrik(rubric, ch.get("Model")),
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
        st.session_state.pop(f"device_{device}_{f['name']}_manual", None)
    for key in (
        "participant_Code",
        "participant_Initials",
        "participant_Age",
        "participant_Gender",
        "participant_DominantHand",
        "participant_Notes",
    ):
        st.session_state.pop(key, None)
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
    if "active_dataset_dir" not in st.session_state:
        st.session_state["active_dataset_dir"] = ""


def get_live_view_placeholder():
    placeholder = st.session_state.get("_live_view_placeholder")
    if placeholder is None:
        placeholder = st.empty()
        st.session_state["_live_view_placeholder"] = placeholder
    return placeholder


def _fetch_actichamp_impedances(fs_value: Any, *, force: bool = False) -> None:
    if st.session_state.get("_actichamp_impedance_loaded") and not force:
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
            f"Loaded {len(values)} impedance values ({low:.1f}-{high:.1f} kOhm)."
        )
    st.session_state["_actichamp_impedance_timestamp"] = time.time()


def _position_angle(label: str, fallback_idx: int) -> float:
    total = len(_STANDARD_POSITION_ORDER) or 1
    key = _normalize_position_label(label)
    idx = _POSITION_ANGLE_LOOKUP.get(key)
    if idx is None:
        idx = fallback_idx % total
    return 2 * math.pi * (idx / total)


def _safe_channel_coords(label: str, fallback_idx: int = 0) -> tuple[float, float]:
    pos_x, pos_y = _channel_default_coords(label)
    if pos_x is not None and pos_y is not None:
        return float(pos_x), float(pos_y)
    angle = _position_angle(label, fallback_idx)
    return float(0.8 * math.cos(angle)), float(0.8 * math.sin(angle))


def render_config_snapshot(method: str, device: str, general: Dict[str, Any]) -> None:
    method = method or "N/A"
    device = device or "N/A"
    full_name = METHOD_FULL_NAMES.get(method)
    desc = METHOD_DESCRIPTIONS.get(method)
    params_block = general if "Parameters" not in general else general.get("Parameters", {})
    fs_value = coerce_number(params_block.get("fs") or general.get("fs"))
    channels = coerce_number(params_block.get("NumberEEGChannels") or general.get("NumberEEGChannels"))
    duration = coerce_number(params_block.get("RecordingTime") or general.get("RecordingTime"))

    method_label = f"{method} - {full_name}" if full_name else method
    fs_txt = f"{fs_value} Hz" if fs_value is not None else "N/A"
    ch_txt = f"{int(channels)}" if channels is not None else "N/A"
    dur_txt = f"{duration} s" if duration is not None else "N/A"

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
        ("Code", participant.get("Code") or "N/A"),
        ("Initials", participant.get("Initials") or "N/A"),
        ("Age", participant.get("Age") or "N/A"),
        ("Gender", participant.get("Gender") or "N/A"),
        ("Dominant hand", participant.get("DominantHand") or "N/A"),
    ]
    info_html = "".join(
        f"<div class='field'><div><div class='label'>{label}</div>"
        f"<div class='value'>{html.escape(str(value))}</div></div></div>"
        for label, value in info_rows
    )
    notes_text = str(participant.get("Notes") or "").strip()
    notes_html = ""
    if notes_text:
        notes_html = f"<div class='notes'>Notes: {html.escape(notes_text)}</div>"
    card_html = (
        f"<div class='participant-card'><div class='summary-head'><span class='avatar'>&#9786;</span>"
        f"<span>Participant summary</span></div>"
        f"<div class='fields'>{info_html}</div>{notes_html}</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_general_form() -> Dict[str, Any]:
    general = st.session_state["general_form"]
    method_field = next(field for field in GENERAL_SCHEMA if field.name == "Method")
    device_field = next(field for field in GENERAL_SCHEMA if field.name == "Device")
    environment_field = next((field for field in GENERAL_SCHEMA if field.name == "Environment"), None)

    with st.expander("Session configuration", expanded=True):
        form_col, viz_col = st.columns((3, 2))
        top_cols = form_col.columns(3)
        method_options = method_field.options or sorted(METHOD_SCHEMAS.keys())
        method_value = resolve_choice(method_options, general.get("Method"))
        method = top_cols[0].selectbox(
            "Method",
            options=method_options,
            index=method_options.index(method_value),
            help=method_field.tooltip or None,
            key="general_Method",
        )
        device_options = sorted(set((device_field.options or []) + SUPPORTED_EXTRA_DEVICES))
        device_value = resolve_choice(device_options, general.get("Device"))
        device = top_cols[1].selectbox(
            "Device",
            options=device_options,
            index=device_options.index(device_value),
            help=device_field.tooltip or None,
            key="general_Device",
        )
        if environment_field is not None:
            env_options = environment_field.options or [""]
            env_value = resolve_choice(env_options, general.get("Environment"))
            environment = top_cols[2].selectbox(
                "Environment",
                options=env_options,
                index=env_options.index(env_value),
                help=environment_field.tooltip or None,
                key="general_Environment",
            )
            general["Environment"] = environment
        method_full = METHOD_FULL_NAMES.get(method)
        description = METHOD_DESCRIPTIONS.get(method)
        if method_full or description:
            caption_parts = [method_full or ""]
            if description:
                caption_parts.append(description)
            top_cols[0].caption(" - ".join(part for part in caption_parts if part))
        general["Method"] = method
        general["Device"] = device

    other_fields = [field for field in GENERAL_SCHEMA if field.name not in {"Method", "Device", "Environment"}]
    device_is_unicorn = device.lower() == "unicorn"
    ncols = 3 if len(other_fields) > 4 else 2  # denser grid = fewer rows to scroll
    cols = form_col.columns(ncols)
    for idx, field in enumerate(other_fields):
        target = cols[idx % ncols]
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
            render_config_snapshot(general.get("Method", ""), general.get("Device", ""), general)
    previous_device = st.session_state.get("_last_device_selection")
    device_changed = device != previous_device
    if device_changed:
        st.session_state["_last_device_selection"] = device
        st.session_state["_actichamp_impedance_loaded"] = False
        st.session_state["_actichamp_impedance_status"] = None
        st.session_state["_actichamp_impedance_timestamp"] = None

    return dict(general)



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
            if field.name == "ReferenceChannel":
                device = st.session_state.get("general_form", {}).get("Device", "")
                ref_options, ref_labels = _reference_channel_options(device)
                if ref_options:
                    current_ref = coerce_number(current)
                    resolved_ref = int(current_ref) if current_ref in ref_options else ref_options[0]
                    value = target.selectbox(
                        "Reference: all EEG - selected channel",
                        options=ref_options,
                        index=ref_options.index(resolved_ref),
                        format_func=lambda option: ref_labels.get(option, str(option)),
                        help="During analysis, every EEG channel is referenced as EEG channel minus this selected channel.",
                        key=key,
                    )
                else:
                    value = render_numeric_input(target, field, current, key)
            elif field.kind == "dropdown":
                options = field.options or [""]
                resolved = resolve_choice(options, current)
                value = target.selectbox(field.name, options=options, index=options.index(resolved), help=field.tooltip or None, key=key)
            elif field.kind == "numeric":
                value = render_numeric_input(target, field, current, key)
            else:
                value = target.text_input(field.name, value=current or "", help=field.tooltip or None, key=key)
            method_state[field.name] = value
    return dict(method_state)


def _reference_channel_options(device: str) -> tuple[List[int], Dict[int, str]]:
    rows = ensure_channel_rows(device, st.session_state["channel_tables"].get(device))
    extras = set(DEVICE_EXTRA_LABELS.get(device, []))
    eeg_rows = [row for row in rows if row.get("Channel") not in extras]
    active_rows = [row for row in eeg_rows if bool(row.get("Active"))]
    display_rows = active_rows or eeg_rows
    options: List[int] = []
    labels: Dict[int, str] = {}
    for eeg_index, row in enumerate(eeg_rows, start=1):
        if row not in display_rows:
            continue
        label = row.get("Position") or row.get("Channel") or f"Ch {eeg_index}"
        options.append(eeg_index)
        labels[eeg_index] = f"{eeg_index}: {label}"
    return options, labels


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
        default_model = _model_for_rubrik(default_rubric)
        position_label = label.replace(" ", "")
        if index is not None and index < len(default_positions):
            position_label = default_positions[index]
        pos_x, pos_y = _safe_channel_coords(position_label, index or 0)
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
        row["Rubrik"] = _valid_electrode_rubrik(row.get("Rubrik") or row.get("Rubric"))
        row["Model"] = _model_for_rubrik(row.get("Rubrik"), row.get("Model"))
        rows.append(row)

    for idx in range(base_count):
        label = f"Ch {idx + 1}"
        row = existing_map.get(label, base_row(label, False, idx))
        row["Rubrik"] = _valid_electrode_rubrik(row.get("Rubrik") or row.get("Rubric"))
        row["Model"] = _model_for_rubrik(row.get("Rubrik"), row.get("Model"))
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
            auto_cols = st.columns([1, 1, 2])
            continuous_impedance = auto_cols[0].toggle(
                "Continuous impedance refresh",
                value=bool(st.session_state.get("_actichamp_impedance_auto", False)),
                key="_actichamp_impedance_auto",
            )
            refresh_interval = auto_cols[1].number_input(
                "Refresh interval (s)",
                min_value=5,
                max_value=60,
                value=max(5, int(st.session_state.get("_actichamp_impedance_interval", 10) or 10)),
                step=1,
                key="_actichamp_impedance_interval",
                disabled=not continuous_impedance,
            )
            if continuous_impedance:
                run_every = float(refresh_interval)
                fragment = getattr(st, "fragment", None)
                if callable(fragment):
                    @fragment(run_every=run_every)
                    def _poll_actichamp_impedance() -> None:
                        before = st.session_state.get("_actichamp_impedance_timestamp")
                        _fetch_actichamp_impedances(fs_value, force=True)
                        after = st.session_state.get("_actichamp_impedance_timestamp")
                        message = st.session_state.get("_actichamp_impedance_status")
                        if message:
                            st.caption(message)
                        if after and after != before:
                            st.rerun()

                    _poll_actichamp_impedance()
                    auto_cols[2].caption(f"Polling every {int(run_every)}s while this page is open.")
                else:
                    now = time.time()
                    if not last_ts or now - float(last_ts) >= run_every:
                        _fetch_actichamp_impedances(fs_value, force=True)
                        rows = st.session_state["channel_tables"].get("ActiCHamp", rows)
                        editor_revision = int(st.session_state["_channel_editor_revision"].get(device, 0))
                    auto_cols[2].caption("Updates whenever Streamlit reruns while the toggle is enabled.")

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
                    row["PosX"], row["PosY"] = _safe_channel_coords(preset[eeg_idx], eeg_idx)
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

        # ReferenceChannel is a 1-based index over EEG channels. Analysis subtracts the
        # selected channel from every EEG channel; Ground/Reference extras are not EEG columns.
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
                "Reference channel",
                options=list(range(len(ref_labels))),
                format_func=lambda i: ref_labels[i],
                index=cur_idx,
                key=f"ref_electrode_{device}",
                help="Analysis uses all EEG channels minus this selected ReferenceChannel.",
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
                    "Model": st.column_config.TextColumn(
                        "Model",
                        width="large",
                        disabled=True,
                        help="Automatically selected from the electrode type (Rubrik).",
                    ),
                    "Impedance": st.column_config.NumberColumn(
                        "Impedance (kOhm)",
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
            corrected_model = False
            for row_idx, row in enumerate(edited):
                if row["Channel"] in extras:
                    row["Active"] = True
                if not row.get("Position"):
                    row["Position"] = row["Channel"].replace(" ", "")
                pos_x = coerce_number(row.get("PosX"))
                pos_y = coerce_number(row.get("PosY"))
                if pos_x is None or pos_y is None:
                    pos_x, pos_y = _safe_channel_coords(row["Position"], row_idx)
                row["PosX"] = float(pos_x)
                row["PosY"] = float(pos_y)
                row["Rubrik"] = _valid_electrode_rubrik(row.get("Rubrik") or row.get("Rubric"))
                next_model = _model_for_rubrik(row["Rubrik"], row.get("Model"))
                if row.get("Model") != next_model:
                    corrected_model = True
                row["Model"] = next_model
            channel_state[device] = edited
            if corrected_model:
                _bump_channel_editor_revision(device)
                st.rerun()

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
        _reset_plot_window_open_state(
            "live_preview_signal",
            "live_preview_fft",
            *[f"live_channel_{idx + 1}" for idx in selected_indices],
        )
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
        form_col, viz_col = st.columns((2.2, 0.9))
        with form_col:
            cols = form_col.columns([1.25, 1.0, 0.65, 1.0, 1.0])
            participant["Code"] = cols[0].text_input(
                "Participant code",
                value=participant.get("Code", ""),
                placeholder="e.g. VEP_023",
                key="participant_Code",
            )
            participant["Initials"] = cols[1].text_input(
                "Initials",
                value=participant.get("Initials", ""),
                key="participant_Initials",
            )
            age_number = coerce_number(participant.get("Age"))
            age_default = int(age_number) if isinstance(age_number, (int, float)) and age_number > 0 else 0
            participant["Age"] = cols[2].number_input(
                "Age",
                min_value=0,
                max_value=110,
                value=age_default,
                key="participant_Age",
            )
            gender_value = resolve_choice(GENDER_OPTIONS, participant.get("Gender"))
            hand_value = resolve_choice(HANDEDNESS_OPTIONS, participant.get("DominantHand"))
            participant["Gender"] = cols[3].selectbox(
                "Gender",
                options=GENDER_OPTIONS,
                index=GENDER_OPTIONS.index(gender_value),
                key="participant_Gender",
            )
            participant["DominantHand"] = cols[4].selectbox(
                "Dominant hand",
                options=HANDEDNESS_OPTIONS,
                index=HANDEDNESS_OPTIONS.index(hand_value),
                key="participant_DominantHand",
            )
            notes_col, _ = form_col.columns([3, 1])
            participant["Notes"] = notes_col.text_area(
                "Session notes",
                value=participant.get("Notes", ""),
                height=128,
                key="participant_Notes",
            )
        with viz_col:
            _render_participant_card(participant)
    return dict(participant)


def build_channels(device: str, requested_eeg_channels: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = st.session_state["channel_tables"].get(device) or ensure_channel_rows(device)
    extras = set(DEVICE_EXTRA_LABELS.get(device, []))
    if requested_eeg_channels and requested_eeg_channels > 0:
        active_eeg_rows = [
            row for row in rows
            if (row.get("Channel") or row.get("Label")) not in extras and bool(row.get("Active"))
        ]
        if not active_eeg_rows:
            remaining = int(requested_eeg_channels)
            synced_rows: List[Dict[str, Any]] = []
            for row in rows:
                updated = dict(row)
                channel_name = updated.get("Channel") or updated.get("Label")
                if channel_name in extras:
                    updated["Active"] = True
                elif remaining > 0:
                    updated["Active"] = True
                    remaining -= 1
                else:
                    updated["Active"] = False
                synced_rows.append(updated)
            rows = synced_rows
            st.session_state["channel_tables"][device] = rows
    active_rows: List[Dict[str, Any]] = []
    for row in rows:
        channel_name = row.get("Channel") or row.get("Label")
        if not channel_name:
            continue
        is_active = bool(row.get("Active")) or channel_name in extras
        rubric = _valid_electrode_rubrik(row.get("Rubrik") or row.get("Rubric"))
        entry = {
            "Channel": channel_name,
            "Position": row.get("Position") or channel_name.replace(" ", ""),
            "Rubrik": rubric,
            "Model": _model_for_rubrik(rubric, row.get("Model")),
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
    if any(field.name == "ReferenceChannel" for field in METHOD_SCHEMAS.get(params["Method"], [])):
        reference_value = coerce_number(params["Parameters"].get("ReferenceChannel"))
        if reference_value is None or reference_value < 1:
            params["Parameters"]["ReferenceChannel"] = 1
    if device_specific:
        for key, value in device_specific.items():
            if value in (None, "", []):
                continue
            params["Parameters"][key] = value
            for alias in DEVICE_FIELD_ALIASES.get(key, []):
                params["Parameters"][alias] = value

    if params["Device"].lower() == "unicorn":
        params["Parameters"]["fs"] = 250

    requested_channels_value = coerce_number(params["Parameters"].get("NumberEEGChannels"))
    requested_channels = int(requested_channels_value) if requested_channels_value and requested_channels_value > 0 else None
    channels = build_channels(params["Device"], requested_channels)
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
    if not _has_active_eeg_channels(params):
        issues.append("Select at least one EEG channel.")
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


def _connection_signature(params: Dict[str, Any]) -> str:
    parameters = params.get("Parameters", {}) if isinstance(params, dict) else {}

    def pick(*names: str) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for name in names:
            if name in parameters:
                values[name] = parameters.get(name)
            elif name in params:
                values[name] = params.get(name)
        return values

    payload: Dict[str, Any] = {"Device": str(params.get("Device") or "").strip().lower()}
    device = payload["Device"]
    if device == "unicorn":
        payload.update(
            pick(
                "fs",
                "NumberEEGChannels",
                "UNICORNPort",
                "UNICORNAddress",
                "UnicornPort",
                "UnicornAddress",
                "UNICORNDeviceName",
                "UnicornDeviceName",
                "UnicornTimeout",
                "UNICORNTimeout",
            )
        )
    elif device == "actichamp":
        payload.update(
            pick(
                "fs",
                "NumberEEGChannels",
                "NumberAUXChannels",
                "ActiChampPath",
                "actichampPath",
                "ActiChampTimeout",
                "UseActiveElectrodes",
                "IncludeTriggers",
            )
        )
    elif device == "lsl":
        payload.update(pick("fs", "NumberEEGChannels", "StreamName"))
    elif device in {"dummy", "sim", "simulation"}:
        payload.update(pick("fs", "NumberEEGChannels", "Noise"))
    return json.dumps(
        payload,
        sort_keys=True,
        default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else str(obj),
    )


def _disconnect_session_device() -> None:
    device = st.session_state.pop("_connected_device", None)
    st.session_state.pop("_connected_device_signature", None)
    if device is not None:
        try:
            device.disconnect()
        except Exception:
            LOGGER.debug("Disconnecting cached device raised", exc_info=True)


def _session_device_ready(params: Optional[Dict[str, Any]]) -> bool:
    if params is None:
        return False
    return (
        st.session_state.get("_connected_device") is not None
        and st.session_state.get("_connected_device_signature") == _connection_signature(params)
    )


def _requires_device_connection(
    params: Optional[Dict[str, Any]],
    *,
    simulate: bool,
    use_imported_data: bool,
    imported_data: Optional[np.ndarray],
) -> bool:
    if params is None or simulate:
        return False
    if use_imported_data and imported_data is not None:
        return False
    device = str(params.get("Device") or "").strip().lower()
    return device not in {"offline", "file"}


def connect_device_for_run(params: Dict[str, Any]) -> tuple[DeviceInterface, str]:
    """Connect to the configured device, probe a short window, and keep it ready."""
    device = DeviceFactory.create(params)
    aux = _resolve_aux_channels(params)
    probe_s = 0.5
    try:
        device.connect()
        sample = _as_2d_array(device.acquire(probe_s, aux))
        if sample.size == 0 or sample.shape[0] == 0:
            raise RuntimeError("Connected, but no samples were received (device may still be settling).")
        eff_fs = sample.shape[0] / probe_s
        selected_s = _selected_recording_seconds(params)
        selected_text = f"Selected recording: {selected_s:.0f}s" if selected_s else "Selected recording length not set"
        return device, f"Streaming - {sample.shape[1]} channels, ~{eff_fs:.0f} Hz. {selected_text}."
    except Exception:
        try:
            device.disconnect()
        except Exception:
            LOGGER.debug("Device disconnect after failed connect raised", exc_info=True)
        raise


def verify_device_connection(params: Dict[str, Any]) -> tuple[bool, str]:
    """Connect to the configured device and confirm it streams a short window.

    Returns (ok, message). Used as a pre-flight 'first-connect confirm' so a recording
    isn't started against a device that hasn't actually come up / is still settling.
    """
    device: Optional[DeviceInterface] = None
    try:
        device, msg = connect_device_for_run(params)
        return True, msg
    finally:
        if device is not None:
            try:
                device.disconnect()
            except Exception:  # pragma: no cover - best-effort cleanup
                LOGGER.debug("Device disconnect after verify raised", exc_info=True)


def export_recording(data: Any, params: Dict[str, Any], out_dir: Path, container: str, raw_format: str) -> Path:
    """Export a recording as BIDS or JSON-LD metadata with a selectable raw layer."""
    from cortipy.shared.bids import BIDSLoader, BIDSLoadResult, _coerce_to_raw_array

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Recording data must be 2-D (samples x channels).")
    pblock = params.get("Parameters", {}) if isinstance(params, dict) else {}
    fs = float(coerce_number(pblock.get("fs")) or 250.0)
    chans = params.get("Channels") or []
    ch_names = [(c.get("Position") or c.get("Channel")) for c in chans] or None
    if ch_names and len(ch_names) != arr.shape[1]:
        ch_names = None  # fall back to Ch1..N when the montage doesn't match the data width
    part = (params.get("Metadata") or {}).get("Participant") or {}
    subject = (str(part.get("Code") or "01").replace(" ", "") or "01")
    task = str(params.get("Method") or "task").lower()
    fmt = raw_format.lower()
    out_dir = Path(out_dir)
    container_key = str(container or "").upper().replace("-", "").replace("_", "").replace(" ", "")

    if container_key == "BIDS":
        if fmt == "npz":
            raise ValueError(
                "BIDS export does not support NPZ raw data. "
                "Choose JSON-LD + NPZ, or use BIDS + Parquet/EDF."
            )
        root = out_dir / "bids_export"
        BIDSLoader(root).to_bids(
            arr, sampling_rate=fs, ch_names=ch_names, subject=subject, task=task,
            format=fmt, overwrite=True, dataset_description={"Name": params.get("Method") or "CortiPy export"},
        )
        return root

    from cortipy.shared.dataset import CortiDataset  # JSON-LD/SBIDS path
    raw = _coerce_to_raw_array(arr, fs, ch_names, None)
    result = BIDSLoadResult(
        raw=raw, data=arr, sampling_rate=fs, events=None, channels=None,
        metadata={"params": params}, source_path=Path(f"sub-{subject}_{task}_scalpdata"), ancillary_files=[],
    )
    out = out_dir / "jsonld_export" / f"sbids_meta_sub-{subject}.jsonld"
    out.parent.mkdir(parents=True, exist_ok=True)
    CortiDataset(result).to_sbids(out, export_format=fmt)
    return out.parent


def simulated_recording_data(params: Dict[str, Any]) -> np.ndarray:
    parameters = params.get("Parameters", {}) if isinstance(params, dict) else {}
    fs = float(coerce_number(parameters.get("fs")) or 250.0)
    recording_seconds = _selected_recording_seconds(params) or 1.0
    n_channels_value = coerce_number(
        parameters.get("NumberEEGChannels")
        or len(params.get("Channels", []))
        or 8
    )
    n_channels = max(1, int(n_channels_value or 8))
    n_samples = max(1, int(round(fs * recording_seconds)))
    return np.zeros((n_samples, n_channels), dtype=float)


@st.cache_data
def list_saved_sessions(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    return sorted([path for path in base_dir.iterdir() if path.is_dir()], reverse=True)


def run_pipeline_once(
    params: Dict[str, Any],
    save_dir: Path,
    live_view: Optional[LiveViewService] = None,
    connected_device: Optional[DeviceInterface] = None,
    target_dir: Optional[Path] = None,
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
        saver(run_params, target_dir=target_dir)
        captured["path"] = getattr(saver, "last_target_dir", None)

    def attach_context(context):
        if live_view is not None:
            context.attach_service("live_view", live_view)
        if connected_device is not None:
            if live_view is not None:
                context.device = live_view.wrap_device(connected_device, context.params)
            else:
                context.device = connected_device

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


def _load_data_file_path(path: Path) -> Optional[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz_array(path)
    if suffix == ".parquet":
        return _load_parquet_array(path)
    if suffix == ".edf":
        return _load_edf_array(path)
    return None


def _find_first_existing_file(base_dir: Path, candidates: List[str]) -> Optional[Path]:
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists() and path.is_file():
            return path
    return None


def _load_dataset_folder(dataset_dir: Path) -> tuple[Optional[Dict[str, Any]], Optional[np.ndarray], str]:
    """Load params/metadata plus raw data from an existing dataset folder."""
    dataset_dir = Path(dataset_dir).expanduser()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return None, None, f"Dataset folder not found: {dataset_dir}"

    params_content: Optional[Dict[str, Any]] = None
    source_label = ""
    params_path = _find_first_existing_file(dataset_dir, ["params.json"])
    if params_path is None:
        jsonld_files = sorted(dataset_dir.rglob("*.jsonld"))
        if jsonld_files:
            params_path = jsonld_files[0]
    if params_path is None:
        json_files = sorted(path for path in dataset_dir.glob("*.json") if path.name.lower() != "dataset_description.json")
        if json_files:
            params_path = json_files[0]

    if params_path is not None:
        try:
            parsed = json.loads(params_path.read_text(encoding="utf-8"))
            if params_path.suffix.lower() == ".jsonld":
                params_content = _params_from_jsonld_doc(parsed, params_path.name)
            else:
                params_content = normalize_params(parsed)
            source_label = params_path.name
        except Exception as exc:
            return None, None, f"Failed to load settings from {params_path.name}: {exc}"

    data_array: Optional[np.ndarray] = None
    data_path: Optional[Path] = None
    if params_content:
        raw_name = params_content.get("DataFile")
        if raw_name:
            raw_path = Path(str(raw_name))
            raw_candidates = [
                dataset_dir / raw_path,
                dataset_dir / "raw_data" / raw_path.name,
                dataset_dir / "jsonld_export" / "raw_data" / raw_path.name,
            ]
            data_path = next((path for path in raw_candidates if path.exists() and path.is_file()), None)
    if data_path is None:
        data_path = _find_first_existing_file(dataset_dir, ["data.npz"])
    if data_path is None:
        for pattern in ("*.npz", "*.parquet", "*.edf"):
            matches = sorted(dataset_dir.rglob(pattern))
            if matches:
                data_path = matches[0]
                break
    if data_path is not None:
        data_array = _load_data_file_path(data_path)

    if params_content is None:
        return None, data_array, f"No params.json, JSON config, or JSON-LD metadata found in {dataset_dir}."

    data_note = f" with {data_path.name}" if data_path is not None and data_array is not None else ""
    return params_content, data_array, f"Loaded {source_label or 'settings'}{data_note} from {dataset_dir.name}."


def active_dataset_path(default_save: str | Path) -> Optional[Path]:
    raw = str(st.session_state.get("active_dataset_dir") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(default_save).expanduser() / path
    return path


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
                st.session_state["active_dataset_dir"] = str(primary)
                st.session_state.pop("active_dataset_dir_input", None)
                st.session_state["last_results"] = {
                    "label": primary.name,
                    "params": normalize_params(params_content),
                    "data": data_array,
                }
                st.session_state["_flash"] = "Session loaded. This folder is now the active dataset target."
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
    with target.expander("Load settings / data", expanded=False):
        uploaded = st.file_uploader(
            "Load JSON/TOML config, params.json, or JSON-LD metadata",
            type=["json", "jsonld", "toml", "tml"],
            key="config_uploader",
        )
        data_upload = st.file_uploader(
            "Attach data file(s): .npz, .parquet, .edf, or JSON-LD + raw file",
            type=["npz", "parquet", "edf", "jsonld"],
            accept_multiple_files=True,
            key="data_uploader_v2",
        )

        if uploaded:
            config_bytes = uploaded.getvalue()
            config_digest = f"{uploaded.name}:{hashlib.sha1(config_bytes).hexdigest()}"
            if st.session_state.get("_last_config_upload_digest") == config_digest:
                st.caption(f"Loaded '{uploaded.name}'. Upload a different file to apply new settings.")
            else:
                suffix = Path(uploaded.name).suffix.lower()
                try:
                    if suffix in {".toml", ".tml"}:
                        config = tomllib.loads(config_bytes.decode("utf-8"))
                        params = normalize_params(config)
                    elif suffix == ".jsonld":
                        config = json.loads(config_bytes.decode("utf-8"))
                        params = _params_from_jsonld_doc(config, uploaded.name)
                        if params is None:
                            return
                    else:
                        config = json.loads(config_bytes.decode("utf-8"))
                        params = normalize_params(config)
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to parse uploaded config: {exc}")
                else:
                    load_params_into_state(params)
                    st.session_state["_last_config_upload_digest"] = config_digest
                    st.session_state["_flash"] = f"Imported parameters from '{uploaded.name}'."
                    st.rerun()

        if data_upload:
            uploads = list(data_upload) if isinstance(data_upload, list) else [data_upload]
            jsonld_upload = next((item for item in uploads if Path(item.name).suffix.lower() == ".jsonld"), None)
            data_file = next((item for item in uploads if Path(item.name).suffix.lower() in {".npz", ".parquet", ".edf"}), None)
            upload_digest = hashlib.sha1(
                b"".join(item.name.encode("utf-8") + b"\0" + item.getvalue() for item in uploads)
            ).hexdigest()
            already_loaded = st.session_state.get("_last_data_upload_digest") == upload_digest
            if already_loaded:
                st.caption("Attached data already loaded. Upload different files to replace it.")
                return
            params_from_jsonld: Optional[Dict[str, Any]] = None
            if jsonld_upload is not None:
                try:
                    params_from_jsonld = _params_from_jsonld_doc(
                        json.loads(jsonld_upload.getvalue().decode("utf-8")),
                        jsonld_upload.name,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to parse uploaded JSON-LD: {exc}")
                    params_from_jsonld = None
            data_array = _load_uploaded_data_array(data_file) if data_file is not None else None
            if data_array is not None:
                st.session_state["imported_data"] = data_array
                st.session_state["use_imported_data"] = True
                st.success(f"Attached data from '{data_file.name}'.")
                st.session_state["_last_data_upload_digest"] = upload_digest
            if params_from_jsonld is not None:
                load_params_into_state(params_from_jsonld, data_array)
                st.session_state["_last_data_upload_digest"] = upload_digest
                st.session_state["_flash"] = f"Imported JSON-LD metadata from '{jsonld_upload.name}'."
                st.rerun()

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

        exp_dir = Path(default_save).expanduser()
        prior_sessions = list_saved_sessions(exp_dir) if exp_dir.exists() else []

        active_default = str(st.session_state.get("active_dataset_dir") or "")
        active_dataset_value = st.text_input(
            "Active dataset folder",
            value=active_default,
            help=(
                "Optional. When set, Start measurement saves params.json, data.npz, and exports into this folder "
                "instead of creating a new timestamped run folder."
            ),
            placeholder="Leave empty for a new timestamped run folder",
            key="active_dataset_dir_input",
        ).strip()
        st.session_state["active_dataset_dir"] = active_dataset_value
        dataset_path = active_dataset_path(default_save)
        dataset_cols = st.columns(2)
        if dataset_cols[0].button(
            "Load dataset folder",
            key="load_active_dataset_folder",
            width="stretch",
            disabled=dataset_path is None,
            help="Load params.json or JSON-LD metadata, plus data.npz/parquet/edf when present.",
        ):
            assert dataset_path is not None
            params_content, data_array, message = _load_dataset_folder(dataset_path)
            if params_content:
                load_params_into_state(params_content, data_array)
                st.session_state["active_dataset_dir"] = str(dataset_path)
                st.session_state["last_results"] = {
                    "label": dataset_path.name,
                    "params": params_content,
                    "data": data_array,
                }
                st.session_state["_flash"] = message
                st.rerun()
            else:
                st.warning(message)
        if dataset_cols[1].button(
            "Clear active folder",
            key="clear_active_dataset_folder",
            width="stretch",
            disabled=dataset_path is None,
        ):
            st.session_state["active_dataset_dir"] = ""
            st.session_state.pop("active_dataset_dir_input", None)
            st.rerun()
        if dataset_path is not None:
            st.caption(f"Active target: {dataset_path}")

        if prior_sessions:
            latest = prior_sessions[0]
            st.caption(f"{len(prior_sessions)} prior session(s) - latest: {latest.name}")

            if st.button(
                "Continue experiment (load latest settings)",
                key="continue_experiment",
                width="stretch",
                help="Load the most recent session's parameters (not its data) so you can record the next subject.",
            ):
                params_content, _ = _load_session_contents(latest)

                if params_content:
                    load_params_into_state(normalize_params(params_content))
                    st.session_state["imported_data"] = None
                    st.session_state["use_imported_data"] = False
                    st.session_state["active_dataset_dir"] = ""
                    st.session_state.pop("active_dataset_dir_input", None)
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
            "Export settings (params.json)",
            data=params_to_json(snapshot) if snapshot else "{}",
            file_name=f"{(snapshot or {}).get('Method', 'cortipy')}_params.json",
            mime="application/json",
            disabled=snapshot is None,
            width="stretch",
            help="Download the current method/device/electrode configuration to reuse later.",
        )

        last = st.session_state.get("last_results") or {}
        imported_data = st.session_state.get("imported_data")
        last_data = last.get("data")
        export_data = last_data if last_data is not None else imported_data
        export_params = last.get("params") or snapshot or st.session_state.get("imported_params_raw") or {}
        has_export_params = bool(export_params)
        has_export_data = export_data is not None

        st.markdown("**Export recording**")

        exp_cols = st.columns(2)
        export_container_options = ["JSON-LD", "BIDS"]
        if st.session_state.get("export_container") == "SBIDS":
            st.session_state["export_container"] = "JSON-LD"
        if st.session_state.get("export_container") not in (None, *export_container_options):
            st.session_state["export_container"] = "JSON-LD"

        export_container = exp_cols[0].selectbox(
            "Container",
            export_container_options,
            key="export_container",
        )

        export_format_options = (
            ["Parquet", "EDF", "NPZ"]
            if export_container == "JSON-LD"
            else ["Parquet", "EDF"]
        )
        if st.session_state.get("export_raw_format") not in export_format_options:
            st.session_state["export_raw_format"] = "Parquet"

        export_fmt = exp_cols[1].selectbox(
            "Raw format",
            export_format_options,
            key="export_raw_format",
        )

        if st.button(
            f"Export as {export_container} + {export_fmt}",
            disabled=not has_export_params,
            width="stretch",
            key="export_recording_btn",
        ):
            if not has_export_data:
                st.info(
                    f"{export_container} + {export_fmt} is selected for the next run. "
                    "Run or import data to write an export immediately."
                )
            else:
                try:
                    export_dir = active_dataset_path(default_save) or Path(default_save).expanduser()
                    export_dir.mkdir(parents=True, exist_ok=True)
                    out = export_recording(
                        export_data,
                        export_params,
                        export_dir,
                        export_container,
                        export_fmt,
                    )
                    st.success(f"Exported {export_container} + {export_fmt} -> {out}")
                except Exception as exc:
                    LOGGER.exception("Recording export failed")
                    st.error(f"Export failed: {exc}")

        if has_export_data:
            source_label = "last run" if last_data is not None else "imported data"
            st.caption(f"Use the button to write another copy of the {source_label} in the selected export format.")
        elif has_export_params:
            st.caption("Click the export button to keep these settings for the next run; no recording data is attached yet.")
        else:
            st.caption(
                "Choose export settings now. Each new run writes this format into the run folder; "
                "run or load a session to enable manual re-export."
            )

        handle_upload(st)

        imported_data = st.session_state.get("imported_data")
        default_use_imported = st.session_state.get("use_imported_data", False) or bool(imported_data)

        use_imported_data = st.checkbox(
            "Use imported data for offline replay",
            value=default_use_imported and imported_data is not None,
            disabled=imported_data is None,
            key="use_imported_data_checkbox",
        )

        st.session_state["use_imported_data"] = use_imported_data and imported_data is not None

    with sidebar.container(border=True):
        st.subheader("Live view")

        live_view_enabled = st.toggle(
            "During measurement",
            value=True,
            key="live_view_enabled_toggle",
        )

        live_view_window = st.slider(
            "Window (s)",
            min_value=1,
            max_value=60,
            value=5,
            disabled=not live_view_enabled,
            key="live_view_window_slider",
        )

    with sidebar.container(border=True):
        st.subheader("Run")
        snap = current_params_snapshot()
        imported_data = st.session_state.get("imported_data")
        use_imported = st.session_state.get("use_imported_data", False)
        connection_required = _requires_device_connection(
            snap,
            simulate=simulate,
            use_imported_data=use_imported,
            imported_data=imported_data,
        )

        if not connection_required and st.session_state.get("_connected_device") is not None:
            _disconnect_session_device()
            st.session_state["_device_check"] = (
                "warn",
                "Hardware connection released because this mode does not require it.",
            )
        connected_device_cached = st.session_state.get("_connected_device") is not None
        if snap is not None and connected_device_cached and not _session_device_ready(snap):
            _disconnect_session_device()
            st.session_state["_device_check"] = ("warn", "Configuration changed. Connect the device again before starting.")

        if st.button(
            "Connect device",
            width="stretch",
            key="test_device_connection",
            disabled=snap is None or not connection_required,
            help="Connect, probe, and keep the device ready so Start begins without another connection handshake.",
        ):
            if snap is None:
                st.session_state["_device_check"] = ("warn", "Choose a method and device first.")
            else:
                _disconnect_session_device()
                with st.spinner("Connecting..."):
                    try:
                        device, msg = connect_device_for_run(snap)
                        st.session_state["_connected_device"] = device
                        st.session_state["_connected_device_signature"] = _connection_signature(snap)
                        st.session_state["_device_check"] = ("ok", f"{msg} Ready to start.")
                    except Exception as exc:
                        LOGGER.exception("Device connection failed")
                        st.session_state["_device_check"] = ("err", str(exc))

        check = st.session_state.get("_device_check")

        if check:
            kind, msg = check
            {"ok": st.success, "warn": st.warning}.get(kind, st.error)(
                {"ok": "[ok] ", "warn": "", "err": "[error] "}.get(kind, "") + msg
            )

        connection_ready = _session_device_ready(snap)
        start_disabled = snap is None or (connection_required and not connection_ready)
        start_button = st.button(
            "Start measurement",
            type="primary",
            width="stretch",
            key="start_measurement",
            disabled=start_disabled,
        )

        if connection_required:
            if connection_ready:
                st.caption("Device is connected. Start begins acquisition using the live connection.")
            else:
                st.caption("Connect the device first. Simulate run and offline replay do not require hardware.")
        else:
            st.caption("This mode does not require a hardware connection.")

    # ONLY ONE diagnostics block (FIXED)
    with sidebar.expander("Diagnostics (logs)", expanded=False):
        st.caption(f"Log file: {LOG_PATH}")

        if st.button("Refresh logs", key="refresh_logs_button"):
            st.rerun()

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
        active_dataset_dir=str(active_dataset_path(default_save)) if active_dataset_path(default_save) else None,
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
