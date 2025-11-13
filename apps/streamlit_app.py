"""Streamlit UI for configuring and running cortipy sessions."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import streamlit as st
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

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


def render_general_form() -> Dict[str, Any]:
    general = st.session_state["general_form"]
    method_field = next(field for field in GENERAL_SCHEMA if field.name == "Method")
    device_field = next(field for field in GENERAL_SCHEMA if field.name == "Device")

    with st.container():
        left, right = st.columns(2)
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
        general["Method"] = method
        general["Device"] = device

    other_fields = [field for field in GENERAL_SCHEMA if field.name not in {"Method", "Device"}]
    cols = st.columns(2)
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
    return dict(general)


def render_device_config(device: str) -> Dict[str, Any]:
    schema = DEVICE_CONFIG_SCHEMA.get(device)
    if not schema:
        return {}
    device_forms = st.session_state.setdefault("device_forms", {})
    form_state = device_forms.setdefault(device, device_default_values(device))

    st.subheader(f"{device} device settings")
    st.caption("Configure hardware-specific parameters required to establish the live connection.")
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

    st.subheader(f"{method} parameters")
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
    st.subheader(f"Electrodes ({device})")
    st.caption(
        "Toggle the channels you intend to record, set the 10-20 name (Position), and choose the electrode hardware "
        "model for documentation. Ground/Reference rows stay enabled automatically."
    )
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
    extras = set(DEVICE_EXTRA_LABELS.get(device, []))
    for row in edited:
        if row["Channel"] in extras:
            row["Active"] = True
    channel_state[device] = edited
    return edited


def render_participant_form() -> Dict[str, Any]:
    st.subheader("Participant / proband information")
    participant = st.session_state["participant"]
    cols = st.columns(2)
    participant["Code"] = cols[0].text_input("Participant code", value=participant.get("Code", ""), placeholder="e.g. VEP_023")
    participant["Initials"] = cols[1].text_input("Initials", value=participant.get("Initials", ""))
    cols = st.columns(3)
    age_number = coerce_number(participant.get("Age"))
    age_default = int(age_number) if isinstance(age_number, (int, float)) and age_number > 0 else 0
    participant["Age"] = cols[0].number_input("Age", min_value=0, max_value=110, value=age_default)
    gender_value = resolve_choice(GENDER_OPTIONS, participant.get("Gender"))
    hand_value = resolve_choice(HANDEDNESS_OPTIONS, participant.get("DominantHand"))
    participant["Gender"] = cols[1].selectbox("Gender", options=GENDER_OPTIONS, index=GENDER_OPTIONS.index(gender_value))
    participant["DominantHand"] = cols[2].selectbox("Dominant hand", options=HANDEDNESS_OPTIONS, index=HANDEDNESS_OPTIONS.index(hand_value))
    participant["Notes"] = st.text_area("Session notes", value=participant.get("Notes", ""), height=80)
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


def run_pipeline_once(params: Dict[str, Any], save_dir: Path) -> None:
    saver = SaveManager(save_dir)
    provider_called = {"done": False}

    def provider(_: Optional[Dict[str, Any]]):
        if provider_called["done"]:
            return None
        provider_called["done"] = True
        return params

    hooks = PipelineHooks(params_provider=provider, save_callback=saver, should_continue=lambda _: False)
    MeasurementPipeline(hooks=hooks).run()


def render_saved_sessions(base_dir: Path) -> None:
    st.subheader("Saved sessions")
    refresh = st.button("Refresh list")
    if refresh:
        list_saved_sessions.clear()
    sessions = list_saved_sessions(base_dir)
    if not sessions:
        st.info("No sessions saved yet.")
        return
    selected = st.selectbox("Session directory", options=sessions, format_func=lambda p: p.name)
    if not selected:
        return
    params_path = selected / "params.json"
    data_path = selected / "data.npz"
    cols = st.columns(2)
    params_content: Optional[Dict[str, Any]] = None
    if params_path.exists():
        params_content = json.loads(params_path.read_text(encoding="utf-8"))
    with cols[0]:
        if params_content:
            st.caption("params.json")
            st.json(params_content)
            if st.button("Load session into editor", key=f"load_session_{selected.name}"):
                data_array = _load_npz_array(data_path) if data_path.exists() else None
                load_params_into_state(normalize_params(params_content), data_array)
                st.success("Session loaded. Review settings before running.")
        else:
            st.warning("params.json missing.")
    with cols[1]:
        if data_path.exists():
            st.caption("data.npz (arrays and shapes)")
            with np.load(data_path) as npz:
                st.write({name: arr.shape for name, arr in npz.items()})
        else:
            st.info("data.npz missing.")


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

    session_tab, electrodes_tab, preview_tab, saved_tab = st.tabs(
        ["Session configuration", "Electrodes", "Preview", "Saved sessions"]
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

    with saved_tab:
        render_saved_sessions(Path(default_save).expanduser())

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
            SaveManager(save_dir)(params_to_run)
            st.success("Simulated data saved.")
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
                run_pipeline_once(params_to_run, save_dir)
                st.success("Measurement finished and saved.")
            except Exception as exc:  # pragma: no cover
                st.error(f"Measurement failed: {exc}")


if __name__ == "__main__":
    main()
