"""Data and SBIDS JSON-LD import helpers for the Streamlit UI."""

from __future__ import annotations

import re
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st

from cortipy.shared.bids import raw_to_microvolts
from cortipy.shared.sbids import RECORDING_PROPERTIES
from cortipy.ui_streamlit.fields import coerce_number, resolve_choice


def load_npz_array(source: Union[Path, Any]) -> Optional[np.ndarray]:
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


def load_parquet_array(source: Union[Path, Any]) -> Optional[np.ndarray]:
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        frame = pd.read_parquet(source)
    except Exception as exc:
        st.error(f"Failed to load Parquet data: {exc}")
        return None
    return frame.to_numpy(dtype=float, copy=False)


def load_edf_array(source: Union[Path, Any]) -> Optional[np.ndarray]:
    try:
        import mne
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to load EDF data: mne is required ({exc})")
        return None

    temp_path: Optional[Path] = None
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
        else:
            suffix = Path(getattr(source, "name", "")).suffix or ".edf"
            if hasattr(source, "seek"):
                source.seek(0)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(source.read())
                temp_path = Path(tmp.name)
            path = temp_path
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
        return raw_to_microvolts(raw)
    except Exception as exc:
        st.error(f"Failed to load EDF data: {exc}")
        return None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def load_uploaded_data_array(uploaded: Any) -> Optional[np.ndarray]:
    suffix = Path(getattr(uploaded, "name", "")).suffix.lower()
    if suffix == ".npz":
        return load_npz_array(uploaded)
    if suffix == ".parquet":
        return load_parquet_array(uploaded)
    if suffix == ".edf":
        return load_edf_array(uploaded)
    st.error(f"Unsupported data file type: {suffix or 'unknown'}")
    return None


def jsonld_value(value: Any) -> Any:
    if isinstance(value, list):
        return jsonld_value(value[0]) if value else None
    if isinstance(value, dict):
        if "@value" in value:
            return value.get("@value")
        if "@id" in value:
            return value.get("@id")
        if "name" in value:
            return value.get("name")
        if "schema:name" in value:
            return jsonld_value(value.get("schema:name"))
    return value


def jsonld_types(node: Dict[str, Any]) -> set[str]:
    raw = node.get("@type") or node.get("type") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(item).split(":")[-1].lower() for item in raw}


def jsonld_node_id(node: Dict[str, Any]) -> Optional[str]:
    node_id = node.get("@id") or node.get("id")
    return str(node_id) if node_id not in (None, "") else None


def duration_seconds(value: Any) -> Optional[float]:
    raw = str(jsonld_value(value) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    match = re.fullmatch(
        r"P(?:(?P<days>\d+(?:\.\d+)?)D)?(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?",
        raw,
    )
    if not match:
        return None
    total = 0.0
    total += float(match.group("days") or 0.0) * 86400.0
    total += float(match.group("hours") or 0.0) * 3600.0
    total += float(match.group("minutes") or 0.0) * 60.0
    total += float(match.group("seconds") or 0.0)
    return total if total > 0 else None


def additional_property_lookup(node: Dict[str, Any]) -> Dict[str, Any]:
    props = node.get("schema:additionalProperty") or node.get("additionalProperty") or []
    if isinstance(props, dict):
        props = [props]
    lookup: Dict[str, Any] = {}
    for prop in props:
        if not isinstance(prop, dict):
            continue
        name = jsonld_value(prop.get("schema:name") or prop.get("name"))
        value = jsonld_value(prop.get("schema:value") or prop.get("value"))
        if name not in (None, ""):
            lookup[str(name)] = value
    return lookup


def _valid_electrode_rubrik(electrode_library: Mapping[str, List[str]], value: Any) -> str:
    rubrics = list(electrode_library.keys())
    rubric = resolve_choice(rubrics, value)
    if rubric:
        return str(rubric)
    return rubrics[0] if rubrics else ""


def _model_for_rubrik(electrode_library: Mapping[str, List[str]], rubrik: Any, current: Any = None) -> str:
    rubric_key = _valid_electrode_rubrik(electrode_library, rubrik)
    models = electrode_library.get(rubric_key, [])
    current_text = str(current or "").strip()
    if current_text in models:
        return current_text
    if models:
        return models[0]
    return current_text


def _strip_bids_prefixes(stem: str) -> str:
    """Drop BIDS entity prefixes (``sub-``, ``ses-``…) and the sbids_meta_ stem.

    The exporter names its file ``sbids_meta_sub-01.jsonld``; feeding that stem straight
    back into ``Filename`` made the next export emit ``urn:recording:01_sbids_meta_sub-01``
    — the subject appearing twice, once bare and once ``sub-``-prefixed.
    """
    text = str(stem or "")
    for marker in ("sbids_meta_", "sbids_"):
        if text.startswith(marker):
            text = text[len(marker) :]
    parts = [p for p in text.split("_") if p]
    cleaned = [p for p in parts if not re.match(r"^(sub|ses|task|run|acq)-", p, flags=re.IGNORECASE)]
    return "_".join(cleaned) if cleaned else text


def _is_aux_channel_node(node: Mapping[str, Any], position: str) -> bool:
    """AUXChannel-typed nodes, or the GND/REF positions the exporter types that way."""
    types = jsonld_types(node)
    if "auxchannel" in types:
        return True
    return str(position or "").strip().upper() in {"GND", "REF"}


def _device_from_graph(recording: Mapping[str, Any], id_map: Mapping[str, Any]) -> str:
    """Resolve schema:instrument to the device name the exporter recorded."""
    ref = jsonld_value(recording.get("schema:instrument") or recording.get("instrument"))
    if not ref:
        return ""
    node = id_map.get(str(ref)) or {}
    name = jsonld_value(node.get("schema:name") or node.get("name"))
    if name:
        return str(name)
    # Fall back to the URN tail: urn:device:UNICORN -> UNICORN
    return str(ref).rsplit(":", 1)[-1]


def params_from_jsonld_doc(
    doc: Dict[str, Any],
    file_name: str = "import.jsonld",
    *,
    method_names: Iterable[str],
    electrode_library: Mapping[str, List[str]],
    default_method: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    graph = doc.get("@graph") if isinstance(doc, dict) else None
    nodes = graph if isinstance(graph, list) else [doc]
    nodes = [node for node in nodes if isinstance(node, dict)]
    id_map = {jsonld_node_id(node): node for node in nodes if jsonld_node_id(node)}

    recording = None
    for node in nodes:
        types = jsonld_types(node)
        if "createaction" in types or "recording" in types or node.get("schema:result") or node.get("result"):
            recording = node
            break
    recording = recording or (nodes[0] if nodes else {})
    if not recording:
        st.error("JSON-LD import did not contain a recording node.")
        return None

    result_ref = jsonld_value(recording.get("schema:result") or recording.get("result"))
    file_node = id_map.get(str(result_ref), {}) if result_ref else {}
    if not file_node:
        for node in nodes:
            types = jsonld_types(node)
            if "mediaobject" in types or "digitaldocument" in types or node.get("schema:contentUrl") or node.get("contentUrl"):
                file_node = node
                break

    prop_lookup = additional_property_lookup(recording)
    prop_lookup.update(additional_property_lookup(file_node))

    def prop(*names: str) -> Any:
        for name in names:
            if name in prop_lookup:
                return prop_lookup[name]
            lowered = name.lower()
            for key, value in prop_lookup.items():
                if key.lower() == lowered:
                    return value
        return None

    fs = coerce_number(prop("SamplingRate", "SamplingFrequency", "sfreq", "sfreq_Hz"))
    duration = duration_seconds(recording.get("schema:duration") or recording.get("duration"))
    n_eeg = coerce_number(prop("NumberEEGChannels", "EEGChannels", "channels"))
    n_aux = coerce_number(prop("NumberAUXChannels", "AUXChannels"))

    variables = recording.get("schema:variableMeasured") or recording.get("variableMeasured") or []
    if isinstance(variables, dict):
        variables = [variables]
    channels: List[Dict[str, Any]] = []
    aux_electrodes: List[Dict[str, Any]] = []
    for idx, item in enumerate(variables):
        if isinstance(item, dict):
            position = jsonld_value(item.get("schema:name") or item.get("name")) or f"Ch {idx + 1}"
            column = jsonld_value(item.get("columnName")) or position
            ch_props = additional_property_lookup(item)
            active_raw = ch_props.get("Active", True)
            active = not (active_raw is False or str(active_raw).strip().lower() in {"false", "0", "no"})
            rubric = _valid_electrode_rubrik(electrode_library, ch_props.get("Rubrik"))
            model = _model_for_rubrik(electrode_library, rubric, ch_props.get("ElectrodeModel") or ch_props.get("Model"))
            entry = {
                "Channel": str(column),
                "Position": str(position),
                "Active": active,
                "Rubrik": rubric,
                "Model": model,
            }
            impedance = coerce_number(item.get("impedance"))
            if impedance is not None:
                entry["Impedance"] = impedance
            # GND/Ref are AUXChannel nodes: physical electrodes with no data column. Counting
            # them as EEG is what inflated NumberEEGChannels on import.
            if _is_aux_channel_node(item, position):
                aux_electrodes.append(entry)
            else:
                channels.append(entry)
            continue
        name = jsonld_value(item) or f"Ch {idx + 1}"
        channels.append({"Channel": str(name), "Position": str(name), "Active": True})
    if not channels and n_eeg:
        channels = [{"Channel": f"Ch {idx + 1}", "Position": f"Ch {idx + 1}", "Active": True} for idx in range(int(n_eeg))]

    subject_ref = jsonld_value(recording.get("schema:object") or recording.get("object"))
    subject_code = ""
    if subject_ref:
        subject_node = id_map.get(str(subject_ref), {})
        subject_code = str(
            jsonld_value(subject_node.get("schema:identifier") or subject_node.get("identifier"))
            # A URN has no "/" to split on, so the old fallback yielded the literal
            # "urn:subject:01" and a later BIDS export wrote sub-urn:subject:01.
            or str(subject_ref).rsplit(":", 1)[-1]
        )
        subject_code = _strip_bids_prefixes(subject_code)

    raw_file = jsonld_value(file_node.get("schema:contentUrl") or file_node.get("contentUrl") or file_node.get("schema:name") or file_node.get("name"))
    method_options = list(method_names)
    fallback_method = default_method or ("Alpha" if "Alpha" in method_options else (method_options[0] if method_options else ""))
    method_raw = jsonld_value(recording.get("schema:measurementTechnique") or recording.get("measurementTechnique"))
    method = resolve_choice(method_options, method_raw) if method_raw else fallback_method
    parameters = {
        "fs": fs or 250,
        "RecordingTime": duration or 0,
        # len(channels) now excludes the AUX/GND/Ref nodes.
        "NumberEEGChannels": int(n_eeg or len(channels) or 0),
        # AUX *data columns* — not the GND/Ref electrodes, which occupy no column.
        "NumberAUXChannels": int(n_aux or 0),
        "Filename": _strip_bids_prefixes(Path(str(raw_file or file_name)).stem),
    }
    # Carry back every parameter the exporter wrote, StimFreq included. Reading only a
    # hand-picked subset is what let the stimulation frequency vanish on round-trip.
    for _param_key, prop_name, _unit in RECORDING_PROPERTIES:
        if prop_name in {"SamplingRate", "NumberEEGChannels", "NumberAUXChannels"}:
            continue  # handled above
        value = prop(prop_name)
        if value not in (None, "", []):
            parameters[_param_key] = value

    return {
        "Method": method,
        # Report the device that actually made the recording. Hardcoding "Offline" here also
        # disabled the Connect button, so an imported session could never be re-recorded.
        "Device": _device_from_graph(recording, id_map) or "Offline",
        "Parameters": parameters,
        "Channels": channels,
        "ReferenceElectrodes": aux_electrodes,
        "Metadata": {"Participant": {"Code": subject_code}} if subject_code else {},
        "DataFile": raw_file,
    }
