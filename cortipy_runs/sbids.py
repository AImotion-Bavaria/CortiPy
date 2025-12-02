"""SBIDS utilities for exporting CortiPy run metadata."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Tuple


class SbidsExporter:
    """Build an SBIDS JSON-LD document from CortiPy-style metadata JSONs."""

    BASE_CONTEXT = {
        "schema": "https://schema.org/",
        "prov": "http://www.w3.org/ns/prov#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "nemo": "http://purl.bioontology.org/ontology/NEMO/",
        "sbids": "urn:sbids:",
        "EEGChannel": "sbids:EEGChannel",
        "AUXChannel": "sbids:AUXChannel",
        "columnName": {"@id": "sbids:columnName", "@type": "xsd:string"},
        "impedance": {"@id": "sbids:impedance", "@type": "xsd:float"},
        "signalUnit": {"@id": "sbids:signalUnit", "@type": "xsd:string"},
        "impedanceUnit": {"@id": "sbids:impedanceUnit", "@type": "xsd:string"},
    }

    def __init__(self, dataset_id: str, dataset_name: str):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.doc = {
            "@context": deepcopy(self.BASE_CONTEXT),
            "@id": f"urn:dataset:{dataset_id}",
            "@graph": [
                {
                    "@id": f"urn:dataset:{dataset_id}",
                    "@type": "schema:Dataset",
                    "schema:name": dataset_name,
                }
            ],
        }
        self._graph = self.doc["@graph"]
        self._known_subjects = set()
        self._known_devices = set()

    def add_recording_from_cortipy_json(
        self,
        meta_json: dict,
        raw_file: str,
        subject_id: str | None = None,
        start_time_iso: str | None = None,
        end_time_iso: str | None = None,
        file_size_bytes: int | None = None,
    ) -> None:
        method = meta_json.get("Method")
        device = meta_json.get("Device")
        params = meta_json.get("Parameters", {})
        channels = meta_json.get("Channels", [])
        participant = meta_json.get("Metadata", {}).get("Participant", {})
        subj_id = (
            subject_id
            or participant.get("Code")
            or f"{device}_sub_{params.get('TestSubjectNo', '1')}"
        )
        self._add_subject(
            subj_id=subj_id,
            gender=participant.get("Gender"),
            dominant_hand=participant.get("DominantHand"),
        )
        self._add_device(name=device)
        fs = params.get("fs")
        rec_time = params.get("RecordingTime")
        env = params.get("Environment")
        n_eeg = params.get("NumberEEGChannels")
        n_aux = params.get("NumberAUXChannels")
        ref_ch = params.get("ReferenceChannel")
        trig_ch = params.get("TriggerChannel")
        lowest_f = params.get("LowestFrequency")
        highest_f = params.get("HighestFrequency")
        stimulus = params.get("Stimulus") or params.get("Start")
        filename_stem = params.get("Filename") or os.path.splitext(os.path.basename(raw_file))[0]
        recording_urn = f"urn:recording:{subj_id}_{filename_stem}"
        file_urn = f"urn:file:{subj_id}_{filename_stem}"
        recording_node = {
            "@id": recording_urn,
            "@type": ["schema:CreateAction", "prov:Activity"],
            "schema:name": f"EEG Recording Session ({method})" if method else "EEG Recording Session",
            "schema:instrument": {"@id": f"urn:device:{device}"},
            "schema:object": {"@id": f"urn:subject:{subj_id}"},
            "schema:result": {"@id": file_urn},
            "signalUnit": "microvolt",
            "impedanceUnit": "KOHM",
            "schema:additionalProperty": [],
        }
        if start_time_iso:
            recording_node["schema:startTime"] = start_time_iso
        if end_time_iso:
            recording_node["schema:endTime"] = end_time_iso
        if rec_time is not None:
            recording_node["schema:duration"] = f"PT{int(rec_time)}S"
        if method:
            recording_node["schema:measurementTechnique"] = method
        if env:
            recording_node["schema:additionalProperty"].append(
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "Environment",
                    "schema:value": env,
                }
            )
        if fs is not None:
            recording_node["schema:additionalProperty"].append(
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "SamplingRate",
                    "schema:value": fs,
                    "schema:unitCode": "HZ",
                }
            )
        if n_eeg is not None:
            recording_node["schema:additionalProperty"].append(
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "NumberEEGChannels",
                    "schema:value": n_eeg,
                }
            )
        if n_aux is not None:
            recording_node["schema:additionalProperty"].append(
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "NumberAUXChannels",
                    "schema:value": n_aux,
                }
            )
        if ref_ch is not None:
            recording_node["schema:additionalProperty"].append(
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "ReferenceChannel",
                    "schema:value": ref_ch,
                }
            )
        if trig_ch is not None:
            recording_node["schema:additionalProperty"].append(
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "TriggerChannel",
                    "schema:value": trig_ch,
                }
            )
        if lowest_f is not None:
            recording_node["schema:additionalProperty"].append(
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "LowestFrequency",
                    "schema:value": lowest_f,
                }
            )
        if highest_f is not None:
            recording_node["schema:additionalProperty"].append(
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "HighestFrequency",
                    "schema:value": highest_f,
                }
            )
        if stimulus:
            recording_node["schema:additionalProperty"].append(
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "Stimulus",
                    "schema:value": stimulus,
                }
            )
        variable_measured = []
        for idx, ch in enumerate(channels):
            label = ch.get("Channel")
            pos = ch.get("Position")
            imp = ch.get("Impedance")
            active = ch.get("Active", True)
            rubrik = ch.get("Rubrik")
            model = ch.get("Model")
            if (pos or "").upper() in ("GND", "REF"):
                ch_type = "AUXChannel"
            else:
                ch_type = "EEGChannel"
            chan_id = f"urn:channel:{subj_id}_{filename_stem}_{(label or pos or f'ch{idx+1}').replace(' ', '')}"
            chan_node = {
                "@id": chan_id,
                "@type": ["schema:PropertyValue", ch_type],
                "schema:name": pos or label or f"Ch{idx+1}",
                "columnName": label or pos or f"Ch{idx+1}",
            }
            if imp is not None:
                chan_node["impedance"] = imp
            if not active:
                chan_node.setdefault("schema:additionalProperty", []).append(
                    {
                        "@type": "schema:PropertyValue",
                        "schema:name": "Active",
                        "schema:value": False,
                    }
                )
            if rubrik:
                chan_node.setdefault("schema:additionalProperty", []).append(
                    {
                        "@type": "schema:PropertyValue",
                        "schema:name": "Rubrik",
                        "schema:value": rubrik,
                    }
                )
            if model:
                chan_node.setdefault("schema:additionalProperty", []).append(
                    {
                        "@type": "schema:PropertyValue",
                        "schema:name": "ElectrodeModel",
                        "schema:value": model,
                    }
                )
            variable_measured.append(chan_node)
        if variable_measured:
            recording_node["schema:variableMeasured"] = variable_measured
        self._graph.append(recording_node)
        if file_size_bytes is None and os.path.exists(raw_file):
            file_size_bytes = os.path.getsize(raw_file)
        encoding = self._guess_mime_from_extension(raw_file)
        file_node = {
            "@id": file_urn,
            "@type": ["schema:DigitalDocument", "prov:Entity"],
            "schema:name": os.path.basename(raw_file),
            "schema:encodingFormat": encoding,
            "schema:contentUrl": f"dataset/raw/{os.path.basename(raw_file)}",
            "prov:wasGeneratedBy": {"@id": recording_urn},
        }
        if file_size_bytes is not None:
            file_node["schema:fileSize"] = int(file_size_bytes)
        self._graph.append(file_node)

    def to_dict(self) -> dict:
        return deepcopy(self.doc)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.doc, indent=indent)

    def save(self, path: str, indent: int = 2) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    def _add_subject(self, subj_id: str, gender: str | None, dominant_hand: str | None) -> None:
        if subj_id in self._known_subjects:
            return
        node = {
            "@id": f"urn:subject:{subj_id}",
            "@type": "schema:Patient",
            "schema:identifier": str(subj_id),
        }
        if gender:
            node["schema:gender"] = gender
        if dominant_hand:
            node["schema:description"] = f"DominantHand: {dominant_hand}"
        self._graph.append(node)
        self._known_subjects.add(subj_id)

    def _add_device(self, name: str, manufacturer: str | None = None, model: str | None = None) -> None:
        if not name or name in self._known_devices:
            return
        node = {
            "@id": f"urn:device:{name}",
            "@type": ["schema:MedicalDevice", "prov:Agent"],
            "schema:name": name,
        }
        if manufacturer:
            node["schema:manufacturer"] = manufacturer
        if model:
            node["schema:model"] = model
        self._graph.append(node)
        self._known_devices.add(name)

    @staticmethod
    def _guess_mime_from_extension(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            return "application/vnd.apache.parquet"
        if ext == ".edf":
            return "application/x-edf"
        if ext == ".eeg":
            return "application/x-brainvision-eeg"
        if ext in (".h5", ".hdf5"):
            return "application/x-hdf5"
        if ext == ".zarr":
            return "application/x-zarr"
        return "application/octet-stream"


def iter_param_files(root: Path) -> Iterable[Path]:
    """Legacy helper: iterate params.json files under a root (kept for compatibility)."""
    for path in sorted(root.rglob("params.json")):
        if path.is_file():
            yield path


def load_params(dataset_dir: Path) -> dict:
    """Load params.json from a single dataset directory."""
    params_path = dataset_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"params.json not found in {dataset_dir}")
    return json.loads(params_path.read_text(encoding="utf-8"))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _isoformat(dt: datetime | None) -> str | None:
    if not dt:
        return None
    return dt.replace(microsecond=0).isoformat() + "Z"


def infer_times(meta: dict) -> Tuple[str | None, str | None]:
    start_dt = _parse_timestamp(meta.get("Timestamp"))
    params = meta.get("Parameters", {})
    rec_time = params.get("RecordingTime")
    try:
        rec_seconds = float(rec_time) if rec_time is not None else None
    except (TypeError, ValueError):
        rec_seconds = None
    end_dt = None
    if start_dt and rec_seconds:
        end_dt = start_dt + timedelta(seconds=rec_seconds)
    return _isoformat(start_dt), _isoformat(end_dt)


def resolve_raw_file(run_dir: Path, meta: dict) -> str:
    data_file = meta.get("DataFile")
    if data_file:
        candidate = Path(data_file)
        if not candidate.is_absolute():
            candidate = run_dir / data_file
        if candidate.exists():
            return str(candidate)
    filename = meta.get("Parameters", {}).get("Filename")
    if filename:
        candidate = run_dir / filename
        if candidate.exists():
            return str(candidate)
        matches = sorted(run_dir.glob(f"{filename}.*"))
        if matches:
            return str(matches[0])
    for file in sorted(run_dir.iterdir()):
        if file.is_file() and file.name != "params.json":
            return str(file)
    fallback = run_dir / (filename or data_file or "raw")
    return str(fallback)


def build_sbids_document_for_dataset(
    dataset_dir: Path,
    dataset_id: str,
    dataset_name: str,
) -> tuple[dict, int]:
    """Build SBIDS JSON-LD for a single dataset directory containing params.json."""
    meta = load_params(dataset_dir)
    exporter = SbidsExporter(dataset_id=dataset_id, dataset_name=dataset_name)
    raw_file = resolve_raw_file(dataset_dir, meta)
    start_time, end_time = infer_times(meta)
    exporter.add_recording_from_cortipy_json(
        meta_json=meta,
        raw_file=raw_file,
        start_time_iso=start_time,
        end_time_iso=end_time,
    )
    return exporter.to_dict(), 1


def export_dataset(
    dataset_dir: Path,
    dataset_id: str,
    dataset_name: str,
    output: Path,
    indent: int,
) -> int:
    doc, count = build_sbids_document_for_dataset(
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=indent), encoding="utf-8")
    return count


def default_output_path(dataset_name: str, dataset_id: str | None = None) -> Path:
    """Pick a deterministic output name: sbids_meta_<dataset>.jsonld."""
    stub = dataset_name or dataset_id or "dataset"
    safe_stub = stub.replace(" ", "_")
    return Path(f"sbids_meta_{safe_stub}.jsonld")
