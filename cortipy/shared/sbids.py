"""SBIDS utilities integrated with CortiPy (export + load)."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from mne.io import BaseRaw

from .bids import BIDSLoadResult, BIDSLoader, ExperimentBinLoader, _coerce_to_raw_array


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
        content_url: str | None = None,
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
        content_url = content_url or f"raw_data/{os.path.basename(raw_file)}"
        file_node = {
            "@id": file_urn,
            "@type": ["schema:DigitalDocument", "prov:Entity"],
            "schema:name": os.path.basename(raw_file),
            "schema:encodingFormat": encoding,
            "schema:contentUrl": content_url,
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


def _build_node_map(doc: dict) -> dict[str, dict]:
    return {node.get("@id"): node for node in doc.get("@graph", []) if "@id" in node}


def _extract_dataset_info(doc: dict) -> tuple[str, str]:
    dataset_id = doc.get("@id", "").replace("urn:dataset:", "")
    dataset_name = None
    for node in doc.get("@graph", []):
        if node.get("@type") == "schema:Dataset":
            dataset_name = node.get("schema:name")
            break
    if not dataset_id or not dataset_name:
        raise ValueError("SBIDS JSON-LD missing dataset id or name.")
    return dataset_id, dataset_name


def _recording_ids_from_doc(doc: dict) -> list[str]:
    ids = []
    for node in doc.get("@graph", []):
        types = node.get("@type") or []
        if isinstance(types, str):
            types = [types]
        if "schema:CreateAction" in types or "schema:result" in node:
            rid = node.get("@id")
            if rid:
                ids.append(rid)
    return ids


class SBIDSLoader:
    """Load recordings described by an SBIDS JSON-LD document."""

    def __init__(self, root: str | Path, *, data_roots: Sequence[str | Path] | None = None) -> None:
        self.root = Path(root).expanduser()
        roots: list[Path] = []
        roots.append(self.root if self.root.is_dir() else self.root.parent)
        if data_roots:
            roots.extend(Path(p).expanduser() for p in data_roots)
        seen: list[Path] = []
        for entry in roots:
            if entry not in seen:
                seen.append(entry)
        self.data_roots = seen

    def read_sbids(
        self,
        meta_path: str | Path | None = None,
        *,
        recording_id: str | None = None,
        preload: bool = True,
    ) -> BIDSLoadResult:
        sbids_path = self._resolve_meta_path(meta_path)
        doc = json.loads(sbids_path.read_text(encoding="utf-8"))
        dataset_id, dataset_name = _extract_dataset_info(doc)
        recording = self._select_recording(doc, recording_id)
        file_node = self._find_file_node(doc, recording)
        data_path = self._resolve_data_path(file_node, dataset_id, dataset_name, sbids_path)
        channels = self._channels_from_recording(recording, data_path)
        sampling_rate = self._sampling_rate(recording, data_path)

        if data_path.suffix.lower() == ".bin" and (data_path.parent / "params.json").exists():
            bin_loader = ExperimentBinLoader(data_path.parent)
            result = bin_loader.read_bin(
                dataset=data_path.parent,
                data_path=data_path,
                params_path=data_path.parent / "params.json",
                sampling_rate=sampling_rate,
                channel_names=list(channels["name"]) if channels is not None else None,
                channel_types=list(channels["type"]) if channels is not None else None,
                channel_count=len(channels) if channels is not None else None,
            )
            metadata = dict(result.metadata or {})
            metadata["sbids"] = self._sbids_metadata(sbids_path, dataset_id, dataset_name, recording, file_node, sampling_rate)
            result.metadata = metadata
            return result

        raw, data = self._load_data(data_path, sampling_rate, channels, preload=preload)
        metadata = {
            "sbids": self._sbids_metadata(sbids_path, dataset_id, dataset_name, recording, file_node, sampling_rate)
        }
        return BIDSLoadResult(
            raw=raw,
            data=data,
            sampling_rate=float(raw.info.get("sfreq", 0.0)),
            events=None,
            channels=channels,
            metadata=metadata,
            source_path=data_path,
            ancillary_files=[],
        )

    def _resolve_meta_path(self, meta_path: str | Path | None) -> Path:
        if meta_path:
            path = Path(meta_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"SBIDS file not found: {path}")
            return path
        if self.root.is_file() and self.root.suffix.lower() == ".jsonld":
            return self.root
        candidates = sorted(self.root.glob("sbids_meta_*.jsonld"))
        if not candidates:
            candidates = sorted(self.root.glob("*.jsonld"))
        if not candidates:
            raise FileNotFoundError("No SBIDS JSON-LD files found.")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple SBIDS files found under {self.root}; pass `meta_path` to disambiguate."
            )
        return candidates[0]

    def _select_recording(self, doc: dict, recording_id: str | None) -> dict:
        candidates = []
        for node in doc.get("@graph", []):
            types = node.get("@type") or []
            if isinstance(types, str):
                types = [types]
            if "schema:CreateAction" in types or "schema:result" in node:
                candidates.append(node)
        if recording_id:
            for node in candidates:
                if node.get("@id") == recording_id:
                    return node
            raise ValueError(f"Recording id {recording_id} not found in SBIDS graph.")
        if not candidates:
            raise ValueError("SBIDS document does not contain any recording nodes.")
        return candidates[0]

    def _find_file_node(self, doc: dict, recording: dict) -> dict:
        result = recording.get("schema:result") or {}
        if isinstance(result, dict):
            target = result.get("@id")
        else:
            target = None
        node_map = _build_node_map(doc)
        if target and target in node_map:
            return node_map[target]
        raise ValueError("Could not locate file node for recording.")

    def _resolve_data_path(
        self, file_node: dict, dataset_id: str, dataset_name: str, sbids_path: Path
    ) -> Path:
        url = file_node.get("schema:contentUrl") or ""
        filename = file_node.get("schema:name") or Path(url).name
        candidates: list[Path] = []
        pieces = [p for p in (url, filename) if p]
        search_roots = list(self.data_roots)
        search_roots.append(sbids_path.parent)
        for base in list(search_roots):
            for sub in (dataset_id, dataset_name):
                if sub:
                    search_roots.append(base / sub)
                    search_roots.append(base.parent / sub if base.parent != base else base / sub)
            if base.name != "cortipy_runs":
                search_roots.append(base / "cortipy_runs")
                if base.parent != base:
                    search_roots.append(base.parent / "cortipy_runs")
        checked: list[Path] = []
        for root in search_roots:
            if root and root not in checked:
                checked.append(root)
        search_roots = checked
        for piece in pieces:
            rel = Path(piece.replace("dataset/", "")).with_name(Path(piece).name)
            if str(rel).startswith("raw_data/"):
                rel = rel.relative_to("raw_data")
            for base in search_roots:
                candidates.append(base / rel)
                candidates.append(base / "raw_data" / rel.name)
                candidates.append(base / rel.name)
                if dataset_name:
                    candidates.append(base / dataset_name / rel.name)
                if dataset_id and dataset_id != dataset_name:
                    candidates.append(base / dataset_id / rel.name)
        for candidate in candidates:
            if ".git" in candidate.parts:
                continue
            if candidate.exists():
                return candidate.resolve()
        # Fallback: scan for the filename under the search roots if direct joins failed.
        for base in search_roots:
            if ".git" in base.parts:
                continue
            try:
                match = next(p for p in base.rglob(filename) if ".git" not in p.parts)
                return match.resolve()
            except StopIteration:
                continue
        raise FileNotFoundError(f"Could not resolve raw data path for SBIDS entry {filename} ({url}).")

    def _channels_from_recording(self, recording: dict, data_path: Path) -> pd.DataFrame | None:
        variable = recording.get("schema:variableMeasured") or []
        if isinstance(variable, dict):
            variable = [variable]
        rows = []
        for idx, entry in enumerate(variable):
            name = entry.get("columnName") or entry.get("schema:name") or f"Ch{idx+1}"
            types = entry.get("@type") or []
            if isinstance(types, str):
                types = [types]
            ch_type = "EEG"
            if "AUXChannel" in types:
                ch_type = "MISC"
            elif "EEGChannel" in types:
                ch_type = "EEG"
            rows.append(
                {
                    "name": str(name),
                    "type": ch_type,
                    "impedance": entry.get("impedance"),
                    "active": self._extract_active_flag(entry),
                }
            )
        if not rows:
            return None
        frame = pd.DataFrame(rows)
        column_count = None
        try:
            column_count = np.load(data_path)["data"].shape[1] if data_path.suffix.lower() == ".npz" else None
        except Exception:
            column_count = None
        if column_count is None and data_path.exists() and data_path.suffix.lower() == ".npy":
            try:
                column_count = np.load(data_path).shape[1]
            except Exception:
                column_count = None
        if column_count is not None:
            if len(frame) < column_count:
                missing = column_count - len(frame)
                for idx in range(missing):
                    frame.loc[len(frame)] = {
                        "name": f"Ch{len(frame)+1}",
                        "type": "EEG",
                        "impedance": None,
                        "active": None,
                    }
            elif len(frame) > column_count:
                frame = frame.head(column_count)
        return frame.reset_index(drop=True)

    @staticmethod
    def _extract_active_flag(entry: dict) -> bool | None:
        additional = entry.get("schema:additionalProperty") or []
        if isinstance(additional, dict):
            additional = [additional]
        for prop in additional:
            if prop.get("schema:name") == "Active":
                return bool(prop.get("schema:value"))
        return None

    def _sampling_rate(self, recording: dict, data_path: Path) -> float | None:
        properties = recording.get("schema:additionalProperty") or []
        if isinstance(properties, dict):
            properties = [properties]
        lookup = {prop.get("schema:name"): prop.get("schema:value") for prop in properties}
        for key in ("SamplingRate", "SamplingFrequency", "SamplingFrequencyHz", "fs"):
            if key in lookup:
                try:
                    return float(lookup[key])
                except (TypeError, ValueError):
                    continue
        params_path = data_path.parent / "params.json"
        if params_path.exists():
            params = json.loads(params_path.read_text())
            try:
                return float(params.get("Parameters", {}).get("fs"))
            except (TypeError, ValueError):
                pass
        return None

    def _load_data(
        self,
        data_path: Path,
        sampling_rate: float | None,
        channels: pd.DataFrame | None,
        *,
        preload: bool,
    ) -> tuple[BaseRaw, np.ndarray]:
        suffix = data_path.suffix.lower()
        if suffix in {".npz", ".npy"}:
            if sampling_rate is None:
                raise ValueError("Sampling rate missing in SBIDS metadata; required to load NumPy data.")
            data = self._load_numpy_data(data_path)
            ch_names = list(channels["name"]) if channels is not None else None
            ch_types = [ct.lower() for ct in channels["type"]] if channels is not None and "type" in channels else None
            raw = _coerce_to_raw_array(data, sampling_rate, ch_names, ch_types)
            return raw, data

        sidecars = {}
        if sampling_rate is not None:
            sidecars["sbids_metadata.json"] = {"SamplingFrequency": sampling_rate}
        loader_metadata = {"sidecars": sidecars} if sidecars else {}
        loader = BIDSLoader(data_path.parent)
        raw = loader._load_raw(data_path, preload=preload, metadata=loader_metadata, channels=channels)
        return raw, raw.get_data().T

    @staticmethod
    def _load_numpy_data(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            data = np.load(path)
            return data if data.ndim == 2 else np.atleast_2d(data)
        with np.load(path) as npz:
            if "data" in npz:
                data = npz["data"]
            else:
                keys = list(npz.files)
                if not keys:
                    raise ValueError(f"No arrays stored in {path}")
                data = npz[keys[0]]
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D array in {path}, got shape {data.shape}.")
        return data

    @staticmethod
    def _sbids_metadata(
        sbids_path: Path,
        dataset_id: str,
        dataset_name: str,
        recording: dict,
        file_node: dict,
        sampling_rate: float | None,
    ) -> dict:
        payload = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "recording": recording,
            "file": file_node,
            "source": str(sbids_path),
        }
        if sampling_rate is not None:
            payload["sampling_rate"] = float(sampling_rate)
        return payload


def read_sbids(
    sbids_path: str | Path,
    *,
    data_roots: Sequence[str | Path] | None = None,
    recording_id: str | None = None,
    preload: bool = True,
    all_recordings: bool = False,
) -> BIDSLoadResult | list[BIDSLoadResult]:
    """Convenience loader akin to pandas.read_csv."""
    loader = SBIDSLoader(sbids_path, data_roots=data_roots)
    if not all_recordings:
        return loader.read_sbids(meta_path=sbids_path, recording_id=recording_id, preload=preload)

    doc = json.loads(Path(sbids_path).read_text(encoding="utf-8"))
    recording_ids = _recording_ids_from_doc(doc)
    results: list[BIDSLoadResult] = []
    for rid in recording_ids:
        results.append(loader.read_sbids(meta_path=sbids_path, recording_id=rid, preload=preload))
    return results


def to_sbids(
    bids_root: str | Path,
    *,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    allowed_ext: Sequence[str] | None = None,
    preload: bool = True,
    output: str | Path | None = None,
    all_recordings: bool = False,
    export_format: str = "parquet",
) -> Path:
    """Convert a BIDS dataset into SBIDS."""
    return convert_bids_to_sbids(
        bids_root=Path(bids_root),
        subject=subject,
        session=session,
        task=task,
        run=run,
        allowed_ext=list(allowed_ext) if allowed_ext is not None else None,
        preload=preload,
        output=Path(output) if output is not None else None,
        all_recordings=all_recordings,
        export_format=export_format,
    )


def _parse_bids_tokens(path: Path) -> dict[str, str | None]:
    tokens = path.stem.split("_")
    out: dict[str, str | None] = {"subject": None, "session": None, "task": None, "run": None}
    for token in tokens:
        if token.startswith("sub-"):
            out["subject"] = token.split("-", 1)[1]
        elif token.startswith("ses-"):
            out["session"] = token.split("-", 1)[1]
        elif token.startswith("task-"):
            out["task"] = token.split("-", 1)[1]
        elif token.startswith("run-"):
            out["run"] = token.split("-", 1)[1]
    return out


def _meta_from_bids_result(result: BIDSLoadResult, root: Path) -> dict:
    samples, channels = result.data.shape
    params = {
        "fs": result.sampling_rate,
        "RecordingTime": samples / float(result.sampling_rate) if result.sampling_rate else None,
        "NumberEEGChannels": channels,
        "NumberAUXChannels": 0,
        "Filename": Path(result.source_path).stem,
    }
    tokens = _parse_bids_tokens(Path(result.source_path))
    subject = tokens.get("subject")
    meta = {
        "Method": "BIDS import",
        "Device": "Unknown",
        "Parameters": params,
        "Channels": [{"Channel": name, "Position": name, "Active": True} for name in result.raw.ch_names],
        "Metadata": {"Participant": {"Code": subject}} if subject else {},
        "DataFile": str(Path(result.source_path).relative_to(root)),
    }
    return meta


def _ext_for_format(fmt: str) -> str:
    mapping = {
        "edf": "edf",
        "hdf5": "hdf5",
        "zarr": "zarr",
    }
    return mapping.get(fmt.lower(), fmt.lower())


def _dedupe_recordings(paths: Sequence[Path]) -> list[Path]:
    """Prefer BrainVision headers when multiple files share the same stem."""
    priority = {".vhdr": 0, ".vmrk": 1, ".eeg": 2}
    chosen: dict[tuple[Path, str], Path] = {}
    for rec in paths:
        key = (rec.parent, rec.stem)
        rank = priority.get(rec.suffix.lower(), 10)
        prev = chosen.get(key)
        if prev is None or priority.get(prev.suffix.lower(), 10) > rank:
            chosen[key] = rec
    return list(chosen.values())


def _export_recording_data(rec: BIDSLoadResult, raw_dir: Path, export_format: str) -> tuple[Path, str, int]:
    """Materialize the recording data into raw_data/ in the requested format."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(rec.source_path)
    stem = src_path.stem

    if export_format == "copy":
        dest_path = raw_dir / src_path.name
        if not dest_path.exists():
            dest_path.write_bytes(src_path.read_bytes())
        size = dest_path.stat().st_size
        return dest_path, str(Path("raw_data") / dest_path.name), size

    if export_format == "npz":
        dest_path = raw_dir / f"{stem}.npz"
        np.savez_compressed(dest_path, data=rec.data)
        size = dest_path.stat().st_size
        return dest_path, str(Path("raw_data") / dest_path.name), size

    if export_format in {"edf", "hdf5", "zarr"}:
        dest_path = raw_dir / f"{stem}.{_ext_for_format(export_format)}"
        helper = BIDSLoader(raw_dir)
        helper._write_raw(rec.raw, dest_path, format=export_format, overwrite=True)
        size = dest_path.stat().st_size
        return dest_path, str(Path("raw_data") / dest_path.name), size

    dest_path = raw_dir / f"{stem}.parquet"
    frame = pd.DataFrame(rec.data, columns=list(rec.raw.ch_names))
    frame.to_parquet(dest_path)
    size = dest_path.stat().st_size
    return dest_path, str(Path("raw_data") / dest_path.name), size


def convert_bids_to_sbids(
    bids_root: Path,
    *,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    allowed_ext: Sequence[str] | None = None,
    preload: bool = True,
    output: Path | None = None,
    all_recordings: bool = False,
    export_format: str = "parquet",
) -> Path:
    loader = BIDSLoader(bids_root)
    allowed = tuple(allowed_ext) if allowed_ext else (
        ".edf",
        ".bdf",
        ".vhdr",
        ".set",
        ".fif",
        ".eeg",
        ".parquet",
        ".h5",
        ".hdf5",
        ".zarr",
    )

    if all_recordings:
        rec_paths = _dedupe_recordings(
            loader._collect_recordings(
                allowed, ("eeg", "ieeg"), subject=subject, session=session, task=task, run=run
            )
        )
        if not rec_paths:
            raise SystemExit("No recordings found in the BIDS dataset for the given filters.")
        loaded = [loader._load_single_recording(rec, preload=preload) for rec in rec_paths]
    else:
        loaded = loader.read_bids_dataset(
            allowed_file_structures=allowed,
            subject=subject,
            session=session,
            task=task,
            run=run,
            preload=preload,
        )
    if not loaded:
        raise SystemExit("No recordings found in the BIDS dataset for the given filters.")

    dataset_name = bids_root.name
    dataset_id = dataset_name.replace(" ", "_").upper()
    exporter = SbidsExporter(dataset_id=dataset_id, dataset_name=dataset_name)
    sbids_root = output.parent if output else bids_root
    raw_dir = sbids_root / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for rec in loaded:
        meta = _meta_from_bids_result(rec, bids_root)
        dest_path, content_url, file_size = _export_recording_data(
            rec=rec,
            raw_dir=raw_dir,
            export_format=export_format,
        )
        exporter.add_recording_from_cortipy_json(
            meta_json=meta,
            raw_file=str(dest_path),
            subject_id=_parse_bids_tokens(Path(rec.source_path)).get("subject"),
            file_size_bytes=file_size,
            content_url=content_url,
        )

    output_path = output or bids_root / default_output_path(dataset_name, dataset_id)
    exporter.save(str(output_path))
    return output_path
