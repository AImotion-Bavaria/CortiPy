"""Utilities for loading EEG data from BIDS datasets."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
import shutil

import mne
import numpy as np
import pandas as pd
from mne.io import BaseRaw


@dataclass
class BIDSLoadResult:
    """Structured representation of a loaded BIDS recording."""

    raw: BaseRaw
    data: np.ndarray
    sampling_rate: float
    events: Optional[pd.DataFrame]
    channels: Optional[pd.DataFrame]
    metadata: Dict[str, Any]
    source_path: Path
    ancillary_files: Sequence[tuple[Path, Path]] = field(default_factory=list)


def _coerce_to_raw_array(
    data: np.ndarray | BaseRaw,
    sampling_rate: Optional[float],
    ch_names: Optional[Sequence[str]],
    channel_types: str | Sequence[str] | None,
) -> BaseRaw:
    """Convert numpy data (samples x channels) into an MNE RawArray."""
    if isinstance(data, BaseRaw):
        return data

    if sampling_rate is None:
        raise ValueError("`sampling_rate` is required when providing numpy data.")

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError("`data` must be 2D with shape (samples, channels).")
    samples, channels = array.shape

    names = list(ch_names) if ch_names is not None else [f"Ch{i+1}" for i in range(channels)]
    if len(names) != channels:
        raise ValueError("`ch_names` length must match the number of channels.")

    ch_types: Sequence[str]
    if channel_types is None:
        ch_types = ["eeg"] * channels
    elif isinstance(channel_types, str):
        ch_types = [channel_types] * channels
    else:
        ch_types = list(channel_types)

    if len(ch_types) != channels:
        raise ValueError("`channel_types` length must match the number of channels.")

    info = mne.create_info(ch_names=names, sfreq=float(sampling_rate), ch_types=ch_types)
    return mne.io.RawArray(array.T, info)


class BIDSLoader:
    """Load EEG recordings stored in a BIDS directory layout.

    Only a handful of common EEG file types are supported out of the box. Pass
    `allowed_file_structures` to widen or restrict which files are considered when
    searching through the dataset.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError(f"BIDS root must be a directory: {self.root}")

    def read_bids(
        self,
        allowed_file_structures: Sequence[str] = (
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
        ),
        *,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
            run: Optional[str] = None,
            modality_dirs: Sequence[str] = ("eeg", "ieeg"),
            preload: bool = True,
        ) -> BIDSLoadResult:
        """Load the first BIDS recording that matches the provided filters."""
        allowed_ext = tuple(self._normalize_extension(ext) for ext in allowed_file_structures)
        recording = self._select_recording(
            allowed_ext, modality_dirs, subject=subject, session=session, task=task, run=run
        )
        return self._load_single_recording(recording, preload=preload)

    def read_bids_dataset(
        self,
        allowed_file_structures: Sequence[str] = (
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
        ),
        *,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
        run: Optional[str] = None,
        modality_dirs: Sequence[str] = ("eeg", "ieeg"),
        preload: bool = True,
    ) -> list[BIDSLoadResult]:
        """Load all recordings in the dataset that match the provided filters."""
        allowed_ext = tuple(self._normalize_extension(ext) for ext in allowed_file_structures)
        recordings = self._collect_recordings(
            allowed_ext, modality_dirs, subject=subject, session=session, task=task, run=run
        )
        return [self._load_single_recording(path, preload=preload) for path in recordings]

    def to_bids(
        self,
        data: np.ndarray | BaseRaw,
        sampling_rate: Optional[float] = None,
        *,
        ch_names: Optional[Sequence[str]] = None,
        channel_types: str | Sequence[str] | None = None,
        subject: str,
        session: Optional[str] = None,
        task: Optional[str] = None,
        run: Optional[str] = None,
        modality: str = "eeg",
        format: str = "fif",
        events: Optional[pd.DataFrame] = None,
        channels: Optional[pd.DataFrame] = None,
        sidecar: Optional[Dict[str, Any]] = None,
        dataset_description: Optional[Dict[str, Any]] = None,
        ancillary_files: Optional[Sequence[tuple[Path, Path]]] = None,
        overwrite: bool = False,
    ) -> Path:
        """Write data into a minimal BIDS layout and return the main data path."""
        raw = self._coerce_to_raw(data, sampling_rate, ch_names, channel_types)
        modality_dir = self._ensure_modality_dir(subject, session, modality)
        base_stem = self._compose_stem(subject, session, task, run)
        data_stem = f"{base_stem}_{modality}"
        extension = self._extension_for_format(format)
        data_path = modality_dir / f"{data_stem}.{extension}"

        if data_path.exists() and not overwrite:
            raise FileExistsError(
                f"Destination already exists: {data_path}. Pass overwrite=True to replace it."
            )

        self._write_raw(raw, data_path, format=format.lower(), overwrite=overwrite)

        if events is not None:
            self._write_tsv(events, modality_dir / f"{base_stem}_events.tsv", overwrite=overwrite)
        if channels is not None:
            self._write_tsv(channels, modality_dir / f"{base_stem}_channels.tsv", overwrite=overwrite)
        sidecar_payload = dict(sidecar) if sidecar is not None else {}
        sidecar_payload.setdefault("SamplingFrequency", raw.info["sfreq"])
        self._write_json(sidecar_payload, modality_dir / f"{data_stem}.json", overwrite=overwrite)
        if dataset_description is not None:
            self._write_json(dataset_description, self.root / "dataset_description.json", overwrite=overwrite)
        if ancillary_files:
            for src, rel in ancillary_files:
                dest = self.root / rel
                if dest.exists() and not overwrite:
                    raise FileExistsError(
                        f"Destination already exists: {dest}. Pass overwrite=True to replace it."
                    )
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

        return data_path

    def _select_recording(
        self,
        allowed_ext: Sequence[str],
        modality_dirs: Sequence[str],
        *,
        subject: Optional[str],
        session: Optional[str],
        task: Optional[str],
        run: Optional[str],
    ) -> Path:
        recordings, broken_symlinks = self._discover_recordings(allowed_ext, modality_dirs)
        filtered = self._filter_recordings(
            recordings, subject=subject, session=session, task=task, run=run
        )
        if filtered:
            return filtered[0]

        if broken_symlinks:
            broken_display = ", ".join(p.name for p in broken_symlinks[:5])
            raise FileNotFoundError(
                "Found BIDS EEG file symlinks whose targets are missing. "
                "Fetch dataset contents (e.g., `datalad get` / `git annex get`) and retry. "
                f"Examples: {broken_display}"
            )

        raise FileNotFoundError(
            "No BIDS recordings found matching the provided filters. "
            f"Allowed extensions: {', '.join(allowed_ext)}."
        )

    def _collect_recordings(
        self,
        allowed_ext: Sequence[str],
        modality_dirs: Sequence[str],
        *,
        subject: Optional[str],
        session: Optional[str],
        task: Optional[str],
        run: Optional[str],
    ) -> list[Path]:
        recordings, broken_symlinks = self._discover_recordings(allowed_ext, modality_dirs)
        filtered = self._filter_recordings(
            recordings, subject=subject, session=session, task=task, run=run
        )
        if filtered:
            return filtered
        if broken_symlinks:
            broken_display = ", ".join(p.name for p in broken_symlinks[:5])
            raise FileNotFoundError(
                "Found BIDS EEG file symlinks whose targets are missing. "
                "Fetch dataset contents (e.g., `datalad get` / `git annex get`) and retry. "
                f"Examples: {broken_display}"
            )
        raise FileNotFoundError(
            "No BIDS recordings found matching the provided filters. "
            f"Allowed extensions: {', '.join(allowed_ext)}."
        )

    def _discover_recordings(
        self, allowed_ext: Sequence[str], modality_dirs: Sequence[str]
    ) -> tuple[list[Path], list[Path]]:
        candidates: list[Path] = []
        for ext in allowed_ext:
            for modality in modality_dirs:
                candidates.extend(self.root.rglob(f"*/{modality}/*{ext}"))

        seen_resolved = set()
        recordings: list[Path] = []
        broken_symlinks: list[Path] = []
        for path in candidates:
            resolved = path.resolve()
            if path.is_symlink() and not path.exists():
                broken_symlinks.append(path)
                continue
            if not resolved.exists():
                continue
            if ext == ".zarr":
                if not (resolved.is_file() or resolved.is_dir()):
                    continue
            else:
                if not resolved.is_file():
                    continue
            if "derivatives" in resolved.parts:
                continue
            if resolved in seen_resolved:
                continue
            seen_resolved.add(resolved)
            recordings.append(path)
        recordings.sort()
        return recordings, broken_symlinks

    def _filter_recordings(
        self,
        recordings: Sequence[Path],
        *,
        subject: Optional[str],
        session: Optional[str],
        task: Optional[str],
        run: Optional[str],
    ) -> list[Path]:
        return [
            path
            for path in recordings
            if self._matches_filters(path, subject=subject, session=session, task=task, run=run)
        ]

    def _load_single_recording(self, recording: Path, *, preload: bool) -> BIDSLoadResult:
        metadata = self._load_metadata(recording)
        events = self._load_table(recording, "_events.tsv")
        channels = self._load_table(recording, "_channels.tsv")
        raw = self._load_raw(recording, preload=preload, metadata=metadata, channels=channels)
        data = raw.get_data().T
        ancillary_files = self._collect_ancillary(recording)

        return BIDSLoadResult(
            raw=raw,
            data=data,
            sampling_rate=float(raw.info.get("sfreq", 0.0)),
            events=events,
            channels=channels,
            metadata=metadata,
            source_path=recording,
            ancillary_files=ancillary_files,
        )

    def _matches_filters(
        self,
        path: Path,
        *,
        subject: Optional[str],
        session: Optional[str],
        task: Optional[str],
        run: Optional[str],
    ) -> bool:
        tokens = set(path.stem.split("_"))
        return all(
            [
                self._token_present(tokens, "sub", subject),
                self._token_present(tokens, "ses", session),
                self._token_present(tokens, "task", task),
                self._token_present(tokens, "run", run),
            ]
        )

    def _token_present(self, tokens: Iterable[str], prefix: str, value: Optional[str]) -> bool:
        if value is None:
            return True
        return f"{prefix}-{value}" in tokens

    def _load_raw(
        self,
        path: Path,
        *,
        preload: bool,
        metadata: Dict[str, Any],
        channels: Optional[pd.DataFrame],
    ) -> BaseRaw:
        suffix = path.suffix.lower()
        if suffix == ".edf":
            reader = mne.io.read_raw_edf
        elif suffix == ".bdf":
            reader = mne.io.read_raw_bdf
        elif suffix == ".vhdr":
            reader = mne.io.read_raw_brainvision
        elif suffix == ".set":
            reader = mne.io.read_raw_eeglab
        elif suffix == ".fif":
            reader = mne.io.read_raw_fif
        elif suffix == ".eeg":
            vhdr = path.with_suffix(".vhdr")
            if vhdr.exists():
                return mne.io.read_raw_brainvision(vhdr, preload=preload)
            raise RuntimeError(
                f"BrainVision header (.vhdr) not found for {path.name}; cannot load .eeg without it."
            )
        elif suffix in {".parquet", ".h5", ".hdf5", ".zarr"}:
            return self._load_tabular_raw(path, metadata=metadata, channels=channels)
        else:
            raise RuntimeError(f"Unsupported EEG file type: {suffix}")
        return reader(path, preload=preload)

    def _load_table(self, data_path: Path, suffix: str) -> Optional[pd.DataFrame]:
        stems = {data_path.stem}
        if data_path.stem.endswith("_eeg") or data_path.stem.endswith("_ieeg"):
            stems.add(data_path.stem.rsplit("_", 1)[0])

        for stem in stems:
            candidate = data_path.with_name(f"{stem}{suffix}")
            if candidate.exists():
                return pd.read_csv(candidate, sep="\t")
        return None

    def _load_metadata(self, data_path: Path) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        description_path = self.root / "dataset_description.json"
        if description_path.exists():
            metadata["dataset_description"] = json.loads(description_path.read_text())

        sidecars: Dict[str, Any] = {}
        stems = {data_path.stem}
        if data_path.stem.endswith("_eeg") or data_path.stem.endswith("_ieeg"):
            stems.add(data_path.stem.rsplit("_", 1)[0])
        for stem in stems:
            for suffix in (".json", "_coordsystem.json"):
                sidecar = data_path.with_name(f"{stem}{suffix}")
                if sidecar.exists():
                    sidecars[sidecar.name] = json.loads(sidecar.read_text())
        if sidecars:
            metadata["sidecars"] = sidecars

        return metadata

    def _collect_ancillary(self, data_path: Path) -> list[tuple[Path, Path]]:
        ancillary: list[tuple[Path, Path]] = []

        def add(path: Path) -> None:
            if path.exists():
                ancillary.append((path, path.relative_to(self.root)))

        data_dir = data_path.parent
        session_dir = data_dir.parent

        for pattern in ("*electrodes.tsv", "*coordsystem*.json", "*events.json"):
            for candidate in data_dir.glob(pattern):
                add(candidate)
        for candidate in session_dir.glob("*scans.tsv"):
            add(candidate)
        for name in ("participants.tsv", "participants.json"):
            add(self.root / name)

        return ancillary

    def _normalize_extension(self, value: str) -> str:
        value = value.lower()
        return value if value.startswith(".") else f".{value}"

    def _load_tabular_raw(
        self, path: Path, *, metadata: Dict[str, Any], channels: Optional[pd.DataFrame]
    ) -> BaseRaw:
        array, column_names, sampling_rate = self._load_tabular_array(path, metadata)

        ch_names = column_names
        if channels is not None and "name" in channels:
            ch_names = list(channels["name"])

        ch_types: Sequence[str]
        if channels is not None and "type" in channels:
            ch_types = [ct.lower() for ct in channels["type"]]
        else:
            ch_types = ["eeg"] * len(ch_names)

        info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
        return mne.io.RawArray(array.T, info)

    def _load_tabular_array(self, path: Path, metadata: Dict[str, Any]) -> tuple[np.ndarray, list[str], float]:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            try:
                frame = pd.read_parquet(path)
            except ImportError as exc:
                raise RuntimeError("Reading parquet requires the 'pyarrow' or 'fastparquet' dependency.") from exc
            data = frame.to_numpy()
            columns = list(frame.columns)
        elif suffix in {".h5", ".hdf5"}:
            data, columns = self._read_hdf5(path)
        elif suffix == ".zarr":
            try:
                import zarr
            except ImportError as exc:
                raise RuntimeError("Reading zarr requires the optional dependency 'zarr'.") from exc
            store = zarr.open(str(path), mode="r")
            dataset = store["data"] if "data" in store else store
            data = np.asarray(dataset)
            if hasattr(dataset, "attrs"):
                columns = list(dataset.attrs.get("ch_names", []))
            else:
                columns = []
        else:
            raise RuntimeError(f"Unsupported tabular file type: {suffix}")

        sampling_rate = self._infer_sampling_rate(metadata)
        if sampling_rate is None:
            if suffix in {".h5", ".hdf5"}:
                sampling_rate = self._read_sampling_rate_from_hdf5(path)
            if sampling_rate is None:
                raise ValueError(
                    f"SamplingFrequency missing for {path.name}. Provide it via the sidecar JSON."
                )

        if not columns:
            columns = [f"Ch{i+1}" for i in range(data.shape[1])]

        return np.asarray(data), columns, float(sampling_rate)

    def _compose_stem(
        self, subject: str, session: Optional[str], task: Optional[str], run: Optional[str]
    ) -> str:
        if not subject:
            raise ValueError("`subject` is required to construct BIDS file names.")
        pieces = [f"sub-{subject}"]
        if session:
            pieces.append(f"ses-{session}")
        if task:
            pieces.append(f"task-{task}")
        if run:
            pieces.append(f"run-{run}")
        return "_".join(pieces)

    def _extension_for_format(self, format: str) -> str:
        fmt = format.lower()
        mapping = {
            "fif": "fif",
            "edf": "edf",
            "bdf": "bdf",
            "brainvision": "vhdr",
            "eeglab": "set",
            "parquet": "parquet",
            "hdf5": "hdf5",
            "h5": "h5",
            "zarr": "zarr",
        }
        if fmt not in mapping:
            raise ValueError(
                f"Unsupported export format '{format}'. "
                "Choose from fif, edf, bdf, brainvision, eeglab, parquet, hdf5, h5, zarr."
            )
        return mapping[fmt]

    def _ensure_modality_dir(self, subject: str, session: Optional[str], modality: str) -> Path:
        parts = [self.root, f"sub-{subject}"]
        if session:
            parts.append(f"ses-{session}")
        parts.append(modality.lower())
        target = Path(*parts)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _coerce_to_raw(
        self,
        data: np.ndarray | BaseRaw,
        sampling_rate: Optional[float],
        ch_names: Optional[Sequence[str]],
        channel_types: str | Sequence[str] | None,
    ) -> BaseRaw:
        return _coerce_to_raw_array(data, sampling_rate, ch_names, channel_types)

    def _write_raw(self, raw: BaseRaw, path: Path, *, format: str, overwrite: bool) -> None:
        fmt = format.lower()
        if fmt in {"edf", "bdf"}:
            self._write_edf_bdf(raw, path, fmt=fmt, overwrite=overwrite)
            return
        if fmt == "fif":
            raw.save(str(path), overwrite=overwrite)
            return
        if fmt in {"parquet", "hdf5", "h5", "zarr"}:
            self._write_tabular(raw, path, fmt=fmt, overwrite=overwrite)
            return
        try:
            mne.export.export_raw(str(path), raw, fmt=fmt, overwrite=overwrite)
        except RuntimeError as exc:
            missing_dep = None
            message = str(exc)
            if "EDFlib" in message:
                missing_dep = "EDFlib-Python"
            elif "pybv" in message:
                missing_dep = "pybv"
            elif "eeglabio" in message:
                missing_dep = "eeglabio"
            if missing_dep:
                raise RuntimeError(
                    f"Exporting format '{fmt}' requires the optional dependency '{missing_dep}'. "
                    "Install it or choose a different format."
                ) from exc
            raise

    def _write_edf_bdf(self, raw: BaseRaw, path: Path, *, fmt: str, overwrite: bool) -> None:
        try:
            import pyedflib
        except ImportError as exc:
            raise RuntimeError(
                f"Exporting format '{fmt}' requires the optional dependency 'pyedflib'. "
                "Install it or choose a different format."
            ) from exc

        if path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {path}. Pass overwrite=True to replace it.")

        data = raw.get_data()  # shape (channels, samples)
        signal_headers = []

        def _edf_friendly_bound(value: float, *, is_min: bool) -> float:
            if not np.isfinite(value):
                return 0.0
            for decimals in (6, 5, 4, 3, 2, 1, 0):
                factor = 10**decimals
                if is_min:
                    rounded = float(np.floor(value * factor) / factor)
                else:
                    rounded = float(np.ceil(value * factor) / factor)
                if len(str(rounded)) <= 8:
                    return rounded
            magnitude = min(abs(value), 9_999_999)
            return -magnitude if value < 0 else magnitude

        for idx, ch_name in enumerate(raw.ch_names):
            channel_data = data[idx]
            vmin = float(channel_data.min()) if channel_data.size else -1.0
            vmax = float(channel_data.max()) if channel_data.size else 1.0
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
            # Expand bounds slightly to avoid pyedflib warnings when values hit the edge.
            pad = max(1e-6, abs(vmax) * 1e-4, abs(vmin) * 1e-4)
            vmin = _edf_friendly_bound(vmin - pad, is_min=True)
            vmax = _edf_friendly_bound(vmax + pad, is_min=False)
            signal_headers.append(
                pyedflib.highlevel.make_signal_header(
                    ch_name,
                    sample_frequency=float(raw.info["sfreq"]),
                    physical_min=vmin,
                    physical_max=vmax,
                )
            )

        file_type = (
            pyedflib.FILETYPE_BDFPLUS if fmt == "bdf" else pyedflib.FILETYPE_EDFPLUS
        )
        pyedflib.highlevel.write_edf(
            str(path),
            data,
            signal_headers=signal_headers,
            file_type=file_type,
        )

    def _write_tsv(self, frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {path}. Pass overwrite=True to replace it.")
        frame.to_csv(path, sep="\t", index=False)

    def _write_json(self, payload: Dict[str, Any], path: Path, overwrite: bool) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {path}. Pass overwrite=True to replace it.")
        path.write_text(json.dumps(payload, indent=2))

    def _write_tabular(self, raw: BaseRaw, path: Path, *, fmt: str, overwrite: bool) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {path}. Pass overwrite=True to replace it.")

        data = raw.get_data().T  # samples x channels
        ch_names = raw.ch_names
        if fmt == "parquet":
            frame = pd.DataFrame(data, columns=ch_names)
            try:
                frame.to_parquet(path)
            except ImportError as exc:
                raise RuntimeError("Exporting to parquet requires the optional dependency 'pyarrow' or 'fastparquet'.") from exc
            return

        if fmt in {"hdf5", "h5"}:
            try:
                import h5py
            except ImportError as exc:
                raise RuntimeError("Exporting to HDF5 requires the optional dependency 'h5py'.") from exc
            with h5py.File(path, "w") as h5file:
                dset = h5file.create_dataset("data", data=data, compression="gzip")
                dset.attrs["ch_names"] = ch_names
                dset.attrs["sfreq"] = float(raw.info["sfreq"])
            return

        if fmt == "zarr":
            try:
                import zarr
            except ImportError as exc:
                raise RuntimeError("Exporting to zarr requires the optional dependency 'zarr'.") from exc
            store = zarr.open_group(str(path), mode="w")
            chunk_shape = (min(4096, data.shape[0]), min(data.shape[1], 64))
            dataset = store.create_dataset(
                "data",
                data=data,
                shape=data.shape,
                chunks=chunk_shape,
                compressor=None,
            )
            if hasattr(dataset, "attrs"):
                dataset.attrs["ch_names"] = ch_names
                dataset.attrs["sfreq"] = float(raw.info["sfreq"])
            return

        raise RuntimeError(f"Unsupported tabular export format: {fmt}")

    def _read_hdf5(self, path: Path) -> tuple[np.ndarray, list[str]]:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Reading HDF5 requires the optional dependency 'h5py'.") from exc

        with h5py.File(path, "r") as h5file:
            dataset = h5file["data"]
            data = np.asarray(dataset)
            ch_names = list(dataset.attrs.get("ch_names", []))
        return data, ch_names

    def _read_sampling_rate_from_hdf5(self, path: Path) -> Optional[float]:
        try:
            import h5py
        except ImportError:
            return None
        with h5py.File(path, "r") as h5file:
            dataset = h5file.get("data")
            if dataset is None:
                return None
            sfreq = dataset.attrs.get("sfreq")
            return float(sfreq) if sfreq is not None else None

    def _infer_sampling_rate(self, metadata: Dict[str, Any]) -> Optional[float]:
        sidecars = metadata.get("sidecars", {})
        if not sidecars:
            return None
        for payload in sidecars.values():
            for key in ("SamplingFrequency", "SamplingRate", "SamplingFrequencyHz", "sfreq", "sampling_rate"):
                if key in payload:
                    try:
                        return float(payload[key])
                    except (TypeError, ValueError):
                        continue
        return None


class ExperimentBinLoader:
    """Load/save CortiPy experiment binaries (params.json + *.bin)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def read_bin(
        self,
        dataset: str | Path,
        *,
        dtype: str | np.dtype = np.float64,
        sampling_rate: Optional[float] = None,
        channel_names: Optional[Sequence[str]] = None,
        channel_types: str | Sequence[str] | None = None,
        params_path: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        channel_count: Optional[int] = None,
    ) -> BIDSLoadResult:
        """Read a CortiPy experiment binary into a `BIDSLoadResult`.

        The expected layout is a folder containing `params.json` plus a `.bin`
        matrix with shape (samples x channels). This is the structure used in
        `experiments/datasets/D4_sereega_evoked_potentials/*_scalpdata.bin` (and similar folders).
        """
        dataset_dir = self._resolve_dataset(dataset)
        params_file = Path(params_path).expanduser() if params_path else dataset_dir / "params.json"
        if not params_file.exists():
            candidates = sorted(dataset_dir.glob("*.json"))
            if candidates:
                params_file = candidates[0]
        if not params_file.exists():
            raise FileNotFoundError(f"params.json not found at {params_file}")

        params = json.loads(params_file.read_text())
        data_file = Path(data_path).expanduser() if data_path else self._guess_data_path(dataset_dir, params)
        if not data_file.exists():
            raise FileNotFoundError(f"Binary data file not found at {data_file}")

        resolved_dtype = np.dtype(dtype)
        flat = np.fromfile(data_file, dtype=resolved_dtype)

        n_channels = self._infer_channel_count(params, channel_names, channel_count, data_len=flat.size)
        if n_channels is None or n_channels <= 0:
            raise ValueError("Channel count could not be inferred; pass `channel_count` or `channel_names`.")
        if flat.size % n_channels != 0:
            raise ValueError(
                f"Data length ({flat.size}) is not divisible by channel count ({n_channels}). "
                "Check the dtype or override `channel_count`."
            )

        samples = flat.size // n_channels
        data = flat.reshape((samples, n_channels))

        param_block = params.get("Parameters", {}) if isinstance(params, dict) else {}
        epoch_count = param_block.get("Epochs") or params.get("Epochs") if isinstance(params, dict) else None
        epoch_len_ms = param_block.get("EpochLength") or params.get("EpochLength") if isinstance(params, dict) else None

        # Detect MATLAB-exported 3-D arrays (channels x samples x epochs) saved in column-major order.
        sfreq = sampling_rate or self._resolve_sampling_rate(params, samples)
        epoch_events = None
        if sfreq and epoch_count and epoch_len_ms:
            try:
                epoch_count_int = int(epoch_count)
                epoch_len_ms_f = float(epoch_len_ms)
                samples_per_epoch = int(round(epoch_len_ms_f / 1000.0 * float(sfreq)))
            except Exception:
                samples_per_epoch = 0
                epoch_count_int = 0
            expected = n_channels * samples_per_epoch * epoch_count_int
            if samples_per_epoch > 0 and epoch_count_int > 0 and expected == flat.size:
                # Reorder from (channels, samples, epochs) Fortran order -> (samples_total, channels) C order.
                data_epochs = flat.reshape((n_channels, samples_per_epoch, epoch_count_int), order="F")
                data = data_epochs.transpose(2, 1, 0).reshape(samples_per_epoch * epoch_count_int, n_channels)
                samples = data.shape[0]
                epoch_events = pd.DataFrame(
                    {
                        "onset_sample": np.arange(epoch_count_int) * samples_per_epoch,
                        "onset_time": (np.arange(epoch_count_int) * samples_per_epoch) / float(sfreq),
                        "trial_type": ["epoch"] * epoch_count_int,
                    }
                )

        names = self._resolve_channel_names(params, channel_names, n_channels)
        sfreq = sampling_rate or self._resolve_sampling_rate(params, samples)
        if sfreq is None:
            raise ValueError(
                "Sampling frequency could not be inferred. "
                "Pass `sampling_rate` or add `fs`/`RecordingTime` to params."
            )
        channels_df = self._channels_frame(params, n_channels)

        metadata: Dict[str, Any] = {"params": params, "source": "experiment_bin"}
        recording_time = params.get("Parameters", {}).get("RecordingTime")
        if recording_time not in (None, 0):
            try:
                metadata["sampling_rate_inferred_from_duration"] = samples / float(recording_time)
            except (TypeError, ValueError):
                pass
        if epoch_events is not None:
            metadata["epochs"] = {
                "count": int(epoch_events.shape[0]),
                "samples_per_epoch": int(epoch_events["onset_sample"].diff().dropna().iloc[0])
                if epoch_events.shape[0] > 1
                else samples,
                "duration_s": float(samples) / float(sfreq) / max(int(epoch_events.shape[0]), 1),
            }
            metadata["data_layout"] = "channels x samples x epochs (Fortran) reshaped to continuous trials"

        raw = _coerce_to_raw_array(data, sfreq, names, channel_types)
        return BIDSLoadResult(
            raw=raw,
            data=data,
            sampling_rate=float(sfreq) if sfreq is not None else 0.0,
            events=epoch_events,
            channels=channels_df,
            metadata=metadata,
            source_path=data_file,
            ancillary_files=[],
        )

    def write_bin(
        self,
        data: np.ndarray | BaseRaw,
        target_dir: str | Path,
        *,
        params: Optional[Dict[str, Any]] = None,
        sampling_rate: Optional[float] = None,
        ch_names: Optional[Sequence[str]] = None,
        channel_types: str | Sequence[str] | None = None,
        dtype: str | np.dtype = np.float64,
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> tuple[Path, Path]:
        """Export Raw/numpy data into the CortiPy `.bin` + `params.json` format."""
        target = Path(target_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)

        raw = _coerce_to_raw_array(data, sampling_rate, ch_names, channel_types)
        samples = raw.n_times
        channels = len(raw.ch_names)

        resolved_dtype = np.dtype(dtype)
        data_path = target / self._export_filename(filename, params, target)
        if data_path.exists() and not overwrite:
            raise FileExistsError(f"{data_path} already exists; pass overwrite=True to replace it.")

        raw.get_data().T.astype(resolved_dtype, copy=False).tofile(data_path)

        params_payload = self._prepare_params_for_export(params, raw, data_path, samples, channels)
        params_file = target / "params.json"
        if params_file.exists() and not overwrite:
            raise FileExistsError(f"{params_file} already exists; pass overwrite=True to replace it.")
        params_file.write_text(json.dumps(params_payload, indent=2))

        return data_path, params_file

    def _resolve_dataset(self, dataset: str | Path) -> Path:
        base = Path(dataset).expanduser()
        if not base.is_absolute():
            base = (self.root / base).resolve()
        if base.is_file():
            return base.parent
        return base

    def _guess_data_path(self, dataset_dir: Path, params: Dict[str, Any]) -> Path:
        params_block = params.get("Parameters", {})
        for candidate in (
            params.get("OutputFile"),
            params_block.get("OutputFile"),
            params_block.get("Filename"),
        ):
            if candidate:
                path = dataset_dir / f"{candidate}"
                if path.suffix.lower() != ".bin":
                    path = path.with_suffix(".bin")
                if path.exists():
                    return path

        bin_files = list(dataset_dir.glob("*.bin"))
        if len(bin_files) == 1:
            return bin_files[0]

        raise FileNotFoundError(
            f"No .bin file found in {dataset_dir}. "
            "Pass `data_path` explicitly or add `OutputFile`/`Filename` to params."
        )

    def _infer_channel_count(
        self,
        params: Dict[str, Any],
        channel_names: Optional[Sequence[str]],
        channel_count: Optional[int],
        *,
        data_len: int | None = None,
    ) -> Optional[int]:
        if channel_count is not None:
            return int(channel_count)

        params_block = params.get("Parameters", {})
        eeg = params_block.get("NumberEEGChannels")
        aux = params_block.get("NumberAUXChannels", 0)
        if eeg is not None:
            try:
                return int(eeg) + int(aux or 0)
            except (TypeError, ValueError):
                pass

        for key in ("channel_count", "ChannelCount", "channels", "ChannelsTotal"):
            if key in params:
                try:
                    return int(params[key]) if not isinstance(params[key], list) else len(params[key])
                except (TypeError, ValueError):
                    continue

        if channel_names is not None:
            return len(channel_names)

        channels_meta = params.get("Channels")
        if channels_meta:
            return len(channels_meta)

        if data_len:
            for guess in (64, 32, 16, 8, 4, 2):
                if data_len % guess == 0:
                    return guess
        return None

    def _resolve_channel_names(
        self, params: Dict[str, Any], channel_names: Optional[Sequence[str]], expected: int
    ) -> list[str]:
        if channel_names is not None:
            names = list(channel_names)
        else:
            names = []
            for entry in params.get("Channels", []):
                for key in ("Channel", "Position", "Name"):
                    candidate = entry.get(key)
                    if candidate:
                        names.append(str(candidate))
                        break
            if not names and isinstance(params.get("channels"), list):
                names = [str(x) for x in params["channels"]]

        if not names:
            names = [f"Ch{i+1}" for i in range(expected)]

        if len(names) < expected:
            names.extend(f"Ch{idx+1}" for idx in range(len(names), expected))
        elif len(names) > expected:
            names = names[:expected]
        return names

    def _resolve_sampling_rate(self, params: Dict[str, Any], samples: int) -> Optional[float]:
        params_block = params.get("Parameters", {})
        for key in ("fs", "SamplingFrequency", "SamplingRate", "sfreq"):
            if key in params_block:
                try:
                    return float(params_block[key])
                except (TypeError, ValueError):
                    continue

        for key in ("srate", "sampling_rate", "fs", "SamplingRate"):
            if key in params:
                try:
                    return float(params[key])
                except (TypeError, ValueError):
                    continue

        recording_time = params_block.get("RecordingTime")
        if recording_time not in (None, 0):
            try:
                return float(samples) / float(recording_time)
            except (TypeError, ValueError, ZeroDivisionError):
                return None
        return None

    def _channels_frame(self, params: Dict[str, Any], channel_count: int) -> Optional[pd.DataFrame]:
        channels_meta = params.get("Channels")
        if not channels_meta:
            return None
        frame = pd.DataFrame(channels_meta)
        return frame.head(channel_count).reset_index(drop=True)

    def _export_filename(
        self, filename: Optional[str], params: Optional[Dict[str, Any]], target: Path
    ) -> str:
        if filename:
            stem = Path(filename).stem
        elif params is not None:
            candidate = params.get("OutputFile") or params.get("Parameters", {}).get("Filename")
            stem = Path(candidate).stem if candidate else target.name
        else:
            stem = target.name
        return f"{stem}.bin" if not stem.endswith(".bin") else stem

    def _prepare_params_for_export(
        self,
        params: Optional[Dict[str, Any]],
        raw: BaseRaw,
        data_path: Path,
        samples: int,
        channels: int,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = copy.deepcopy(params) if params is not None else {}
        parameters = payload.setdefault("Parameters", {})
        parameters["fs"] = float(raw.info["sfreq"])
        parameters["RecordingTime"] = float(samples) / float(raw.info["sfreq"])
        parameters["NumberEEGChannels"] = channels
        parameters.setdefault("NumberAUXChannels", 0)
        parameters.setdefault("Filename", data_path.stem)

        payload["OutputFile"] = data_path.name

        if "Channels" not in payload:
            payload["Channels"] = [
                {"Channel": name, "Position": name, "Active": True} for name in raw.ch_names
            ]
        return payload
