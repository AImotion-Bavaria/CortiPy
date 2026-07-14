"""Unified dataset wrapper with pandas-like read/convert helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence
import copy

import mne
import pandas as pd
import numpy as np
import importlib

from .bids import BIDSLoader, BIDSLoadResult, ExperimentBinLoader, raw_to_microvolts
from .units import to_volts
from .sbids import read_sbids


class CortiDataset:
    """Container around a single recording (similar to pandas read_* / to_* flow).

    Usage:
        ds = CortiDataset.from_bids("/path/to/bids", subject="01", task="rest")
        # work with ds.data / ds.raw / ds.events
        ds.to_sbids("/tmp/sbids_meta.jsonld", export_format="parquet")
        ds.to_bids("/tmp/new_bids", subject="02", task="rest")
        ds.to_bin("/tmp/bin_export")
    """

    def __init__(self, result: BIDSLoadResult) -> None:
        self.result = result

    # ---- class-level loaders (read_*) ----
    @classmethod
    def from_bids(
        cls,
        bids_root: str | Path,
        *,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
        run: Optional[str] = None,
        allowed_file_structures: Sequence[str] | None = None,
        preload: bool = True,
    ) -> CortiDataset:
        loader = BIDSLoader(bids_root)
        result = loader.read_bids(
            allowed_file_structures=tuple(allowed_file_structures) if allowed_file_structures else (
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
            subject=subject,
            session=session,
            task=task,
            run=run,
            preload=preload,
        )
        return cls(result)

    @classmethod
    def from_sbids(
        cls,
        sbids_path: str | Path,
        *,
        data_roots: Sequence[str | Path] | None = None,
        recording_id: str | None = None,
        preload: bool = True,
    ) -> CortiDataset:
        result = read_sbids(
            sbids_path,
            data_roots=data_roots,
            recording_id=recording_id,
            preload=preload,
            all_recordings=False,
        )
        if isinstance(result, list):
            if not result:
                raise ValueError("SBIDS document did not yield any recordings.")
            result = result[0]
        return cls(result)

    @classmethod
    def from_bin(
        cls,
        bin_root: str | Path,
        *,
        dataset: str | Path | None = None,
        sampling_rate: Optional[float] = None,
        channel_names: Optional[Sequence[str]] = None,
        channel_types: str | Sequence[str] | None = None,
        params_path: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        channel_count: Optional[int] = None,
    ) -> CortiDataset:
        bin_root = Path(bin_root).expanduser()
        loader = ExperimentBinLoader(bin_root if bin_root.is_dir() else bin_root.parent)
        result = loader.read_bin(
            dataset=dataset or bin_root.resolve(),
            sampling_rate=sampling_rate,
            channel_names=channel_names,
            channel_types=channel_types,
            params_path=params_path,
            data_path=data_path,
            channel_count=channel_count,
        )
        return cls(result)

    @classmethod
    def synthetic_sine_trigger(
        cls,
        *,
        sfreq: float = 250.0,
        duration_s: float = 10.0,
        freq_hz: float = 10.0,
        amplitude_uV: float = 20.0,
        trigger_interval_s: float = 1.0,
        channel: str = "Cz",
        stim_label: str = "TRIG",
        method: str = "SyntheticSine",
    ) -> CortiDataset:
        """Generate a simple sine + trigger Raw and wrap it in a CortiDataset."""
        raw = _synthetic_sine_trigger_raw(
            sfreq=sfreq,
            duration_s=duration_s,
            freq_hz=freq_hz,
            amplitude_uV=amplitude_uV,
            trigger_interval_s=trigger_interval_s,
            channel=channel,
            stim_label=stim_label,
        )
        params = {
            "Method": method,
            "Device": "Simulated",
            "Parameters": {
                "fs": float(sfreq),
                "RecordingTime": float(duration_s),
                "Epochs": int(round(duration_s / trigger_interval_s)) if trigger_interval_s > 0 else 1,
                "EpochLength": int(round(trigger_interval_s * 1000.0)),
                "FrequencyHz": float(freq_hz),
                "Amplitude_uV": float(amplitude_uV),
            },
            "Channels": [
                {"Channel": channel, "Position": channel, "Type": "EEG", "Active": True},
                {"Channel": stim_label, "Position": stim_label, "Type": "Stim", "Active": True},
            ],
        }
        channels_df = pd.DataFrame({"name": [channel, stim_label], "type": ["EEG", "STIM"]})
        metadata: dict[str, Any] = {"params": params, "source": "synthetic_sine_trigger"}
        result = BIDSLoadResult(
            raw=raw,
            data=raw_to_microvolts(raw),
            sampling_rate=float(sfreq),
            events=None,
            channels=channels_df,
            metadata=metadata,
            source_path=Path(f"{method}_synthetic.bin"),
            ancillary_files=[],
        )
        return cls(result)

    @classmethod
    def generate_eeg_samples(
        cls,
        *,
        sampling_rate: float = 250.0,
        duration_s: float = 10.0,
        channel_names: Sequence[str] | None = None,
        channel_types: str | Sequence[str] | None = "eeg",
        channel_count: int = 8,
        base_frequencies: Sequence[float] | None = None,
        line_noise: float | None = 50.0,
        noise: float = 5.0,
        drift: float = 0.5,
        seed: int | None = None,
        dataset_name: str = "synthetic_eeg",
        dtype: str | np.dtype | None = None,
    ) -> CortiDataset:
        """Generate synthetic EEG-like data and return it wrapped in a CortiDataset."""
        names = _resolve_channel_names(channel_names, channel_count)
        types = _resolve_channel_types(channel_types, len(names))
        base_freqs = [float(f) for f in (base_frequencies or (2.5, 6.0, 10.0, 18.0, 35.0))]
        dtype_resolved = np.dtype(dtype) if dtype is not None else np.float64

        data = _synthesize_eeg_data(
            sampling_rate=float(sampling_rate),
            duration_s=float(duration_s),
            ch_count=len(names),
            base_freqs=base_freqs,
            noise_level=float(noise),
            drift_level=float(drift),
            line_freq=line_noise,
            seed=seed,
            dtype=dtype_resolved,
        )

        info = mne.create_info(ch_names=names, sfreq=float(sampling_rate), ch_types=types)
        # `data` is microvolts; MNE needs volts. result.data below stays in uV.
        raw = mne.io.RawArray(to_volts(data.T, types), info)
        channels_df = pd.DataFrame({"name": names, "type": [ct.upper() for ct in types]})

        params = {
            "Method": "SyntheticEEG",
            "Device": "Simulated",
            "Parameters": {
                "fs": float(sampling_rate),
                "RecordingTime": float(duration_s),
                "NumberEEGChannels": int(sum(1 for t in types if t.lower() == "eeg")),
                "NumberAUXChannels": int(sum(1 for t in types if t.lower() != "eeg")),
                "Filename": f"{dataset_name}_scalpdata",
            },
            "Channels": [
                {"Channel": name, "Position": name, "Type": ctype, "Active": True}
                for name, ctype in zip(names, types)
            ],
        }

        metadata: dict[str, Any] = {
            "params": params,
            "generator": {
                "sampling_rate": float(sampling_rate),
                "duration_s": float(duration_s),
                "base_frequencies": base_freqs,
                "noise": float(noise),
                "drift": float(drift),
                "line_noise": line_noise,
                "seed": seed,
                "dtype": str(dtype_resolved),
            },
        }

        result = BIDSLoadResult(
            raw=raw,
            data=data,
            sampling_rate=float(sampling_rate),
            events=None,
            channels=channels_df,
            metadata=metadata,
            source_path=Path(f"{dataset_name}_synthetic"),
            ancillary_files=[],
        )
        return cls(result)

    @classmethod
    def from_synthetic(
        cls,
        sampling_rate: float = 250.0,
        duration_s: float = 10.0,
        channel_names: Sequence[str] | None = None,
        channel_types: str | Sequence[str] | None = "eeg",
        channel_count: int = 8,
        base_frequencies: Sequence[float] | None = None,
        line_noise: float | None = 50.0,
        noise: float = 5.0,
        drift: float = 0.5,
        seed: int | None = None,
        dataset_name: str = "synthetic_eeg",
        dtype: str | np.dtype | None = None,
    ) -> CortiDataset:
        """Alias for generate_eeg_samples for backward compatibility."""
        return cls.generate_eeg_samples(
            sampling_rate=sampling_rate,
            duration_s=duration_s,
            channel_names=channel_names,
            channel_types=channel_types,
            channel_count=channel_count,
            base_frequencies=base_frequencies,
            line_noise=line_noise,
            noise=noise,
            drift=drift,
            seed=seed,
            dataset_name=dataset_name,
            dtype=dtype,
        )

    # ---- conversions (to_*) ----
    def to_bids(
        self,
        bids_root: str | Path,
        *,
        subject: str,
        session: str | None = None,
        task: str | None = None,
        run: str | None = None,
        modality: str = "eeg",
        format: str = "fif",
        overwrite: bool = False,
        events: Optional[pd.DataFrame] = None,
        channels: Optional[pd.DataFrame] = None,
        sidecar: Optional[dict[str, Any]] = None,
        dataset_description: Optional[dict[str, Any]] = None,
        ancillary_files: Optional[Sequence[tuple[Path, Path]]] = None,
    ) -> Path:
        loader = BIDSLoader(bids_root)
        return loader.to_bids(
            self.result.raw,
            subject=subject,
            session=session,
            task=task,
            run=run,
            modality=modality,
            format=format,
            events=events if events is not None else self.result.events,
            channels=channels if channels is not None else self.result.channels,
            sidecar=sidecar,
            dataset_description=dataset_description,
            ancillary_files=ancillary_files or self.result.ancillary_files,
            overwrite=overwrite,
        )

    def to_sbids(
        self,
        output: str | Path,
        *,
        export_format: str = "parquet",
    ) -> Path:
        from .sbids import SbidsExporter
        raw_dir = Path(output).parent / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)

        dest_path, content_url, file_size = _export_recording_data(
            raw=self.result.raw,
            data=self.result.data,
            raw_dir=raw_dir,
            export_format=export_format,
            stem=Path(output).stem,
        )

        dataset_name = Path(output).stem
        dataset_id = dataset_name.replace(" ", "_").upper()
        exporter = SbidsExporter(dataset_id=dataset_id, dataset_name=dataset_name)

        meta = _meta_from_result(self.result, content_url)
        exporter.add_recording_from_cortipy_json(
            meta_json=meta,
            raw_file=str(dest_path),
            subject_id=meta.get("Metadata", {}).get("Participant", {}).get("Code"),
            file_size_bytes=file_size,
            content_url=content_url,
        )

        output_path = Path(output)
        exporter.save(str(output_path))
        return output_path

    def to_bin(
        self,
        target_dir: str | Path,
        *,
        params: Optional[dict[str, Any]] = None,
        sampling_rate: Optional[float] = None,
        ch_names: Optional[Sequence[str]] = None,
        channel_types: str | Sequence[str] | None = None,
        dtype: str | Any = "float64",
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> tuple[Path, Path]:
        loader = ExperimentBinLoader(target_dir)
        return loader.write_bin(
            self.result.raw,
            target_dir=target_dir,
            params=params,
            sampling_rate=sampling_rate,
            ch_names=ch_names,
            channel_types=channel_types,
            dtype=dtype,
            filename=filename,
            overwrite=overwrite,
        )

    # ---- evaluations ----
    def evaluate(
        self,
        method: str,
        *,
        params: Optional[dict[str, Any]] = None,
        show_plots: bool = False,
        save_plots: bool = False,
        save_dir: str | Path | None = None,
        figure_prefix: str | None = None,
        store_result: bool = True,
    ) -> dict[str, Any]:
        """Run any available evaluator (Alpha, SSVEP, BERA, P300, VEP, ASSR).

        This mirrors the pandas-style workflow:
            ds = CortiDataset.from_bids(...)
            eval_result = ds.evaluate(\"alpha\")
            ds.to_sbids(...)
        """
        method_key = str(method).lower()
        evaluator_cls = _resolve_evaluator(method_key)
        prepared_params = self._prepare_eval_params(method_key, params, show_plots)

        from cortipy.core.context import ModuleContext

        ctx = ModuleContext(prepared_params)
        evaluator = evaluator_cls(
            show_plots=show_plots,
            save_plots=save_plots,
            save_dir=save_dir,
            figure_prefix=figure_prefix,
        )  # type: ignore[arg-type]
        evaluator.evaluate(ctx)
        evaluation = ctx.params.get("Evaluation", {})

        if store_result:
            meta = dict(self.metadata or {})
            meta["Evaluation"] = copy.deepcopy(evaluation)
            self.result.metadata = meta

        return evaluation

    def evaluate_alpha(self, **kwargs) -> dict[str, Any]:
        return self.evaluate("alpha", **kwargs)

    def evaluate_ssvep(self, **kwargs) -> dict[str, Any]:
        return self.evaluate("ssvep", **kwargs)

    def evaluate_bera(self, **kwargs) -> dict[str, Any]:
        return self.evaluate("bera", **kwargs)

    def evaluate_p300(self, **kwargs) -> dict[str, Any]:
        return self.evaluate("p300", **kwargs)

    def evaluate_vep(self, **kwargs) -> dict[str, Any]:
        return self.evaluate("vep", **kwargs)

    def evaluate_assr(self, **kwargs) -> dict[str, Any]:
        return self.evaluate("assr", **kwargs)

    # ---- augmentation helpers ----
    def inject_synthetic_trigger(
        self,
        *,
        step_ms: float | None = None,
        step_samples: int | None = None,
        channel_name: str = "TRIG_SYN",
    ) -> int | None:
        """Append a synthetic stim channel at a fixed interval if none exists.

        Returns the 0-based index of the stim channel in ``raw.ch_names`` or ``None`` on failure.
        """
        raw = self.result.raw
        existing = [i for i, ct in enumerate(raw.get_channel_types()) if ct == "stim"]
        if existing:
            return existing[0]

        try:
            fs = float(raw.info["sfreq"])
        except Exception:
            return None
        if not np.isfinite(fs) or fs <= 0:
            return None

        if step_samples is None:
            if step_ms is None:
                meta = self.metadata
                if isinstance(meta, dict):
                    params_block = meta.get("params") or meta.get("Parameters") or {}
                    try:
                        step_ms = float(params_block.get("EpochLength", 0))
                    except Exception:
                        step_ms = None
            if step_ms is not None:
                try:
                    step_samples = int(round((float(step_ms) / 1000.0) * fs))
                except Exception:
                    step_samples = None
        if step_samples is None or step_samples <= 0:
            return None

        stim = np.zeros(raw.n_times, dtype=float)
        stim[::step_samples] = 1.0
        stim_info = mne.create_info(ch_names=[channel_name], sfreq=fs, ch_types=["stim"])
        stim_raw = mne.io.RawArray(stim[np.newaxis, :], stim_info)
        raw.add_channels([stim_raw], force_update_info=True)
        raw.set_channel_types({channel_name: "stim"})

        # Keep result arrays in sync
        self.result.data = raw_to_microvolts(raw)
        ch_types = raw.get_channel_types()
        ch_names = raw.ch_names
        self.result.channels = pd.DataFrame({"name": ch_names, "type": [ct.upper() for ct in ch_types]})

        # Update metadata params and channels for downstream evaluators/montage
        meta = self.metadata
        if isinstance(meta, dict):
            params_block = meta.get("params") or meta.get("Parameters") or {}
            if isinstance(params_block, dict):
                params_block["TriggerChannel"] = len(ch_names)  # 1-based
                meta.setdefault("Parameters", params_block)
            channels_block = meta.get("Channels")
            if isinstance(channels_block, list):
                channels_block.append({"Channel": channel_name, "Position": channel_name, "Type": "Stim", "Active": True})
            else:
                meta["Channels"] = [{"Channel": nm, "Position": nm, "Type": "EEG", "Active": True} for nm in ch_names[:-1]] + [
                    {"Channel": channel_name, "Position": channel_name, "Type": "Stim", "Active": True}
                ]
            self.result.metadata = meta

        return len(raw.ch_names) - 1

    # ---- convenience accessors ----
    @property
    def data(self):
        return self.result.data

    @property
    def raw(self):
        return self.result.raw

    @property
    def events(self):
        return self.result.events

    @property
    def channels(self):
        return self.result.channels

    @property
    def sampling_rate(self) -> float:
        return float(self.result.sampling_rate)

    @property
    def metadata(self) -> dict[str, Any]:
        meta = self.result.metadata or {}
        return dict(meta)

    # ---- internal helpers ----
    def _prepare_eval_params(
        self,
        method_key: str,
        params: Optional[dict[str, Any]],
        show_plots: bool,
    ) -> dict[str, Any]:
        try:
            from tests.regression.dummy_params import dummy_params_alpha  # type: ignore
        except Exception:
            dummy_params_alpha = None  # type: ignore

        base: dict[str, Any]
        if params is not None:
            if not isinstance(params, dict):
                raise ValueError("Evaluator parameters must be provided as a mapping.")
            base = copy.deepcopy(params)
        else:
            meta_params = None
            if isinstance(self.metadata, dict):
                meta_params = self.metadata.get("params") or self.metadata.get("Parameters")
            if isinstance(meta_params, dict):
                base = copy.deepcopy(meta_params)
            elif callable(dummy_params_alpha) and method_key == "alpha":
                base = dummy_params_alpha()
            else:
                base = {}

        if not isinstance(base, dict):
            base = {}

        base = copy.deepcopy(base)
        base["Method"] = method_key.upper()
        base["data"] = self.result.data
        params_block = base.get("Parameters")
        if not isinstance(params_block, dict):
            params_block = {}
        fs_value = params_block.get("fs")
        if not isinstance(fs_value, (int, float)) or not np.isfinite(fs_value) or fs_value <= 0:
            fs_candidate = getattr(self.result.raw.info, "sfreq", None)
            if isinstance(fs_candidate, (int, float)) and np.isfinite(fs_candidate) and fs_candidate > 0:
                fs_value = float(fs_candidate)
            else:
                fs_value = float(self.sampling_rate)
        if not np.isfinite(fs_value) or fs_value <= 0:
            raise ValueError("Sampling rate (fs) must be a positive finite value for evaluation.")
        params_block["fs"] = float(fs_value)
        base["Parameters"] = params_block

        channels_block = base.get("Channels")
        if not isinstance(channels_block, (list, tuple)) or not channels_block:
            channels_block = [
                {"Channel": name, "Position": name, "Type": "EEG", "Active": True}
                for name in self.result.raw.ch_names
            ]
        base["Channels"] = channels_block
        base["ReportAnalyzer"] = not show_plots
        return base


def _export_recording_data(raw, data: np.ndarray, raw_dir: Path, export_format: str, stem: str) -> tuple[Path, str, int]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if export_format == "npz":
        dest_path = raw_dir / f"{stem}.npz"
        np.savez_compressed(dest_path, data=data)
    elif export_format == "copy":
        dest_path = raw_dir / f"{stem}.bin"
        data.astype(np.float64, copy=False).tofile(dest_path)
    elif export_format in {"edf", "hdf5", "zarr"}:
        dest_path = raw_dir / f"{stem}.{_ext_for_format(export_format)}"
        helper = BIDSLoader(raw_dir)
        helper._write_raw(raw, dest_path, format=export_format, overwrite=True)
    else:
        dest_path = raw_dir / f"{stem}.parquet"
        frame = pd.DataFrame(data, columns=list(raw.ch_names))
        frame.to_parquet(dest_path)
    size = dest_path.stat().st_size
    return dest_path, f"raw_data/{dest_path.name}", size


def _meta_from_result(result: BIDSLoadResult, source_rel: str) -> dict[str, Any]:
    params = {}
    if isinstance(result.metadata, dict) and "params" in result.metadata:
        params = result.metadata["params"]
    participant = params.get("Metadata", {}).get("Participant", {}) if isinstance(params, dict) else {}
    subject = participant.get("Code") if isinstance(participant, dict) else None
    # Force channel list to match the actual raw channels to avoid mismatches on import
    channels = [{"Channel": name, "Position": name, "Active": True} for name in result.raw.ch_names]
    # Ensure Parameters contains sampling rate for tabular exports
    parameters_block = params.get("Parameters", {}) if isinstance(params, dict) else {}
    if "fs" not in parameters_block and getattr(result, "sampling_rate", None):
        try:
            parameters_block["fs"] = float(result.sampling_rate)
        except Exception:
            parameters_block.setdefault("fs", float(result.raw.info.get("sfreq", 0.0)))
    return {
        "Method": params.get("Method", "Import") if isinstance(params, dict) else "Import",
        "Device": params.get("Device", "Unknown") if isinstance(params, dict) else "Unknown",
        "Parameters": parameters_block,
        "Channels": channels,
        "Metadata": {"Participant": {"Code": subject}} if subject else {},
        "DataFile": source_rel,
    }


def _ext_for_format(fmt: str) -> str:
    mapping = {"edf": "edf", "hdf5": "hdf5", "zarr": "zarr"}
    return mapping.get(fmt.lower(), fmt.lower())


def _resolve_evaluator(method_key: str):
    registry = {
        "alpha": ("cortipy.evaluation.alpha", "AlphaEvaluator"),
        "ssvep": ("cortipy.evaluation.ssvep", "SsvepEvaluator"),
        "bera": ("cortipy.evaluation.bera", "BeraEvaluator"),
        "p300": ("cortipy.evaluation.p300", "P300Evaluator"),
        "vep": ("cortipy.evaluation.vep", "VepEvaluator"),
        "assr": ("cortipy.evaluation.assr", "AssrEvaluator"),
    }
    module_class = registry.get(method_key.lower())
    if not module_class:
        raise ValueError(f"Unknown evaluator '{method_key}'. Available: {sorted(registry)}")
    module_name, class_name = module_class
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise ImportError(f"{class_name} not found in {module_name}")
    return getattr(module, class_name)


def _resolve_channel_names(channel_names: Sequence[str] | None, channel_count: int) -> list[str]:
    if channel_names is not None:
        return list(channel_names)
    default = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "Fz",
        "Cz",
        "Pz",
        "Oz",
    ]
    if channel_count <= len(default):
        return default[:channel_count]
    extra = [f"Ch{i+1}" for i in range(len(default), channel_count)]
    return default + extra


def _resolve_channel_types(channel_types: str | Sequence[str] | None, channel_count: int) -> list[str]:
    if channel_types is None:
        return ["eeg"] * channel_count
    if isinstance(channel_types, str):
        return [channel_types] * channel_count
    types = list(channel_types)
    if len(types) < channel_count and types:
        types.extend(types[-1:] * (channel_count - len(types)))
    return types[:channel_count] if types else ["eeg"] * channel_count


def _synthetic_sine_trigger_raw(
    *,
    sfreq: float,
    duration_s: float,
    freq_hz: float,
    amplitude_uV: float,
    trigger_interval_s: float,
    channel: str,
    stim_label: str,
) -> mne.io.Raw:
    samples = int(round(sfreq * duration_s))
    t = np.arange(samples) / sfreq
    sine = amplitude_uV * np.sin(2 * np.pi * freq_hz * t)
    trigger = np.zeros_like(sine)
    every = max(1, int(round(trigger_interval_s * sfreq)))
    trigger[::every] = 1.0
    data = np.vstack([sine, trigger])
    ch_types = ["eeg", "stim"]
    info = mne.create_info(ch_names=[channel, stim_label], sfreq=sfreq, ch_types=ch_types)
    # sine is in microvolts (amplitude_uV); the trigger is unitless and must not be scaled.
    return mne.io.RawArray(to_volts(data, ch_types), info)


def _synthesize_eeg_data(
    *,
    sampling_rate: float,
    duration_s: float,
    ch_count: int,
    base_freqs: Sequence[float],
    noise_level: float,
    drift_level: float,
    line_freq: float | None,
    seed: int | None,
    dtype: np.dtype,
) -> np.ndarray:
    samples = max(1, int(round(sampling_rate * duration_s)))
    dtype = np.dtype(dtype)
    t = np.arange(samples, dtype=dtype) / float(sampling_rate)
    rng = np.random.default_rng(seed)
    base_freqs = list(base_freqs) if base_freqs else [10.0]

    def synth_channel() -> np.ndarray:
        sig = np.zeros(samples, dtype=dtype)
        dominant = rng.choice(base_freqs)
        for freq in base_freqs:
            amp = rng.uniform(5.0, 20.0)
            if freq == dominant:
                amp *= 1.4
            phase = rng.uniform(0, 2 * np.pi)
            jitter = rng.uniform(-0.2, 0.2) * freq
            sig += amp * np.sin(2 * np.pi * (freq + jitter) * t + phase)

        drift_phase = rng.uniform(0, 2 * np.pi)
        sig += drift_level * np.sin(2 * np.pi * 0.2 * t + drift_phase)

        colored = _colored_noise(rng, samples, alpha=1.0, dtype=dtype)
        sig += noise_level * colored

        if line_freq is not None:
            sig += 2.0 * np.sin(2 * np.pi * float(line_freq) * t + rng.uniform(0, 2 * np.pi))

        gain = 1.0 + 0.1 * (rng.random() - 0.5)
        return gain * sig

    data = np.empty((samples, ch_count), dtype=dtype)
    for idx in range(ch_count):
        data[:, idx] = synth_channel()
    return data


def _colored_noise(
    rng: np.random.Generator, samples: int, alpha: float = 1.0, dtype: np.dtype = np.float64
) -> np.ndarray:
    """Generate approximate 1/f^alpha noise using frequency-domain shaping."""
    dtype = np.dtype(dtype)
    freqs = np.fft.rfftfreq(samples)
    spectrum = rng.standard_normal(freqs.shape) + 1j * rng.standard_normal(freqs.shape)
    freqs[0] = freqs[1] if freqs.size > 1 else 1.0
    spectrum /= np.maximum(freqs, 1e-6) ** (alpha / 2.0)
    noise = np.fft.irfft(spectrum, n=samples)
    std = noise.std()
    result = noise / std if std > 0 else noise
    return np.asarray(result, dtype=dtype)
