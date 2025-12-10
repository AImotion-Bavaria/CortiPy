"""Synthetic dataset helper for examples and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import pandas as pd

from .bids import BIDSLoadResult, BIDSLoader, ExperimentBinLoader


@dataclass
class DummyDatasetGenerator:
    """Create deterministic sine-wave datasets and export them to BIDS/bin layouts."""

    sampling_rate: float = 250.0
    duration_s: float = 5.0
    frequencies: Sequence[float] = (10.0,)
    channel_names: Sequence[str] | None = None
    channel_types: str | Sequence[str] | None = "eeg"
    amplitude: float = 1.0
    noise: float = 0.0
    trigger_interval_s: float | None = None
    trigger_pulse_ms: float = 5.0
    method: str = "Dummy"
    device: str = "Simulated"
    seed: int | None = None
    powerline: float | None = None

    def generate(self, *, dataset_name: str = "dummy") -> BIDSLoadResult:
        """Return an in-memory dataset (Raw + params metadata)."""
        samples = max(1, int(round(self.sampling_rate * self.duration_s)))
        t = np.arange(samples, dtype=float) / float(self.sampling_rate)

        eeg_names, eeg_types = self._channel_layout()
        data = self._sine_waves(t, len(eeg_names))

        ch_names = list(eeg_names)
        ch_types = list(eeg_types)
        events = None
        trigger_channel = None

        if self.trigger_interval_s is not None:
            trig = self._trigger_channel(samples)
            data = np.column_stack([data, trig])
            ch_names.append("TRIG")
            ch_types.append("stim")
            events = self._events_from_trigger(trig)
            trigger_channel = len(ch_names)

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=float(self.sampling_rate),
            ch_types=ch_types,
            verbose=False,
        )
        raw = mne.io.RawArray(data.T, info)
        channels_df = pd.DataFrame({"name": ch_names, "type": [ct.upper() for ct in ch_types]})

        params = self._params_payload(
            dataset_name=dataset_name,
            ch_names=ch_names,
            ch_types=ch_types,
            eeg_channel_count=len(eeg_names),
            trigger_channel=trigger_channel,
            recording_time=data.shape[0] / float(self.sampling_rate),
        )
        metadata: dict[str, Any] = {
            "params": params,
            "generator": {
                "sampling_rate": self.sampling_rate,
                "duration_s": self.duration_s,
                "frequencies": list(self.frequencies),
                "noise": self.noise,
                "seed": self.seed,
            },
        }

        return BIDSLoadResult(
            raw=raw,
            data=data,
            sampling_rate=float(self.sampling_rate),
            events=events,
            channels=channels_df,
            metadata=metadata,
            source_path=Path(f"{dataset_name}_synthetic"),
            ancillary_files=[],
        )

    def export(
        self,
        *,
        dataset_name: str = "dummy",
        bids_root: str | Path | None = None,
        bin_root: str | Path | None = None,
        subject: str = "01",
        session: str | None = None,
        task: str | None = None,
        run: str | None = "01",
        bids_format: str = "fif",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Write the dummy dataset to BIDS and/or bin layouts."""
        result = self.generate(dataset_name=dataset_name)
        outputs: dict[str, Any] = {}

        if bids_root is not None:
            loader = BIDSLoader(bids_root)
            sidecar = {"SamplingFrequency": result.sampling_rate}
            if self.powerline is not None:
                sidecar["PowerLineFrequency"] = self.powerline
            bids_path = loader.to_bids(
                result.raw,
                subject=subject,
                session=session,
                task=task or dataset_name,
                run=run,
                format=bids_format,
                events=result.events,
                channels=result.channels,
                sidecar=sidecar,
                dataset_description={"Name": f"Dummy dataset ({dataset_name})"},
                overwrite=overwrite,
            )
            outputs["bids"] = bids_path

        if bin_root is not None:
            target_dir = Path(bin_root) / dataset_name
            bin_loader = ExperimentBinLoader(bin_root)
            params_payload = result.metadata.get("params") if isinstance(result.metadata, dict) else None
            data_path, params_path = bin_loader.write_bin(
                result.raw,
                target_dir,
                params=params_payload,
                overwrite=overwrite,
            )
            outputs["bin"] = {"data": data_path, "params": params_path}

        outputs["result"] = result
        return outputs

    def _channel_layout(self) -> tuple[list[str], list[str]]:
        if self.channel_names:
            names = list(self.channel_names)
        else:
            base_count = len(self.frequencies) if self.frequencies else 1
            names = [f"Ch{i+1}" for i in range(max(1, base_count))]

        types: list[str]
        if self.channel_types is None:
            types = ["eeg"] * len(names)
        elif isinstance(self.channel_types, str):
            types = [self.channel_types] * len(names)
        else:
            types = list(self.channel_types)
            if len(types) < len(names) and types:
                types.extend(types[-1:] * (len(names) - len(types)))
            types = types[: len(names)]
            if not types:
                types = ["eeg"] * len(names)
        return names, types

    def _sine_waves(self, t: np.ndarray, channel_count: int) -> np.ndarray:
        freq_list = list(self.frequencies) if self.frequencies else [10.0]
        rng = np.random.default_rng(self.seed)
        waves = []
        for idx in range(channel_count):
            freq = float(freq_list[idx % len(freq_list)])
            wave = self.amplitude * np.sin(2 * np.pi * freq * t)
            if self.noise:
                wave = wave + float(self.noise) * rng.standard_normal(wave.shape)
            waves.append(wave)
        return np.column_stack(waves)

    def _trigger_channel(self, samples: int) -> np.ndarray:
        width = max(1, int(round((self.trigger_pulse_ms / 1000.0) * self.sampling_rate)))
        step = max(1, int(round(float(self.trigger_interval_s) * self.sampling_rate)))
        trig = np.zeros(samples, dtype=float)
        for start in range(0, samples, step):
            trig[start : start + width] = 1.0
        return trig

    def _events_from_trigger(self, trig: np.ndarray) -> pd.DataFrame | None:
        rising = np.flatnonzero(np.diff(np.concatenate([[0.0], trig])) > 0)
        if rising.size == 0:
            return None
        onset = rising / float(self.sampling_rate)
        duration = np.full(rising.size, self.trigger_pulse_ms / 1000.0)
        return pd.DataFrame({"onset": onset, "duration": duration, "trial_type": "trigger"})

    def _params_payload(
        self,
        *,
        dataset_name: str,
        ch_names: Sequence[str],
        ch_types: Sequence[str],
        eeg_channel_count: int,
        trigger_channel: int | None,
        recording_time: float,
    ) -> dict[str, Any]:
        channels_meta = []
        for name, ctype in zip(ch_names, ch_types):
            channels_meta.append(
                {
                    "Channel": name,
                    "Position": name,
                    "Type": ctype,
                    "Active": True,
                }
            )

        parameters: dict[str, Any] = {
            "fs": float(self.sampling_rate),
            "RecordingTime": float(recording_time),
            "NumberEEGChannels": int(eeg_channel_count),
            "NumberAUXChannels": 0,
            "Filename": f"{dataset_name}_scalpdata",
        }
        if trigger_channel is not None:
            parameters["TriggerChannel"] = int(trigger_channel)

        return {
            "Method": self.method,
            "Device": self.device,
            "Parameters": parameters,
            "Channels": channels_meta,
        }
