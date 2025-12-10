"""Tests for the BIDSLoader helper."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from cortipy.shared.bids import BIDSLoader


def _create_dummy_bids(tmp_path: Path):
    root = tmp_path / "bids_root"
    data_dir = root / "sub-01" / "ses-01" / "eeg"
    data_dir.mkdir(parents=True)

    sfreq = 100.0
    times = np.arange(0, 1.0, 1 / sfreq)
    samples = np.vstack([np.sin(2 * np.pi * 5 * times), np.cos(2 * np.pi * 5 * times)])
    info = mne.create_info(ch_names=["Cz", "Pz"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(samples, info)
    raw_path = data_dir / "sub-01_ses-01_task-rest_run-01_eeg.fif"
    raw.save(raw_path, overwrite=True)

    events_path = data_dir / "sub-01_ses-01_task-rest_run-01_events.tsv"
    pd.DataFrame(
        {
            "onset": [0.1, 0.5],
            "duration": [0.05, 0.05],
            "trial_type": ["A", "B"],
        }
    ).to_csv(events_path, sep="\t", index=False)

    events_json = data_dir / "sub-01_ses-01_task-rest_run-01_events.json"
    events_json.write_text(json.dumps({"trial_type": {"Description": "Dummy"}}))

    electrodes_path = data_dir / "sub-01_ses-01_space-CapTrak_electrodes.tsv"
    electrodes_path.write_text("name\tx\ty\tz\nCz\t0\t0\t1\nPz\t0\t1\t0\n")

    coordsys_path = data_dir / "sub-01_ses-01_space-CapTrak_coordsystem.json"
    coordsys_path.write_text(json.dumps({"IntendedFor": "EEG"}))

    scans_path = root / "sub-01" / "ses-01" / "sub-01_ses-01_scans.tsv"
    scans_path.parent.mkdir(parents=True, exist_ok=True)
    scans_path.write_text("filename\tacq_time\nsub-01_ses-01_task-rest_eeg.vhdr\t2024-01-01\n")

    participants_tsv = root / "participants.tsv"
    participants_tsv.write_text("participant_id\tage\nsub-01\t30\n")

    sidecar_path = data_dir / "sub-01_ses-01_task-rest_run-01_eeg.json"
    sidecar_path.write_text(json.dumps({"PowerLineFrequency": 50}))

    description_path = root / "dataset_description.json"
    description_path.write_text(json.dumps({"Name": "Dummy dataset"}))

    return root, raw_path.resolve(), sfreq, samples.shape[1]


def _create_multi_bids(tmp_path: Path):
    root = tmp_path / "multi_bids"
    pairs = []
    for sub in ("01", "02"):
        data_dir = root / f"sub-{sub}" / "ses-01" / "eeg"
        data_dir.mkdir(parents=True)
        sfreq = 50.0
        samples = np.vstack(
            [np.sin(2 * np.pi * 3 * np.arange(0, 1.0, 1 / sfreq)), np.ones(int(sfreq))]
        )
        info = mne.create_info(ch_names=["Cz", "Pz"], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(samples, info)
        raw_path = data_dir / f"sub-{sub}_ses-01_task-rest_eeg.fif"
        raw.save(raw_path, overwrite=True)
        pairs.append(raw_path.resolve())

    (root / "dataset_description.json").write_text(json.dumps({"Name": "Multi dataset"}))
    return root, pairs


def test_read_bids_loads_raw_events_and_metadata(tmp_path):
    """read_bids should return the recording, associated tables, and sidecars."""
    root, raw_path, sfreq, sample_count = _create_dummy_bids(tmp_path)
    loader = BIDSLoader(root)

    result = loader.read_bids(
        allowed_file_structures=(".fif",),
        subject="01",
        session="01",
        task="rest",
    )

    assert result.source_path == raw_path
    assert result.data.shape == (sample_count, 2)
    assert result.sampling_rate == pytest.approx(sfreq)
    assert result.events is not None and len(result.events) == 2
    assert result.metadata["dataset_description"]["Name"] == "Dummy dataset"
    assert "sub-01_ses-01_task-rest_run-01_eeg.json" in result.metadata["sidecars"]
    ancillary_names = {rel.name for _, rel in result.ancillary_files}
    assert "sub-01_ses-01_space-CapTrak_electrodes.tsv" in ancillary_names
    assert "sub-01_ses-01_space-CapTrak_coordsystem.json" in ancillary_names
    assert "sub-01_ses-01_scans.tsv" in ancillary_names
    assert "participants.tsv" in ancillary_names


def test_read_bids_respects_allowed_structures(tmp_path):
    """Filtering by allowed file structures should prevent loading unsupported extensions."""
    root, _, _, _ = _create_dummy_bids(tmp_path)
    loader = BIDSLoader(root)

    with pytest.raises(FileNotFoundError):
        loader.read_bids(allowed_file_structures=(".edf",))


def test_to_bids_writes_structure_and_roundtrips(tmp_path):
    """to_bids should write a BIDS-like layout and be readable via read_bids."""
    loader = BIDSLoader(tmp_path / "new_bids")
    samples = np.arange(0, 1, 0.01)  # 100 samples
    data = np.stack([samples, samples * 2], axis=1)
    events = pd.DataFrame({"onset": [0.1], "duration": [0.05], "trial_type": ["A"]})
    channels = pd.DataFrame({"name": ["Cz", "Pz"], "type": ["EEG", "EEG"]})
    sidecar = {"PowerLineFrequency": 50}
    description = {"Name": "Exported dataset"}

    data_path = loader.to_bids(
        data,
        sampling_rate=100.0,
        ch_names=["Cz", "Pz"],
        subject="02",
        session="S1",
        task="rest",
        run="01",
        format="fif",
        events=events,
        channels=channels,
        sidecar=sidecar,
        dataset_description=description,
        overwrite=True,
    )

    assert data_path.name == "sub-02_ses-S1_task-rest_run-01_eeg.fif"
    assert data_path.exists()

    reloaded = loader.read_bids(
        allowed_file_structures=(".fif",), subject="02", session="S1", task="rest", run="01"
    )
    np.testing.assert_allclose(reloaded.data, data)
    assert reloaded.events is not None and len(reloaded.events) == 1
    assert reloaded.channels is not None and list(reloaded.channels["name"]) == ["Cz", "Pz"]
    assert reloaded.metadata["dataset_description"]["Name"] == "Exported dataset"
    assert "sub-02_ses-S1_task-rest_run-01_eeg.json" in reloaded.metadata["sidecars"]


def test_parquet_roundtrip_with_sampling_rate(tmp_path):
    """Parquet exports should reload with the SamplingFrequency from the sidecar."""
    loader = BIDSLoader(tmp_path / "tabular_bids")
    data = np.arange(0, 1, 0.01)[:, None]  # 100 samples, 1 channel
    channels = pd.DataFrame({"name": ["Cz"], "type": ["EEG"]})

    path = loader.to_bids(
        data,
        sampling_rate=200.0,
        ch_names=["Cz"],
        subject="10",
        session="S1",
        task="rest",
        run="02",
        format="parquet",
        channels=channels,
        overwrite=True,
    )

    assert path.suffix == ".parquet"
    result = loader.read_bids(
        allowed_file_structures=(".parquet",),
        subject="10",
        session="S1",
        task="rest",
        run="02",
    )
    assert result.sampling_rate == pytest.approx(200.0)
    np.testing.assert_allclose(result.data[:, 0], data[:, 0])


def test_read_bids_dataset_loads_all_matching_recordings(tmp_path):
    """read_bids_dataset should return all filtered recordings."""
    root, pairs = _create_multi_bids(tmp_path)
    loader = BIDSLoader(root)

    results = loader.read_bids_dataset(allowed_file_structures=(".fif",), session="01", task="rest")

    assert len(results) == len(pairs)
    loaded_paths = {res.source_path for res in results}
    assert set(pairs) == loaded_paths


def test_to_bids_copies_ancillary_files(tmp_path):
    """to_bids should copy ancillary TSV/JSON files when provided."""
    root, raw_path, _, _ = _create_dummy_bids(tmp_path)
    loader = BIDSLoader(root)
    result = loader.read_bids(allowed_file_structures=(".fif",))

    export_root = tmp_path / "export"
    target = BIDSLoader(export_root)
    out_path = target.to_bids(
        result.raw,
        subject="01",
        session="01",
        task="rest",
        run="01",
        format="fif",
        events=result.events,
        channels=result.channels,
        sidecar=next(iter(result.metadata["sidecars"].values())),
        dataset_description=result.metadata.get("dataset_description"),
        ancillary_files=result.ancillary_files,
        overwrite=True,
    )

    assert out_path.exists()
    ancillary_rel = [rel for _, rel in result.ancillary_files]
    for rel in ancillary_rel:
        exported = export_root / rel
        assert exported.exists()
