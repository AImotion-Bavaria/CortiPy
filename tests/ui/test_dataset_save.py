"""Recordings are saved as one JSON-LD + one Parquet, named by the filename.

Replaces the old params.json + data.npz pair (which duplicated the data and bloated the
dataset folder). The filename drives both file names, and a name clash is reported so the
caller can overwrite or rename.
"""
from __future__ import annotations

import numpy as np

from cortipy.ui_streamlit import session as ui


def _params(filename="subject01"):
    chans = [{"Channel": f"Ch {i+1}", "Position": p, "Impedance": 5.0 + i}
             for i, p in enumerate(["Fz", "Cz", "Pz", "Oz"])]
    return {
        "Method": "SSVEP", "Device": "UNICORN",
        "Parameters": {"fs": 250, "NumberEEGChannels": 4, "StimFreq": 12.0, "Filename": filename},
        "Channels": chans, "Metadata": {"Participant": {"Code": "P01"}},
    }


def test_saves_only_jsonld_and_parquet_named_by_filename(tmp_path):
    data = np.random.default_rng(0).standard_normal((500, 4)) * 20
    status, out = ui.save_recording_to_dataset(data, _params("subject01"), tmp_path)
    assert status == "saved"
    assert out.name == "subject01.jsonld"

    files = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["raw_data/subject01.parquet", "subject01.jsonld"]
    assert not (tmp_path / "params.json").exists()
    assert not (tmp_path / "data.npz").exists()


def test_recording_round_trips_back(tmp_path):
    data = np.random.default_rng(1).standard_normal((300, 4)) * 25
    ui.save_recording_to_dataset(data, _params("rec"), tmp_path)
    loaded, arr, _ = ui._load_dataset_folder(tmp_path)
    assert loaded is not None and loaded.get("Method") == "SSVEP"
    assert loaded["Parameters"].get("StimFreq") == 12.0
    assert arr is not None and arr.shape == (300, 4)
    assert np.allclose(arr, data, rtol=1e-4)


def test_same_name_reports_exists_and_overwrite_replaces(tmp_path):
    a = np.full((50, 4), 1.0)
    b = np.full((50, 4), 9.0)
    assert ui.save_recording_to_dataset(a, _params("x"), tmp_path)[0] == "saved"
    # a second save under the same name does not clobber; it reports the clash
    status, _ = ui.save_recording_to_dataset(b, _params("x"), tmp_path)
    assert status == "exists"
    _, arr, _ = ui._load_dataset_folder(tmp_path)
    assert float(arr.mean()) == 1.0  # original still intact
    # explicit overwrite replaces it
    assert ui.save_recording_to_dataset(b, _params("x"), tmp_path, overwrite=True)[0] == "saved"
    _, arr2, _ = ui._load_dataset_folder(tmp_path)
    assert float(arr2.mean()) == 9.0


def test_two_different_names_coexist(tmp_path):
    ui.save_recording_to_dataset(np.zeros((10, 4)), _params("a"), tmp_path)
    ui.save_recording_to_dataset(np.zeros((10, 4)), _params("b"), tmp_path)
    jsonlds = sorted(p.name for p in tmp_path.glob("*.jsonld"))
    assert jsonlds == ["a.jsonld", "b.jsonld"]


def test_stem_comes_from_filename(tmp_path):
    assert ui.dataset_recording_stem(_params("my recording 7")) == "my_recording_7"
