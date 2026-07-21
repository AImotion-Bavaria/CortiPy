"""A dataset is one folder: a single SBIDS JSON-LD (named after the folder) that accumulates
every run, plus one Parquet per run in raw_data/. No params.json / data.npz. The JSON-LD
carries the full config verbatim, including the UNICORN COM port, so a load can restore it.
"""
from __future__ import annotations

import numpy as np

from cortipy.ui_streamlit import session as ui


def _params(filename="subject01", port="COM15"):
    chans = [{"Channel": f"Ch {i+1}", "Position": p, "Impedance": 5.0 + i}
             for i, p in enumerate(["Fz", "Cz", "Pz", "Oz"])]
    return {
        "Method": "SSVEP", "Device": "UNICORN",
        "Parameters": {"fs": 250, "NumberEEGChannels": 4, "StimFreq": 12.0,
                       "Filename": filename, "UNICORNPort": port, "UNICORNDeviceName": "UN-2023.05.03"},
        "Channels": chans, "Metadata": {"Participant": {"Code": "P01"}},
    }


def test_one_jsonld_named_after_the_folder_plus_a_parquet_per_run(tmp_path):
    folder = tmp_path / "MyDataset"
    status, out = ui.save_recording_to_dataset(np.zeros((100, 4)), _params("run1"), folder)
    assert status == "saved"
    assert out.name == "MyDataset.jsonld"  # named after the folder, not the run

    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    assert files == ["MyDataset.jsonld", "raw_data/run1.parquet"]
    assert not (folder / "params.json").exists() and not (folder / "data.npz").exists()


def test_multiple_runs_accumulate_in_one_jsonld(tmp_path):
    folder = tmp_path / "DS"
    ui.save_recording_to_dataset(np.full((50, 4), 1.0), _params("run1"), folder)
    ui.save_recording_to_dataset(np.full((50, 4), 2.0), _params("run2"), folder)
    assert len(list(folder.glob("*.jsonld"))) == 1  # still a single dataset document
    assert sorted(p.name for p in (folder / "raw_data").glob("*.parquet")) == ["run1.parquet", "run2.parquet"]


def test_load_returns_the_newest_run_and_restores_the_com_port(tmp_path):
    folder = tmp_path / "DS"
    ui.save_recording_to_dataset(np.full((50, 4), 1.0), _params("run1", port="COM3"), folder)
    ui.save_recording_to_dataset(np.full((50, 4), 2.0), _params("run2", port="COM15"), folder)

    loaded, arr, _ = ui._load_dataset_folder(folder)
    assert loaded is not None
    assert loaded["Parameters"].get("Filename") == "run2"          # newest run
    assert loaded["Parameters"].get("UNICORNPort") == "COM15"      # port restored
    assert loaded["Parameters"].get("StimFreq") == 12.0
    assert arr is not None and float(arr.mean()) == 2.0


def test_same_run_name_reports_exists_and_overwrite_replaces_it(tmp_path):
    folder = tmp_path / "DS"
    ui.save_recording_to_dataset(np.full((50, 4), 1.0), _params("rec"), folder)
    # a second run of the same name is not silently duplicated; it is reported
    assert ui.save_recording_to_dataset(np.full((50, 4), 9.0), _params("rec"), folder)[0] == "exists"
    # overwrite replaces that run (and its parquet), still one jsonld
    assert ui.save_recording_to_dataset(np.full((50, 4), 9.0), _params("rec"), folder, overwrite=True)[0] == "saved"
    _, arr, _ = ui._load_dataset_folder(folder)
    assert float(arr.mean()) == 9.0
    assert len(list(folder.glob("*.jsonld"))) == 1
    assert sorted(p.name for p in (folder / "raw_data").glob("*.parquet")) == ["rec.parquet"]


def test_stem_comes_from_filename(tmp_path):
    assert ui.dataset_recording_stem(_params("my recording 7")) == "my_recording_7"
