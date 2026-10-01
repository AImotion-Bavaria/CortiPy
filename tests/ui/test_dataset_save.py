"""A dataset is one folder: a single SBIDS JSON-LD (named after the folder) that accumulates
every run, plus one Parquet per run in raw_data/. No params.json / data.npz. The JSON-LD
carries the full config verbatim, including the UNICORN COM port, so a load can restore it.
"""
from __future__ import annotations

import numpy as np
import pytest

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

    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
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


@pytest.mark.parametrize(
    "device, method, last_column",
    [
        ("ActiCHamp", "VEP", "Trigger"),  # TriggerChannel 33 = AUX1, right after the EEG block
        ("UNICORN", "SSVEP", "Ch5"),  # no trigger: an extra column just gets its slot name
    ],
)
def test_parquet_columns_match_schema_name_and_column_name_is_the_slot(tmp_path, device, method, last_column):
    """Parquet column == schema:name == Position (Trigger / Ch{n} as fallback); columnName == Ch{n}.

    Channels describes only the EEG block, so a live recording (EEG + trailing trigger/AUX)
    is one column wider than the montage. That used to discard every electrode name and
    save Ch1..N, while the JSON-LD said "Ch 1".. — nothing matched.
    """
    import json

    import pandas as pd

    params = _params("live")
    params.update(Device=device, Method=method)
    params["Parameters"]["TriggerChannel"] = 33
    params["Channels"][2]["Position"] = ""  # a blank electrode row
    folder = tmp_path / "DS"
    # 4 EEG columns + 1 trailing column, as a live recording is saved.
    ui.save_recording_to_dataset(np.zeros((50, 5)), params, folder)

    expected = ["Fz", "Cz", "Ch3", "Oz", last_column]
    assert list(pd.read_parquet(folder / "raw_data" / "live.parquet").columns) == expected

    graph = json.loads((folder / "DS.jsonld").read_text(encoding="utf-8"))["@graph"]
    recording = next(node for node in graph if "schema:CreateAction" in node.get("@type", []))
    data_nodes = recording["schema:variableMeasured"][:5]
    assert [node["schema:name"] for node in data_nodes] == expected
    assert [node["columnName"] for node in data_nodes] == ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5"]
