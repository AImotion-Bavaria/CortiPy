"""The run folder must be named after the operator's typed filename, not a generic stem.

Regression: SaveManager always created `{timestamp}_{method}`, so the measurement name was
always the same and never reflected the UI input. Now `Parameters.Filename` drives it, with
same-named runs numbered (never overwritten) and a timestamped fallback when none is typed.
"""
from __future__ import annotations

import numpy as np

from cortipy.ui import SaveManager
from cortipy.ui_streamlit import session as ui


def test_folder_is_named_after_the_typed_filename(tmp_path):
    sm = SaveManager(tmp_path)
    out = sm({"Method": "SSVEP", "Parameters": {"Filename": "subject07_trial1"}})
    assert out.name == "subject07_trial1"


def test_filename_is_sanitised(tmp_path):
    sm = SaveManager(tmp_path)
    out = sm({"Method": "SSVEP", "Parameters": {"Filename": "sub 7/trial#1"}})
    assert out.name == "sub_7trial1"  # spaces -> _, other punctuation dropped


def test_repeated_names_are_numbered_not_overwritten(tmp_path):
    sm = SaveManager(tmp_path)
    a = sm({"Method": "Alpha", "Parameters": {"Filename": "rec"}, "data": np.zeros((3, 2))})
    b = sm({"Method": "Alpha", "Parameters": {"Filename": "rec"}, "data": np.ones((3, 2))})
    assert a.name == "rec" and b.name == "rec_2"
    assert a != b and (a / "data.npz").exists() and (b / "data.npz").exists()


def test_without_a_filename_it_falls_back_to_timestamp_and_method(tmp_path):
    sm = SaveManager(tmp_path)
    out = sm({"Method": "ASSR", "Parameters": {"fs": 250}})
    assert out.name.endswith("_ASSR")
    assert out.name[:8].isdigit()  # YYYYMMDD prefix


def test_an_explicit_target_dir_is_still_respected(tmp_path):
    sm = SaveManager(tmp_path)
    target = tmp_path / "my_dataset"
    out = sm({"Method": "Alpha", "Parameters": {"Filename": "ignored"}}, target_dir=target)
    assert out == target


def test_first_recording_uses_plain_names(tmp_path):
    sm = SaveManager(tmp_path)
    active = tmp_path / "exp"
    sm({"Method": "Alpha", "data": np.zeros((3, 2))}, target_dir=active)
    assert (active / "params.json").exists() and (active / "data.npz").exists()


def test_repeated_recordings_into_one_folder_are_numbered_not_overwritten(tmp_path):
    sm = SaveManager(tmp_path)
    active = tmp_path / "exp"
    for value in (1.0, 2.0, 3.0):
        sm({"Method": "Alpha", "data": np.full((3, 2), value)}, target_dir=active)
    data_files = sorted(p.name for p in active.glob("data*.npz"))
    assert data_files == ["data.npz", "data_run-02.npz", "data_run-03.npz"]
    means = {float(np.load(active / n)["data"].mean()) for n in data_files}
    assert means == {1.0, 2.0, 3.0}  # nothing overwritten


def test_loading_a_numbered_folder_returns_the_newest_run(tmp_path):
    sm = SaveManager(tmp_path)
    active = tmp_path / "exp"
    for value in (1.0, 2.0, 3.0):
        sm({"Method": "SSVEP", "Parameters": {"fs": 250}, "data": np.full((4, 2), value)}, target_dir=active)
    params, data, _ = ui._load_dataset_folder(active)
    assert params is not None and data is not None
    assert float(data.mean()) == 3.0  # the latest recording
