"""The run folder must be named after the operator's typed filename, not a generic stem.

Regression: SaveManager always created `{timestamp}_{method}`, so the measurement name was
always the same and never reflected the UI input. Now `Parameters.Filename` drives it, with
same-named runs numbered (never overwritten) and a timestamped fallback when none is typed.
"""
from __future__ import annotations

import numpy as np

from cortipy.ui import SaveManager


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
