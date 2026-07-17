"""A saved session is any folder the loaders can read back — params.json or JSON-LD.

Regression 1: list_saved_sessions returned every subdirectory, so a run interrupted before
it saved became "the latest session" and "Continue experiment" failed with "params.json
missing".

Regression 2: over-correcting to *only* params.json hid JSON-LD folders — but JSON-LD is
the default export container and is a perfectly loadable session. Both params.json and
.jsonld now count.
"""
from __future__ import annotations

import numpy as np

from cortipy.ui_streamlit import session as ui


def _make(root, name, *, files=()):
    d = root / name
    d.mkdir(parents=True)
    for rel, text in files:
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return d


def test_params_json_folder_is_listed(tmp_path):
    good = _make(tmp_path, "20260102-000000_SSVEP", files=[("params.json", "{}")])
    ui.list_saved_sessions.clear()
    assert ui.list_saved_sessions(tmp_path) == [good]


def test_jsonld_folder_is_listed(tmp_path):
    # A folder whose only metadata is a JSON-LD export must still count as a session.
    j = _make(tmp_path, "20260103-000000_SSVEP", files=[("jsonld_export/meta_run-01.jsonld", "{}")])
    ui.list_saved_sessions.clear()
    assert ui.list_saved_sessions(tmp_path) == [j]


def test_interrupted_and_empty_folders_are_skipped(tmp_path):
    _make(tmp_path, "20260717-100801_SSVEP")  # created but never saved -> nothing loadable
    good = _make(tmp_path, "20260716-090000_SSVEP", files=[("params.json", "{}")])
    ui.list_saved_sessions.clear()
    assert ui.list_saved_sessions(tmp_path) == [good]


def test_newest_loadable_folder_is_first(tmp_path):
    older = _make(tmp_path, "20260101-000000_Alpha", files=[("params.json", "{}")])
    newer = _make(tmp_path, "20260202-000000_Alpha", files=[("x.jsonld", "{}")])
    _make(tmp_path, "20260303-000000_Alpha")  # newest name but empty -> excluded
    ui.list_saved_sessions.clear()
    assert ui.list_saved_sessions(tmp_path) == [newer, older]


def test_missing_base_dir_is_empty(tmp_path):
    ui.list_saved_sessions.clear()
    assert ui.list_saved_sessions(tmp_path / "nope") == []


def test_a_jsonld_export_round_trips_back_into_a_session(tmp_path):
    # The real scenario: a run exported as JSON-LD (no params.json) must load back.
    data = np.random.default_rng(0).standard_normal((250, 3))
    params = {
        "Method": "SSVEP", "Device": "UNICORN",
        "Parameters": {"fs": 250, "NumberEEGChannels": 3, "Filename": "rec"}, "Channels": [],
    }
    folder = _make(tmp_path, "sess")
    ui.export_recording(data, params, folder, "JSON-LD", "NPZ")  # -> folder/jsonld_export/*.jsonld

    ui.list_saved_sessions.clear()
    assert folder in ui.list_saved_sessions(tmp_path)

    loaded, arr = ui._load_session_contents(folder)
    assert loaded is not None and loaded.get("Method") == "SSVEP"
