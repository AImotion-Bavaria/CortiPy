"""Only resumable folders count as saved sessions.

Regression: list_saved_sessions returned every subdirectory, so an interrupted run or an
export-only folder became "the latest session" and "Continue experiment" then failed with
"params.json missing in ...". A session must have a params.json to be listed.
"""
from __future__ import annotations

from cortipy.ui_streamlit import session as ui


def _make_session(root, name, *, with_params):
    d = root / name
    d.mkdir()
    if with_params:
        (d / "params.json").write_text("{}", encoding="utf-8")
    return d


def test_folders_without_params_json_are_skipped(tmp_path):
    _make_session(tmp_path, "20260717-100801_SSVEP", with_params=False)  # interrupted run
    good = _make_session(tmp_path, "20260717-090000_SSVEP", with_params=True)
    (tmp_path / "jsonld_export").mkdir()  # a stray export folder, not a session

    ui.list_saved_sessions.clear()
    result = ui.list_saved_sessions(tmp_path)

    assert result == [good]


def test_latest_is_the_newest_folder_that_has_params(tmp_path):
    older = _make_session(tmp_path, "20260101-000000_Alpha", with_params=True)
    newer = _make_session(tmp_path, "20260202-000000_Alpha", with_params=True)
    _make_session(tmp_path, "20260303-000000_Alpha", with_params=False)  # newest name, but empty

    ui.list_saved_sessions.clear()
    result = ui.list_saved_sessions(tmp_path)

    assert result == [newer, older]  # newest-with-params first, empty one excluded


def test_missing_base_dir_is_empty(tmp_path):
    ui.list_saved_sessions.clear()
    assert ui.list_saved_sessions(tmp_path / "does_not_exist") == []
