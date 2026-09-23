"""Loading a dataset must never turn it into the save target.

Regression: the load paths set active_dataset_dir to the loaded folder, so the *next*
recording saved into it and overwrote the loaded dataset's params.json/data.npz. Loading
now leaves the save target empty; recording writes a fresh folder and the loaded data is
left untouched.
"""
from __future__ import annotations

import json

import numpy as np

from cortipy.ui import SaveManager
from cortipy.ui_streamlit import session as ui

HARNESS = (
    "import streamlit as st\n"
    "from pathlib import Path\n"
    "from cortipy.ui_streamlit import session as ui\n"
    "ui.ensure_state()\n"
    "ui.render_saved_sessions(Path(st.session_state['_probe_base']))\n"
)


def test_loading_a_session_leaves_the_save_target_empty(tmp_path):
    import numpy.testing  # noqa: F401  (guards the macOS numpy fork probe)
    from streamlit.testing.v1 import AppTest

    sess = tmp_path / "subj01_ssvep"
    sess.mkdir()
    (sess / "params.json").write_text(
        json.dumps({"Method": "SSVEP", "Device": "UNICORN", "Parameters": {"fs": 250}})
    )
    np.savez_compressed(sess / "data.npz", data=np.full((10, 3), 1.0))

    harness = tmp_path / "harness.py"
    harness.write_text(HARNESS)

    at = AppTest.from_file(str(harness), default_timeout=60)
    at.session_state["_probe_base"] = str(tmp_path)
    at.run()
    assert not at.exception, at.exception

    button = next(b for b in at.button if b.label == "Load session into editor")
    button.click().run()
    assert not at.exception, at.exception

    # The loaded folder must NOT have become the save target.
    assert at.session_state["active_dataset_dir"] == ""
    assert ui.active_dataset_path(str(tmp_path)) is None
    # And the loaded recording is still on disk, untouched.
    assert float(np.load(sess / "data.npz")["data"].mean()) == 1.0


def test_recording_with_no_active_target_writes_a_fresh_folder(tmp_path):
    # The outcome that protects the loaded dataset: with the target empty (as after a load),
    # a recording lands in a new filename-named folder, not the loaded one.
    loaded = tmp_path / "subj01"
    loaded.mkdir()
    np.savez_compressed(loaded / "data.npz", data=np.full((5, 2), 1.0))
    (loaded / "params.json").write_text("{}")

    out = SaveManager(tmp_path)(
        {"Method": "SSVEP", "Parameters": {"Filename": "subj02"}, "data": np.full((5, 2), 9.0)},
        target_dir=None,
    )
    assert out.name == "subj02" and out != loaded
    assert float(np.load(loaded / "data.npz")["data"].mean()) == 1.0  # original untouched
