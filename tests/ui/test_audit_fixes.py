"""Regression tests for the batch-A audit fixes."""
from __future__ import annotations

import json

import numpy as np

from cortipy.ui import SaveManager
from cortipy.ui.config import normalize_params
from cortipy.ui_streamlit import session as ui
from cortipy.ui_streamlit.constants import DEVICE_POSITION_DEFAULTS


# --- #3 UNICORN fixed montage -------------------------------------------------
def test_unicorn_default_montage_matches_hardware():
    assert DEVICE_POSITION_DEFAULTS["UNICORN"] == ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]


# --- #2 ReferenceElectrodes survive normalize + round-trip --------------------
def test_normalize_params_keeps_reference_electrodes():
    out = normalize_params({"Method": "SSVEP", "ReferenceElectrodes": [{"Channel": "Ref", "Impedance": 3.0}]})
    assert out.get("ReferenceElectrodes") == [{"Channel": "Ref", "Impedance": 3.0}]


def test_reference_electrodes_survive_save_load(tmp_path):
    params = {
        "Method": "SSVEP", "Device": "UNICORN", "Parameters": {"fs": 250},
        "Channels": [{"Channel": "Ch 1", "Position": "Fz"}],
        "ReferenceElectrodes": [{"Channel": "Ref", "Impedance": 3.0}, {"Channel": "GND", "Impedance": 4.0}],
    }
    folder = SaveManager(tmp_path)(dict(params))
    loaded, _, _ = ui._load_dataset_folder(folder)
    assert loaded.get("ReferenceElectrodes") == params["ReferenceElectrodes"]


# --- #7 BIDS session detection requires a sub-* dir ---------------------------
def test_folder_has_session_requires_sub_dir_for_bids(tmp_path):
    only_dd = tmp_path / "dd"
    only_dd.mkdir()
    (only_dd / "dataset_description.json").write_text("{}")
    assert ui._folder_has_session(only_dd) is False  # no sub-* -> not a loadable session

    real = tmp_path / "real"
    (real / "bids_export" / "sub-01").mkdir(parents=True)
    (real / "bids_export" / "dataset_description.json").write_text("{}")
    assert ui._folder_has_session(real) is True


# --- #8 params never paired with a different run's data ----------------------
def test_missing_declared_datafile_does_not_pair_wrong_data(tmp_path):
    sm = SaveManager(tmp_path)
    active = tmp_path / "exp"
    sm({"Method": "SSVEP", "Parameters": {"fs": 250}, "data": np.full((4, 1), 1.0)}, target_dir=active)
    sm({"Method": "SSVEP", "Parameters": {"fs": 250}, "data": np.full((4, 1), 9.0)}, target_dir=active)
    (active / "data_run-02.npz").unlink()  # newest run's data gone
    _, data, _ = ui._load_dataset_folder(active)
    assert data is None  # rather than silently loading run-1's data


# --- #17 numbered-run loader is numeric, not lexical -------------------------
def test_loader_returns_newest_run_beyond_99(tmp_path):
    active = tmp_path / "exp"
    active.mkdir()
    for run, value in ((99, 99.0), (100, 100.0)):
        np.savez_compressed(active / f"data_run-{run:02d}.npz", data=np.full((3, 1), value))
        (active / f"params_run-{run:02d}.json").write_text(
            json.dumps({"Method": "Alpha", "Parameters": {"fs": 250}, "DataFile": f"data_run-{run:02d}.npz"})
        )
    _, data, _ = ui._load_dataset_folder(active)
    assert float(data.mean()) == 100.0  # run-100, not lexical run-99


# --- #12 validate flags an out-of-range reference channel --------------------
def test_validate_flags_out_of_range_reference_channel():
    issues = ui.validate_params({
        "Method": "SSVEP", "Device": "LSL",
        "Parameters": {"fs": 250, "NumberEEGChannels": 4, "ReferenceChannel": 8},
        "Channels": [{"Channel": f"Ch {i}", "Active": True} for i in range(4)],
    })
    assert any("out of range" in msg for msg in issues)


def test_validate_accepts_in_range_reference_channel():
    issues = ui.validate_params({
        "Method": "SSVEP", "Device": "LSL",
        "Parameters": {"fs": 250, "NumberEEGChannels": 4, "ReferenceChannel": 2},
        "Channels": [{"Channel": f"Ch {i}", "Active": True} for i in range(4)],
    })
    assert not any("out of range" in msg for msg in issues)


# --- #16 UNICORN prepare_for_recording drops the stale handshake packet -------
def test_prepare_for_recording_clears_pending():
    from cortipy.devices.unicorn import UnicornDevice

    dev = UnicornDevice("dummy-port")
    dev._serial = type("S", (), {"reset_input_buffer": lambda self: None})()
    dev._pending = bytearray(b"\x00" * 45)  # a stale packet from the sync fallback
    dev.prepare_for_recording()
    assert len(dev._pending) == 0


# --- #4 live FFT labels align with data columns ------------------------------
def test_live_fft_labels_align_with_data_columns():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cortipy.shared.plotting import plot_fft_live

    freq = np.linspace(0, 125, 64)
    data = np.zeros((64, 4))
    data[10, 0] = 100.0  # spike in data COLUMN 0
    params = {"Device": "UNICORN", "Channels": [
        {"Channel": "Ch 1", "Position": "Oz"}, {"Channel": "Ch 2", "Position": "Pz"},
        {"Channel": "Ch 3", "Position": "Cz"}, {"Channel": "Ch 4", "Position": "Fz"}]}
    plt.figure()
    plot_fft_live(freq, data, "amp", "t", params)
    labels = [line.get_label() for line in plt.gca().get_lines()]
    plt.close("all")
    assert labels[:4] == ["Oz", "Pz", "Cz", "Fz"]  # column i -> channel i, no offset


# --- #1 the uploader/replay path no longer crashes ---------------------------
def test_uploader_replay_does_not_crash_the_sidebar(tmp_path):
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h.py"
    harness.write_text(
        "import numpy as np, streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "ui.render_sidebar_controls()\n"                       # instantiates run_mode_choice radio
        "st.session_state['imported_data'] = np.zeros((10, 3))\n"
        "ui.load_params_into_state({'Method': 'SSVEP', 'Device': 'UNICORN', 'Parameters': {'fs': 250}},\n"
        "                          np.zeros((10, 3)), enable_replay=True)\n"  # what handle_upload does, AFTER the radio
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()
    assert not at.exception  # used to raise StreamlitAPIException (set widget key post-instantiation)
    assert at.session_state["_pending_run_mode"] == ui.RUN_MODE_REPLAY


# --- Channels-to-record stepper: stable key, no revert on +/- ----------------
def test_channels_stepper_updates_without_reverting(tmp_path):
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "st.session_state['general_form']['Device'] = 'UNICORN'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "if '_seeded' not in st.session_state:\n"
        "    st.session_state.setdefault('method_forms', {}).setdefault('SSVEP', {})['NumberEEGChannels'] = 3\n"
        "    st.session_state['_seeded'] = True\n"
        "ui.render_channel_editor('UNICORN')\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()

    def stepper():
        return next(n for n in at.number_input if n.key == "electrodes_count_UNICORN")

    def count():
        return at.session_state["method_forms"]["SSVEP"]["NumberEEGChannels"]

    assert stepper().key == "electrodes_count_UNICORN"  # stable key, not electrodes_count_UNICORN_3
    assert stepper().value == 3

    stepper().set_value(6).run()
    assert not at.exception
    assert count() == 6 and stepper().value == 6

    stepper().set_value(2).run()  # a further change must not snap back
    assert count() == 2 and stepper().value == 2


# --- Numeric method/general steppers: key-only, stick, and re-seed on load ----
def test_numeric_fields_stick_and_reseed_on_load(tmp_path):
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "st.session_state['general_form']['Device'] = 'ActiCHamp'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "if st.session_state.pop('_do_load', False):\n"
        "    ui.load_params_into_state({'Method': 'SSVEP', 'Device': 'ActiCHamp',\n"
        "        'Parameters': {'fs': 500, 'StimFreq': 12, 'NumberEEGChannels': 5}})\n"
        "ui.render_method_form('SSVEP')\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()

    def nec():
        return next(n for n in at.number_input if n.label == "NumberEEGChannels")

    nec().set_value(16).run()
    assert nec().value == 16  # a stepper change sticks (no value= to snap it back)
    nec().set_value(24).run()
    assert nec().value == 24

    at.session_state["_do_load"] = True
    at.run()  # loading a session must re-seed the field (load pops the widget key)
    assert nec().value == 5


# --- Dataset workflow: recording count for the "N recording(s)" caption ------
def test_dataset_recording_count(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ui._dataset_recording_count(empty) == 0

    one = tmp_path / "one"
    one.mkdir()
    (one / "params.json").write_text("{}")
    assert ui._dataset_recording_count(one) == 1

    three = tmp_path / "three"
    three.mkdir()
    (three / "params.json").write_text("{}")
    (three / "params_run-02.json").write_text("{}")
    (three / "params_run-03.json").write_text("{}")
    assert ui._dataset_recording_count(three) == 3

    jsonld_only = tmp_path / "jl"
    (jsonld_only / "jsonld_export").mkdir(parents=True)
    (jsonld_only / "jsonld_export" / "meta_run-01.jsonld").write_text("{}")
    assert ui._dataset_recording_count(jsonld_only) == 1
