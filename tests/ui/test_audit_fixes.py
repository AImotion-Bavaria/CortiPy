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


# --- Electrode edits must survive whatever redraws the table next ------------
def test_electrode_position_edit_survives_a_table_rebuild(tmp_path):
    """A typed position used to be lost when anything rebuilt the table afterwards.

    The edit lives in the data editor's own widget state until the end of the run, so an
    impedance read or a channel-count change (both of which rebuild the editor) restored
    the previous values over it. The on_change handler now stores the edit right away.
    """
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "st.session_state['general_form']['Device'] = 'UNICORN'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "rows = ui.render_channel_editor('UNICORN')\n"
        "if st.session_state.pop('_edit_now', False):\n"
        "    st.session_state['_fake_editor'] = {'edited_rows': {2: {'Position': 'Oz'}}}\n"
        "    ui._persist_channel_edits('UNICORN', '_fake_editor',\n"
        "                              [str(r['Channel']) for r in rows])\n"
        "    ui._bump_channel_editor_revision('UNICORN')\n"  # what an impedance read used to do
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()

    def position_of(channel):
        rows = at.session_state["channel_tables"]["UNICORN"]
        return next(r["Position"] for r in rows if r["Channel"] == channel)

    edited_channel = at.session_state["channel_tables"]["UNICORN"][2]["Channel"]
    at.session_state["_edit_now"] = True
    at.run()
    assert not at.exception
    assert position_of(edited_channel) == "Oz"

    at.run()  # a further rerun (new editor key, table rebuilt) must keep it
    assert position_of(edited_channel) == "Oz"


def test_electrode_model_edit_carries_its_electrode_type(tmp_path):
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    model = next(iter(ui.MODEL_TO_RUBRIK))
    harness = tmp_path / "h.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "st.session_state['general_form']['Device'] = 'UNICORN'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "rows = ui.render_channel_editor('UNICORN')\n"
        "if st.session_state.pop('_edit_now', False):\n"
        f"    st.session_state['_fake_editor'] = {{'edited_rows': {{0: {{'Model': {model!r}}}}}}}\n"
        "    ui._persist_channel_edits('UNICORN', '_fake_editor',\n"
        "                              [str(r['Channel']) for r in rows])\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()
    at.session_state["_edit_now"] = True
    at.run()
    assert not at.exception
    row = at.session_state["channel_tables"]["UNICORN"][0]
    assert row["Model"] == model
    assert row["Rubrik"] == ui.MODEL_TO_RUBRIK[model]


# --- The scalp map is not part of the electrode step anymore -----------------
def test_electrode_editor_has_no_scalp_map(tmp_path):
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "st.session_state['general_form']['Device'] = 'UNICORN'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "ui.render_channel_editor('UNICORN')\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()
    assert not at.exception
    assert not [c for c in at.caption if "Scalp map" in str(c.value)]
    assert not [b for b in at.button if "scalp map" in str(b.label).lower()]


# --- Live impedance: the table's own column, measured for ActiCHamp ----------
def test_impedance_column_for_a_device_that_does_not_measure_it(tmp_path):
    """Without a live reading, Impedance is one ordinary, hand-editable column of the table."""
    import json
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h_unicorn.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "st.session_state['general_form']['Device'] = 'UNICORN'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "ui.render_channel_editor('UNICORN')\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()
    assert not at.exception
    editors = at.get("arrow_data_frame")
    assert len(editors) == 1  # one combined table, nothing split off
    editor = editors[0]
    assert "Impedance" in list(editor.proto.column_order)
    assert json.loads(editor.proto.columns)["Impedance"].get("disabled") is not True


def test_nothing_refreshes_until_live_impedance_is_switched_on(tmp_path):
    """Off by default: no impedance table, and so nothing on the page refreshing.

    A timed refresh is a round trip that ends in a page rerun, and that rerun is what
    closes an open dropdown and loses the scroll position in the montage table -- Streamlit
    cannot refresh one element on its own. So the refresh only exists while it is asked
    for, and the montage table is otherwise the only thing on the page.
    """
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h_actichamp.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        # no amplifier in a test: the poll bails out instead of opening one
        "st.session_state['_impedance_monitor_failed'] = True\n"
        "st.session_state['general_form']['Device'] = 'ActiCHamp'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "ui.render_channel_editor('ActiCHamp')\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()
    assert not at.exception

    toggle = next(c for c in at.checkbox if c.key == "live_impedance_ActiCHamp")
    assert toggle.value is False
    tables = at.get("arrow_data_frame")
    assert len(tables) == 1  # the montage only; impedance is one of its columns
    assert list(tables[0].proto.column_order) == ["Channel", "Position", "Model", "Impedance"]

    toggle.set_value(True).run()
    assert not at.exception
    tables = at.get("arrow_data_frame")
    assert len(tables) == 2  # montage, and the impedance readout beside it

    montage, impedance = tables
    # Impedance leaves the montage table so the refresh has nothing to redraw there. The
    # underlying data still carries the field -- data_editor never strips columns it is
    # not displaying -- but nothing here lets the operator see or type into it.
    assert list(montage.proto.column_order) == ["Channel", "Position", "Model"]
    assert list(impedance.value.columns) == ["Channel", "Impedance"]
    assert list(impedance.value["Channel"])[:1] == ["GND"]


def test_no_live_impedance_offered_where_it_cannot_be_measured(tmp_path):
    # Nothing to switch on for a device that cannot read impedances, or under Simulate run
    # -- which promises no hardware is opened, and measuring holds the amplifier in
    # impedance mode.
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "device = st.session_state.get('_device', 'UNICORN')\n"
        "st.session_state['general_form']['Device'] = device\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "ui.render_channel_editor(device)\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)

    def has_toggle():
        return any(str(c.key or "").startswith("live_impedance_") for c in at.checkbox)

    at.run()
    assert not at.exception
    assert not has_toggle()  # UNICORN cannot measure it at all

    at.session_state["_device"] = "ActiCHamp"
    at.run()
    assert has_toggle()

    at.session_state["simulate_run_toggle"] = True
    at.run()
    assert not at.exception
    assert not has_toggle()


# --- One model list, labelled by category ------------------------------------
def test_the_model_dropdown_offers_the_whole_library_labelled_by_category(tmp_path):
    """Every electrode in electrodes.json, in one list, each showing its category.

    A data_editor column's options are fixed for the whole column, so a per-row list
    narrowed to that row's category is not possible. One list carrying the category in
    each label is: the categories stay in blocks, and nothing can get stuck showing the
    models of a category the operator is not working in.
    """
    import json
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "st.session_state['general_form']['Device'] = 'UNICORN'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "ui.render_channel_editor('UNICORN')\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()
    assert not at.exception

    editor = at.get("arrow_data_frame")[0]
    assert "Rubrik" not in list(editor.proto.column_order)  # the category is in the labels

    options = json.loads(editor.proto.columns)["Model"]["type_config"]["options"]
    every_model = {m for models in ui.ELECTRODE_LIBRARY.values() for m in models}
    # The stored value stays the plain model name; only the label carries the category.
    assert {o["value"] for o in options} == every_model
    by_value = {o["value"]: o["label"] for o in options}
    assert by_value["Grass, Gold Cup E5GH"] == "Wet Electrodes · Grass, Gold Cup E5GH"
    assert by_value["g.tec, g.SAHARA Dry Pin"] == "Dry Electrodes · g.tec, g.SAHARA Dry Pin"


def test_the_electrode_type_is_recorded_without_being_a_column(tmp_path):
    """No Electrode type column — but the category is still saved with the montage.

    Every entry in the Model dropdown already names its category, so a column repeating it
    earned nothing. It still has to reach the recording's metadata, which is what this
    guards: dropping the column must not drop the field.
    """
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "st.session_state['general_form']['Device'] = 'UNICORN'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "ui.render_channel_editor('UNICORN')\n"
        "eeg, extras = ui.build_channels('UNICORN')\n"
        "st.session_state['_built'] = eeg + extras\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()
    assert not at.exception

    assert "Rubrik" not in list(at.get("arrow_data_frame")[0].proto.column_order)
    built = at.session_state["_built"]
    assert built and all(entry["Rubrik"] in ui.ELECTRODE_RUBRICS for entry in built)
    stored = at.session_state["channel_tables"]["UNICORN"][0]
    assert stored["Rubrik"] == ui.MODEL_TO_RUBRIK[stored["Model"]]


def test_picking_a_model_from_another_category_moves_the_type_with_it(tmp_path):
    """Choosing a model is the whole interaction -- the type follows it, every time.

    This is what "pick the type, then the model" turned into: the list is ordered by
    category, so picking within a category is a scroll, and the type column then says
    which one you landed in. It must keep up even when the model comes from a category
    the row was not in before.
    """
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    wet_model = "Grass, Gold Cup E5GH"
    dry_model = "g.tec, g.SAHARA Dry Pin"
    assert ui.MODEL_TO_RUBRIK[wet_model] == "Wet Electrodes"
    assert ui.MODEL_TO_RUBRIK[dry_model] == "Dry Electrodes"

    harness = tmp_path / "h.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "st.session_state['general_form']['Device'] = 'UNICORN'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "rows = ui.render_channel_editor('UNICORN')\n"
        "order = [str(r['Channel']) for r in rows]\n"
        "pick = st.session_state.pop('_pick', None)\n"
        "if pick:\n"
        "    st.session_state['_fake'] = {'edited_rows': {0: {'Model': pick}}}\n"
        "    ui._persist_channel_edits('UNICORN', '_fake', order)\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()

    def row0():
        return at.session_state["channel_tables"]["UNICORN"][0]

    at.session_state["_pick"] = wet_model
    at.run()
    assert (row0()["Model"], row0()["Rubrik"]) == (wet_model, "Wet Electrodes")

    # and back out of that category again -- the delta still carries the earlier model,
    # which used to be enough to drag the type back with it
    at.session_state["_pick"] = dry_model
    at.run()
    assert not at.exception
    assert (row0()["Model"], row0()["Rubrik"]) == (dry_model, "Dry Electrodes")
    at.run()  # and it survives the next render
    assert (row0()["Model"], row0()["Rubrik"]) == (dry_model, "Dry Electrodes")


def test_an_uncategorised_model_leaves_the_type_as_it_stands(tmp_path):
    # "Other / not listed" is offered under every category, so it cannot name one. The
    # type keeps whatever it had rather than the code guessing.
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    harness = tmp_path / "h.py"
    harness.write_text(
        "import streamlit as st\n"
        "from cortipy.ui_streamlit import session as ui\n"
        "ui.ensure_state()\n"
        "st.session_state['general_form']['Device'] = 'UNICORN'\n"
        "st.session_state['general_form']['Method'] = 'SSVEP'\n"
        "rows = ui.render_channel_editor('UNICORN')\n"
        "order = [str(r['Channel']) for r in rows]\n"
        "if st.session_state.pop('_pick', False):\n"
        "    st.session_state['_fake'] = "
        "{'edited_rows': {0: {'Model': 'Other / not listed'}}}\n"
        "    ui._persist_channel_edits('UNICORN', '_fake', order)\n"
    )
    at = AppTest.from_file(str(harness), default_timeout=60)
    at.run()
    before = at.session_state["channel_tables"]["UNICORN"][0]["Rubrik"]
    at.session_state["_pick"] = True
    at.run()
    assert not at.exception
    row = at.session_state["channel_tables"]["UNICORN"][0]
    assert row["Model"] == "Other / not listed"
    assert row["Rubrik"] == before


# --- The "last read / connect device" line no longer appears ------------------
def test_no_last_read_status_line_for_a_live_or_manual_device(tmp_path):
    import numpy.testing  # noqa: F401 (macOS fork guard)
    from streamlit.testing.v1 import AppTest

    def captions_for(device):
        harness = tmp_path / f"h_{device}.py"
        harness.write_text(
            "import streamlit as st\n"
            "from cortipy.ui_streamlit import session as ui\n"
            "ui.ensure_state()\n"
            "st.session_state['_impedance_monitor_failed'] = True\n"
            f"st.session_state['general_form']['Device'] = {device!r}\n"
            "st.session_state['general_form']['Method'] = 'SSVEP'\n"
            f"ui.render_channel_editor({device!r})\n"
        )
        at = AppTest.from_file(str(harness), default_timeout=60)
        at.run()
        assert not at.exception
        return [str(c.value) for c in at.caption]

    for device in ("ActiCHamp", "UNICORN"):
        caps = captions_for(device)
        assert not any("last read" in c or "Connect the device" in c for c in caps)
