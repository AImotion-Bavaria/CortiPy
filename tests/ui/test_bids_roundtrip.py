"""A BIDS export must be loadable again, not export-only.

CortiPy stashes its full params under the sidecar's CortiPyParameters key, so its own BIDS
exports round-trip losslessly; a third-party BIDS dataset still loads with a minimal,
reconstructed config (data + channels + sampling rate).
"""
from __future__ import annotations

import numpy as np

from cortipy.ui_streamlit import session as ui


def _params():
    chans = [
        {"Channel": f"Ch {i + 1}", "Position": nm, "Impedance": 5.0 + i,
         "Rubrik": "Dry Electrodes", "Model": "g.tec g.SAHARA", "Active": True}
        for i, nm in enumerate(["Fz", "Cz", "Pz", "Oz"])
    ]
    return {
        "Method": "SSVEP", "Device": "UNICORN",
        "Parameters": {"fs": 250, "NumberEEGChannels": 4, "StimFreq": 12.0, "Filename": "subj01"},
        "Channels": chans, "Metadata": {"Participant": {"Code": "P01"}},
    }


def test_bids_export_round_trips_losslessly(tmp_path):
    params = _params()
    data = (np.random.default_rng(3).standard_normal((500, 4)) * 20.0)
    ui.export_recording(data, params, tmp_path, "BIDS", "Parquet")

    loaded, arr, msg = ui._load_dataset_folder(tmp_path)
    assert loaded is not None, msg
    assert loaded.get("Method") == "SSVEP"
    assert loaded["Parameters"].get("StimFreq") == 12.0
    assert len(loaded.get("Channels", [])) == 4
    assert loaded["Channels"][0].get("Position") == "Fz"
    assert loaded["Channels"][0].get("Impedance") == 5.0  # montage/impedance preserved
    assert arr is not None and arr.shape == (500, 4)
    assert np.allclose(arr, data, rtol=1e-4, atol=abs(data).mean() * 1e-3)


def test_a_bids_export_folder_counts_as_a_session(tmp_path):
    ui.export_recording(np.zeros((100, 4)), _params(), tmp_path, "BIDS", "Parquet")
    assert ui._folder_has_session(tmp_path) is True
    ui.list_saved_sessions.clear()
    # a run folder containing a bids_export is discoverable as a session
    assert tmp_path in ui.list_saved_sessions(tmp_path.parent)


def test_third_party_bids_without_embedded_params_loads_minimally(tmp_path):
    from cortipy.shared.bids import BIDSLoader

    root = tmp_path / "bids_export"
    BIDSLoader(root).to_bids(
        (np.random.default_rng(0).standard_normal((300, 3)) * 15.0),
        sampling_rate=200.0, ch_names=["A1", "A2", "A3"],
        subject="09", task="rest", run="01", format="parquet", overwrite=True,
        dataset_description={"Name": "SomeExternalDataset"},  # no CortiPyParameters
    )
    loaded, arr, msg = ui._load_dataset_folder(tmp_path)
    assert loaded is not None, msg
    assert loaded["Parameters"]["fs"] == 200.0
    assert loaded["Parameters"]["NumberEEGChannels"] == 3
    assert [c["Position"] for c in loaded["Channels"]] == ["A1", "A2", "A3"]
    assert arr is not None and arr.shape == (300, 3)
