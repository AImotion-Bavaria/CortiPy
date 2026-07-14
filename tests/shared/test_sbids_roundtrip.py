"""SBIDS JSON-LD round-trip and integrity.

Covers the reported round-trip losses: Device came back as a hardcoded "Offline", StimFreq
was never written at all (so a reloaded SSVEP silently used the evaluator's 10 Hz default),
GND/Ref were counted as EEG channels, and a "sub-" prefix leaked from the export filename
into the graph and then doubled on re-import.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cortipy.shared.sbids import SbidsExporter, verify_sbids_checksums

pytest.importorskip("streamlit")
from cortipy.ui_streamlit.data_import import params_from_jsonld_doc  # noqa: E402

METHODS = ["Alpha", "SSVEP", "VEP", "ASSR", "BERA", "P300"]
LIBRARY = {"Standard": ["Model A"]}


@pytest.fixture
def exported(tmp_path):
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir()
    raw = raw_dir / "ssvep_run1.npz"
    np.savez_compressed(raw, data=np.random.default_rng(0).standard_normal((500, 3)))

    params = {
        "Method": "SSVEP",
        "Device": "UNICORN",
        "Parameters": {
            "fs": 250,
            "RecordingTime": 2,
            "NumberEEGChannels": 3,
            "NumberAUXChannels": 0,
            "StimFreq": 12.0,
            "ReferenceChannel": 2,
        },
        "Channels": [
            {"Channel": f"Ch {i + 1}", "Position": pos, "Active": True}
            for i, pos in enumerate(["O1", "O2", "Pz"])
        ],
        "ReferenceElectrodes": [{"Channel": "GND", "Position": "GND", "Active": True}],
        "Metadata": {"Participant": {"Code": "01"}},
    }

    exporter = SbidsExporter(dataset_id="DEMO", dataset_name="demo")
    exporter.add_recording_from_cortipy_json(
        params, str(raw), content_url="raw_data/ssvep_run1.npz"
    )
    out = tmp_path / "sbids_meta_01.jsonld"
    exporter.save(str(out))
    return out, raw, params


def reimport(path):
    return params_from_jsonld_doc(
        json.loads(path.read_text()),
        path.name,
        method_names=METHODS,
        electrode_library=LIBRARY,
    )


class TestGraph:
    def test_no_sub_prefix_anywhere(self, exported):
        out, _raw, _params = exported
        assert "sub-" not in out.read_text()

    def test_content_url_uses_forward_slashes(self, exported):
        out, _raw, _params = exported
        doc = json.loads(out.read_text())
        urls = [
            n["schema:contentUrl"]
            for n in doc["@graph"]
            if "schema:contentUrl" in n
        ]
        assert urls and all("\\" not in u for u in urls)

    def test_gnd_is_typed_as_aux_not_eeg(self, exported):
        out, _raw, _params = exported
        doc = json.loads(out.read_text())
        rec = next(n for n in doc["@graph"] if "schema:CreateAction" in (n.get("@type") or []))
        by_name = {n["schema:name"]: n["@type"] for n in rec["schema:variableMeasured"]}
        assert "AUXChannel" in by_name["GND"]
        assert "EEGChannel" in by_name["O1"]


class TestRoundTrip:
    def test_device_survives(self, exported):
        # Was hardcoded to "Offline", which also disabled the Connect button.
        assert reimport(exported[0])["Device"] == "UNICORN"

    def test_stim_freq_survives(self, exported):
        # Was dropped entirely -> SSVEP silently evaluated against 10.0 Hz.
        assert reimport(exported[0])["Parameters"]["StimFreq"] == pytest.approx(12.0)

    def test_gnd_does_not_inflate_the_eeg_channel_count(self, exported):
        back = reimport(exported[0])
        assert back["Parameters"]["NumberEEGChannels"] == 3
        assert [c["Position"] for c in back["Channels"]] == ["O1", "O2", "Pz"]
        assert [c["Position"] for c in back["ReferenceElectrodes"]] == ["GND"]

    def test_filename_carries_no_bids_prefix(self, exported):
        assert "sub-" not in reimport(exported[0])["Parameters"]["Filename"]

    def test_participant_code_is_bare(self, exported):
        assert reimport(exported[0])["Metadata"]["Participant"]["Code"] == "01"

    def test_reexport_does_not_double_the_subject(self, exported, tmp_path):
        back = reimport(exported[0])
        exporter = SbidsExporter(dataset_id="D2", dataset_name="d2")
        exporter.add_recording_from_cortipy_json(back, str(exported[1]))
        again = tmp_path / "again.jsonld"
        exporter.save(str(again))
        assert "sub-" not in again.read_text()


class TestChecksums:
    def test_digest_is_written(self, exported):
        out, _raw, _params = exported
        doc = json.loads(out.read_text())
        file_node = next(
            n for n in doc["@graph"] if "schema:DigitalDocument" in (n.get("@type") or [])
        )
        assert len(file_node["sha256"]) == 64

    def test_intact_file_verifies(self, exported):
        out, _raw, _params = exported
        assert [r["status"] for r in verify_sbids_checksums(out)] == ["ok"]

    def test_tampered_file_is_detected(self, exported):
        out, raw, _params = exported
        raw.write_bytes(raw.read_bytes() + b"tampered")
        assert [r["status"] for r in verify_sbids_checksums(out)] == ["mismatch"]

    def test_missing_file_is_reported(self, exported):
        out, raw, _params = exported
        raw.unlink()
        assert [r["status"] for r in verify_sbids_checksums(out)] == ["missing"]
