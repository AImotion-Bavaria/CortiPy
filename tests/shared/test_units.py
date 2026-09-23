"""Signal units at the MNE boundary.

CortiPy works in microvolts; MNE's contract is volts. Every RawArray used to be built
straight from the uV array, storing a 50 uV signal as 50 V and pushing a 1e6 error into
every BIDS/EDF/Parquet export.
"""

from __future__ import annotations

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortipy.shared.bids import (  # noqa: E402  (must follow importorskip)
    BIDSLoader,
    _coerce_to_raw_array,
    raw_to_microvolts,
)
from cortipy.shared.units import (  # noqa: E402
    looks_like_microvolts,
    to_microvolts,
    to_volts,
    unit_hint_from_params,
)

FS = 250
PEAK_UV = 50.0


def signal(n: int = 500) -> tuple[np.ndarray, list[str], list[str]]:
    t = np.arange(n) / FS
    eeg = PEAK_UV * np.sin(2 * np.pi * 10 * t)
    trigger = np.zeros(n)
    trigger[::100] = 1.0
    data = np.column_stack([eeg, eeg * 0.5, trigger])
    return data, ["Oz", "Pz", "STI"], ["eeg", "eeg", "stim"]


class TestScaling:
    def test_to_volts_scales_only_voltage_channels(self):
        data = np.array([[50.0, 50.0], [1.0, 1.0]])  # (channels, samples)
        out = to_volts(data, ["eeg", "stim"])
        assert out[0] == pytest.approx([5e-5, 5e-5])
        assert out[1] == pytest.approx([1.0, 1.0])  # trigger must survive intact

    def test_round_trip_is_lossless(self):
        data = np.array([[50.0, -12.5], [1.0, 0.0]])
        back = to_microvolts(to_volts(data, ["eeg", "stim"]), ["eeg", "stim"])
        assert back == pytest.approx(data)


class TestRawArrayBoundary:
    def test_mne_receives_si_volts(self):
        data, names, types = signal()
        raw = _coerce_to_raw_array(data, FS, names, types)
        eeg = raw.get_data()[0]
        # 50 uV == 5e-5 V. Before the fix this was 50.0.
        assert np.abs(eeg).max() == pytest.approx(PEAK_UV * 1e-6, rel=0.01)

    def test_trigger_channel_is_not_scaled(self):
        data, names, types = signal()
        raw = _coerce_to_raw_array(data, FS, names, types)
        assert np.abs(raw.get_data()[2]).max() == pytest.approx(1.0)

    def test_raw_to_microvolts_recovers_the_input(self):
        data, names, types = signal()
        raw = _coerce_to_raw_array(data, FS, names, types)
        assert raw_to_microvolts(raw) == pytest.approx(data, abs=1e-9)


class TestLegacyGuard:
    def test_microvolt_magnitudes_are_recognised(self):
        # A pre-fix export: uV numbers sitting in a volt-typed container.
        legacy = np.array([[50.0, -50.0]])
        assert looks_like_microvolts(legacy, ["eeg"]) is True

    def test_genuine_volt_data_is_not_flagged(self):
        proper = np.array([[5e-5, -5e-5]])
        assert looks_like_microvolts(proper, ["eeg"]) is False

    def test_legacy_file_is_not_scaled_a_second_time(self):
        # Simulate a Raw built the old (wrong) way, straight from uV.
        data, names, types = signal()
        info = mne.create_info(ch_names=names, sfreq=FS, ch_types=types)
        legacy_raw = mne.io.RawArray(data.T, info, verbose="ERROR")
        recovered = raw_to_microvolts(legacy_raw)
        # Must come back as ~50 uV, not 5e7.
        assert np.abs(recovered[:, 0]).max() == pytest.approx(PEAK_UV, rel=0.01)


@pytest.mark.parametrize("value,expected", [("uV", "uV"), ("V", "V"), ("volts", "V"), (None, None)])
def test_unit_hint_from_params(value, expected):
    params = {"Parameters": {"SignalUnit": value}} if value else {"Parameters": {}}
    assert unit_hint_from_params(params) == expected


@pytest.mark.parametrize("fmt", ["fif", "parquet"])
def test_export_reload_preserves_microvolts(tmp_path, fmt):
    data, names, types = signal()
    loader = BIDSLoader(tmp_path / fmt)
    loader.to_bids(
        data, FS, ch_names=names, channel_types=types,
        subject="01", task="alpha", format=fmt, overwrite=True,
    )
    reloaded = loader.read_bids().data
    assert np.abs(reloaded[:, 0]).max() == pytest.approx(PEAK_UV, rel=0.05)
