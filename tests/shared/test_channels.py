"""Channel label <-> data-column mapping.

Regression cover for the bug where every evaluator carried its own label lookup: the SSVEP
one stringified the ``Channels`` dicts, so it never matched and the PSD plot silently drew
column 0 while still titling itself "Oz".
"""

from __future__ import annotations

import pytest

from cortipy.shared.channels import (
    channel_labels,
    resolve_channel_index,
    resolve_plot_channel,
)


def montage(*positions: str) -> dict:
    return {
        "Channels": [
            {"Channel": f"Ch {i + 1}", "Position": pos, "Active": True}
            for i, pos in enumerate(positions)
        ]
    }


class TestResolveChannelIndex:
    def test_resolves_anatomical_position(self):
        params = montage("Fp1", "Cz", "Oz")
        assert resolve_channel_index(params, "Oz") == 2

    def test_resolves_channel_name(self):
        params = montage("Fp1", "Cz", "Oz")
        assert resolve_channel_index(params, "Ch 2") == 1

    @pytest.mark.parametrize("label", ["oz", "OZ", " Oz "])
    def test_is_case_and_space_insensitive(self, label):
        assert resolve_channel_index(montage("Fp1", "Oz"), label) == 1

    def test_numeric_label_is_one_based(self):
        assert resolve_channel_index(montage("Fp1", "Oz"), 2) == 1

    def test_absent_label_returns_none_rather_than_guessing(self):
        # The old lookups silently returned channel 0 here.
        assert resolve_channel_index(montage("Fp1", "Cz"), "Oz") is None

    def test_accepts_a_bare_channel_list(self):
        chans = montage("Fp1", "Oz")["Channels"]
        assert resolve_channel_index(chans, "Oz") == 1


class TestResolvePlotChannel:
    def test_exact_match_reports_exact(self):
        idx, label, exact = resolve_plot_channel(montage("Fp1", "Oz"), "Oz", 2)
        assert (idx, label, exact) == (1, "Oz", True)

    def test_falls_back_to_nearest_occipital_and_says_so(self):
        # UNICORN has no Oz. The fallback must be reported, not silently titled "Oz".
        idx, label, exact = resolve_plot_channel(montage("Fp1", "Fp2", "O1"), "Oz", 3)
        assert exact is False
        assert label == "O1"
        assert idx == 2

    def test_last_resort_is_first_channel_but_never_claims_exact(self):
        idx, label, exact = resolve_plot_channel(montage("Fp1", "Fp2"), "Oz", 2)
        assert (idx, label, exact) == (0, "Fp1", False)

    def test_never_points_past_the_acquired_columns(self):
        # Oz is in the montage but outside the 2 columns actually recorded.
        idx, _label, exact = resolve_plot_channel(montage("Fp1", "Cz", "Oz"), "Oz", count=2)
        assert idx < 2
        assert exact is False


def test_channel_labels_pad_beyond_the_montage():
    assert channel_labels(montage("Fp1", "Oz"), 4) == ["Fp1", "Oz", "Ch 3", "Ch 4"]
