"""Hidden navigation entries, device-specific field gating, and impedance mapping."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from cortipy.ui_streamlit import session
from cortipy.ui_streamlit.constants import (
    DEVICE_ONLY_FIELDS,
    HIDDEN_VIEWS,
    VIEW_OPTIONS,
    field_applies_to_device,
)
from cortipy.ui_streamlit.electrodes import map_impedances_to_channels


class TestHiddenViews:
    def test_hidden_views_are_not_navigable(self):
        for view in HIDDEN_VIEWS:
            assert view not in session.VISIBLE_VIEW_OPTIONS

    def test_hidden_views_still_exist(self):
        # Hidden, not deleted: the pages and their render branches remain.
        for view in HIDDEN_VIEWS:
            assert view in VIEW_OPTIONS
            assert view in session.APP_VIEW_OPTIONS

    def test_landing_page_is_a_visible_view(self):
        assert session.DEFAULT_VIEW in session.VISIBLE_VIEW_OPTIONS

    def test_the_rest_stay_navigable(self):
        for view in ("Session configuration", "Electrodes", "Live preview", "Charts", "Saved sessions"):
            assert view in session.VISIBLE_VIEW_OPTIONS


class TestDeviceFieldGating:
    @pytest.mark.parametrize("field", ["NumberAUXChannels", "TriggerChannel"])
    def test_actichamp_keeps_its_hardware_fields(self, field):
        assert field_applies_to_device(field, "ActiCHamp") is True

    @pytest.mark.parametrize("device", ["UNICORN", "Dummy", "LSL", "Offline"])
    @pytest.mark.parametrize("field", ["NumberAUXChannels", "TriggerChannel"])
    def test_devices_without_aux_or_trigger_hide_those_fields(self, device, field):
        # UNICORN's extra columns are accelerometer/gyro/battery/counter, not a trigger,
        # and none of these devices expose AUX inputs.
        assert field_applies_to_device(field, device) is False

    def test_unlisted_fields_always_apply(self):
        assert field_applies_to_device("NumberEEGChannels", "UNICORN") is True
        assert field_applies_to_device("ReferenceChannel", "Dummy") is True

    def test_gating_is_declarative(self):
        assert set(DEVICE_ONLY_FIELDS) == {"NumberAUXChannels", "TriggerChannel"}


class TestImpedanceMapping:
    # Amplifier reports ohms as |GND|REF|CH1|CH2|... (AmplifierSDK.h)
    VALUES = [5000.0, 4000.0, 12000.0, 13000.0, 14000.0]

    def test_actichamp_rows_get_their_impedances(self):
        rows = [{"Channel": "GND"}] + [{"Channel": f"Ch {i}"} for i in (1, 2, 3)]
        out = map_impedances_to_channels(rows, self.VALUES)
        assert [r["Impedance"] for r in out] == [5.0, 12.0, 13.0, 14.0]

    def test_reference_row_is_matched_case_insensitively(self):
        # The row is labelled "Ref"; the SDK calls it "REF". An exact match left it unset.
        rows = [{"Channel": "GND"}, {"Channel": "Ref"}, {"Channel": "Ch 1"}]
        out = map_impedances_to_channels(rows, self.VALUES)
        assert out[1]["Impedance"] == 4.0

    def test_negative_values_are_treated_as_unavailable(self):
        rows = [{"Channel": "GND"}, {"Channel": "Ch 1", "Impedance": 7.0}]
        out = map_impedances_to_channels(rows, [5000.0, 4000.0, -1.0])
        assert out[1]["Impedance"] == 7.0  # left untouched

    def test_too_few_values_are_ignored(self):
        rows = [{"Channel": "GND", "Impedance": 1.0}]
        assert map_impedances_to_channels(rows, [1.0, 2.0])[0]["Impedance"] == 1.0
