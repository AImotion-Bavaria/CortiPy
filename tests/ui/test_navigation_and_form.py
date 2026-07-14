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


class TestStagedConfiguration:
    """Each step reveals the next only when its own required fields are filled."""

    def base(self, **general):
        g = {"Device": "", "Method": "", "fs": "", "RecordingTime": 0}
        g.update(general)
        return g

    def status(self, general, device_values=None, method_values=None):
        return session._step_status(general, device_values or {}, method_values or {})

    def test_nothing_is_done_on_a_blank_form(self):
        s = self.status(self.base())
        assert s["Device"] is False
        assert s["Method"] is False
        assert s["Acquisition"] is False
        assert s["Channels"] is False

    def test_device_without_a_required_port_is_connected_immediately(self):
        # ActiCHamp has no port to enter, so its connection step must not block.
        s = self.status(self.base(Device="ActiCHamp"))
        assert s["Device"] is True
        assert s["Connection"] is True

    def test_unicorn_blocks_until_a_port_is_given(self):
        g = self.base(Device="UNICORN")
        assert self.status(g, device_values={})["Connection"] is False
        assert self.status(g, device_values={"UNICORNPort": "COM7"})["Connection"] is True

    def test_simulating_relaxes_the_port_requirement(self, monkeypatch):
        # A simulated run never touches hardware, so a missing port must not wall off the form.
        monkeypatch.setattr(session, "_is_simulating", lambda: True)
        assert self.status(self.base(Device="UNICORN"), device_values={})["Connection"] is True

    def test_acquisition_needs_a_rate_and_a_nonzero_duration(self):
        assert self.status(self.base(fs="250", RecordingTime=0))["Acquisition"] is False
        assert self.status(self.base(fs="", RecordingTime=5))["Acquisition"] is False
        assert self.status(self.base(fs="250", RecordingTime=5))["Acquisition"] is True

    def test_channels_step_needs_a_positive_channel_count(self):
        assert self.status(self.base(), method_values={"NumberEEGChannels": 0})["Channels"] is False
        assert self.status(self.base(), method_values={"NumberEEGChannels": 8})["Channels"] is True

    def test_session_details_never_blocks(self):
        assert self.status(self.base())["Session details"] is True

    def test_first_incomplete_walks_the_steps_in_order(self):
        assert session._first_incomplete(self.status(self.base())) == "Device"
        assert session._first_incomplete(self.status(self.base(Device="ActiCHamp"))) == "Method"
        g = self.base(Device="ActiCHamp", Method="Alpha")
        assert session._first_incomplete(self.status(g)) == "Acquisition"
        g = self.base(Device="ActiCHamp", Method="Alpha", fs="250", RecordingTime=5)
        assert session._first_incomplete(self.status(g)) == "Channels"
        done = self.status(g, method_values={"NumberEEGChannels": 8})
        assert session._first_incomplete(done) is None

    def test_step_names_are_ordered(self):
        assert session.CONFIG_STEPS == (
            "Device", "Connection", "Method", "Acquisition", "Channels", "Session details",
        )


class TestRequiredStimulusFrequencies:
    """A stimulus frequency left at 0 must be refused, not silently replaced.

    SSVEP's evaluator substitutes 10 Hz when StimFreq is missing or zero, so a run
    configured with the schema default produced metrics against a frequency nobody chose.
    """

    def test_ssvep_rejects_a_zero_stim_freq(self):
        issues = session.missing_method_frequencies("SSVEP", {"StimFreq": 0})
        assert issues and "StimFreq" in issues[0]

    def test_ssvep_rejects_a_missing_stim_freq(self):
        assert session.missing_method_frequencies("SSVEP", {}) != []

    def test_ssvep_accepts_a_real_stim_freq(self):
        assert session.missing_method_frequencies("SSVEP", {"StimFreq": 12.0}) == []

    def test_assr_needs_both_carrier_and_modulation(self):
        both_zero = session.missing_method_frequencies(
            "ASSR", {"ASSRCarrierFrequency": 0, "ASSRModulationFrequency": 0}
        )
        assert len(both_zero) == 2
        half = session.missing_method_frequencies(
            "ASSR", {"ASSRCarrierFrequency": 1000, "ASSRModulationFrequency": 0}
        )
        assert len(half) == 1 and "Modulation" in half[0]
        ok = session.missing_method_frequencies(
            "ASSR", {"ASSRCarrierFrequency": 1000, "ASSRModulationFrequency": 40}
        )
        assert ok == []

    @pytest.mark.parametrize("method", ["Alpha", "P300"])
    def test_methods_without_a_stimulus_frequency_are_unaffected(self, method):
        assert session.missing_method_frequencies(method, {}) == []

    def test_validation_blocks_the_run(self):
        params = {
            "Method": "SSVEP",
            "Device": "Dummy",
            "Parameters": {"fs": 250, "NumberEEGChannels": 8, "StimFreq": 0},
            "Channels": [{"Channel": "Ch 1", "Active": True}],
        }
        assert any("StimFreq" in issue for issue in session.validate_params(params))


class TestConfigProgressSteps:
    """The top progress bar renders from the form's own step status, so it cannot lie.

    It used to keep its own checklist and ticked "Sampling rate" and "Electrodes" on a
    blank form: fs carries a schema default and the channel table is pre-populated, so both
    were true by construction and the bar claimed 2/4 before anything had been chosen.
    """

    def steps(self, monkeypatch, status):
        monkeypatch.setattr(session, "config_step_status", lambda: status)
        return session.config_progress_steps()

    def blank(self):
        return {name: False for name in session.CONFIG_STEPS}

    def test_a_blank_form_reports_nothing_done(self, monkeypatch):
        steps = self.steps(monkeypatch, self.blank())
        assert [name for name, _ in steps] == list(session.CONFIG_STEPS)
        assert not any(done for _, done in steps)

    def test_later_steps_do_not_tick_before_earlier_ones(self, monkeypatch):
        # "Session details" requires nothing of its own; it must still not tick early.
        status = self.blank()
        status["Session details"] = True
        steps = dict(self.steps(monkeypatch, status))
        assert steps["Session details"] is False

    def test_steps_tick_in_order(self, monkeypatch):
        status = self.blank()
        status.update({"Device": True, "Connection": True, "Method": True})
        steps = dict(self.steps(monkeypatch, status))
        assert [steps[n] for n in session.CONFIG_STEPS] == [True, True, True, False, False, False]

    def test_everything_done_ticks_everything(self, monkeypatch):
        status = {name: True for name in session.CONFIG_STEPS}
        steps = self.steps(monkeypatch, status)
        assert all(done for _, done in steps)
