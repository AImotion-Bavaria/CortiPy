"""Connecting a device after loading a recording.

Loading any recording used to force the session into offline replay, which disabled the
"Connect device" button — so reusing a saved session's settings to record again was
impossible without restarting the app. Unchecking the replay box then crashed the page,
because `bool(ndarray)` raises for anything with more than one element.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("streamlit")

from cortipy.ui_streamlit.session import _requires_device_connection

LIVE = {"Device": "ActiCHamp", "Parameters": {"fs": 250}}
RECORDING = np.zeros((1000, 4))


class TestRequiresDeviceConnection:
    def test_connect_is_available_after_loading_a_recording(self):
        # The reported bug: this returned False, greying out Connect.
        assert _requires_device_connection(
            LIVE, simulate=False, use_imported_data=False, imported_data=RECORDING
        ) is True

    def test_replay_still_skips_the_device(self):
        assert _requires_device_connection(
            LIVE, simulate=False, use_imported_data=True, imported_data=RECORDING
        ) is False

    def test_offline_device_needs_no_connection(self):
        assert _requires_device_connection(
            {"Device": "Offline", "Parameters": {}},
            simulate=False, use_imported_data=False, imported_data=None,
        ) is False

    def test_simulation_needs_no_connection(self):
        assert _requires_device_connection(
            LIVE, simulate=True, use_imported_data=False, imported_data=None
        ) is False

    def test_multi_element_array_does_not_raise(self):
        # Guards the `bool(ndarray)` ValueError that took down the whole page.
        for data in (RECORDING, np.zeros((0, 0)), None):
            _requires_device_connection(
                LIVE, simulate=False, use_imported_data=False, imported_data=data
            )
