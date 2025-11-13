"""Unit tests for AssrModule."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from cortipy.modules.assr import AssrModule


@pytest.fixture
def assr_context(module_context_factory):
    def _factory(**overrides):
        params = {
            "Method": "ASSR",
            "Device": "ActiCHamp",
            "Parameters": {"RecordingTime": 12.0, "NumberAUXChannels": 1},
        }
        params.update(overrides)
        return module_context_factory(**params)

    return _factory


def test_collect_measurements_chunks_recording_time(assr_context, fake_device, spy_factory):
    """ASSR run should buffer consecutive acquisitions until the requested recording time elapses. It reassures clinicians that the run duration they dial in is exactly what gets recorded and plotted."""
    module = AssrModule()
    context = assr_context()

    chunks = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
        np.array([[9.0, 10.0]]),
    ]
    fake_device.acquire_returns = deque(chunks)
    context.device = fake_device

    start_spy = spy_factory("cortipy.modules.assr.info_start_live")
    end_spy = spy_factory("cortipy.modules.assr.info_end_live")

    module.collect_measurements(context)

    expected = np.vstack(chunks)
    np.testing.assert_array_equal(context.data_buffer, expected)
    np.testing.assert_array_equal(context.params["data"], expected)

    assert fake_device.acquire_calls == [(5.0, 1), (5.0, 1), (2.0, 1)]
    assert len(start_spy.calls) == 1
    assert len(end_spy.calls) == 1


def test_collect_measurements_sets_aux_zero_for_non_actichamp(module_context_factory, fake_device):
    """Non-ActiCHamp devices must request zero AUX channels to mimic field practice. This keeps the acquisition API from asking for channels the hardware cannot provide."""
    module = AssrModule()
    context = module_context_factory(
        Method="ASSR",
        Device="UNICORN",
        Parameters={"RecordingTime": 5.0},
    )
    fake_device.acquire_returns = deque([np.array([1.0, 2.0, 3.0])])
    context.device = fake_device

    module.collect_measurements(context)

    assert fake_device.acquire_calls == [(5.0, 0)]
    assert context.data_buffer.shape == (3, 1)
