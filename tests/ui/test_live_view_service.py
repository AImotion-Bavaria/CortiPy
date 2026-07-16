from __future__ import annotations

import numpy as np

from cortipy.ui_streamlit import live


class Placeholder:
    def empty(self) -> None:
        pass


def test_live_view_keeps_moving_window_with_absolute_sample_offset(monkeypatch) -> None:
    calls = []

    def capture_plot(buffer, fs, placeholder, **kwargs):
        calls.append((buffer.copy(), fs, kwargs))

    monkeypatch.setattr(live, "_plot_live_buffer", capture_plot)

    service = live.LiveViewService(Placeholder(), window_seconds=1.0, channel_indices=[0])
    service.push(np.arange(6, dtype=float)[:, None], fs=4.0)

    assert service.samples_seen == 6
    np.testing.assert_array_equal(service.buffer[:, 0], np.array([2.0, 3.0, 4.0, 5.0]))
    assert calls[-1][2]["sample_offset"] == 2
    assert calls[-1][2]["window_seconds"] == 1.0


def test_live_view_renders_the_chosen_plot_type(monkeypatch) -> None:
    # The service renders one plot (the operator's choice) inline; FFT and per-channel are
    # now plot types, not separate windows.
    calls = []
    monkeypatch.setattr(live, "_plot_live_buffer", lambda *a, **k: calls.append(k))

    service = live.LiveViewService(
        Placeholder(), window_seconds=1.0, channel_indices=[0], plot_type=live.PLOT_OVERLAID
    )
    service.push(np.arange(4, dtype=float)[:, None], fs=4.0)
    service.mark_complete()

    assert calls, "the service should render the buffer"
    assert all(k["plot_type"] == live.PLOT_OVERLAID for k in calls)
    assert all(k["render_inline"] is True for k in calls)
