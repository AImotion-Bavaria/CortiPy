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
    monkeypatch.setattr(live, "_plot_fft_spectrum", lambda *args, **kwargs: None)

    service = live.LiveViewService(Placeholder(), window_seconds=1.0, channel_indices=[0])
    service.push(np.arange(6, dtype=float)[:, None], fs=4.0)

    assert service.samples_seen == 6
    np.testing.assert_array_equal(service.buffer[:, 0], np.array([2.0, 3.0, 4.0, 5.0]))
    assert calls[-1][2]["sample_offset"] == 2
    assert calls[-1][2]["window_seconds"] == 1.0


def test_live_view_final_channel_windows_are_opt_in(monkeypatch) -> None:
    individual_calls = []

    monkeypatch.setattr(live, "_plot_live_buffer", lambda *args, **kwargs: None)
    monkeypatch.setattr(live, "_plot_fft_spectrum", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        live,
        "_plot_individual_channels",
        lambda *args, **kwargs: individual_calls.append((args, kwargs)),
    )

    service = live.LiveViewService(Placeholder(), window_seconds=1.0, channel_indices=[0], final_channel_windows=True)
    service.push(np.arange(4, dtype=float)[:, None], fs=4.0)
    service.mark_complete()

    assert len(individual_calls) == 1
    assert individual_calls[0][1]["sample_offset"] == 0
