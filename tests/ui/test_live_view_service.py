from __future__ import annotations

import numpy as np

from cortipy.ui_streamlit import live


class Placeholder:
    def empty(self) -> None:
        pass

    def pyplot(self, *args, **kwargs) -> None:
        pass


def test_display_decimation_does_not_change_source() -> None:
    source = np.arange(1000, dtype=float).reshape(500, 2)
    time_axis = np.arange(500, dtype=float) / 500.0

    displayed, displayed_time = live._decimate_for_display(source, time_axis, 500.0)

    assert displayed.shape == (250, 2)
    np.testing.assert_array_equal(displayed, source[::2])
    np.testing.assert_array_equal(displayed_time, time_axis[::2])
    assert source.shape == (500, 2)


def test_display_keeps_full_rate_at_or_below_limit() -> None:
    source = np.ones((250, 2))
    time_axis = np.arange(250, dtype=float) / 250.0

    displayed, displayed_time = live._decimate_for_display(source, time_axis, 250.0)

    assert displayed is source
    assert displayed_time is time_axis


def test_vep_live_uses_selected_channel_and_trigger(monkeypatch) -> None:
    captured = {}

    def capture_figure(plot_type, buffer, time_axis, xlim, indices, labels, **kwargs):
        captured.update(buffer=buffer, time_axis=time_axis, indices=indices, labels=labels)
        return live.plt.figure()

    monkeypatch.setattr(live, "build_live_figure", capture_figure)
    raw = np.zeros((1000, 33), dtype=float)
    raw[:, 32] = np.tile([0.0, 1.0], 500)
    params = {
        "Method": "VEP",
        "Device": "ActiCHamp",
        "Channels": [{"Channel": f"Ch{i + 1}"} for i in range(32)],
        "Parameters": {
            "NumberEEGChannels": 32,
            "TriggerChannel": 33,
            "ReferenceChannel": 1,
            "LivePlotCH": 7,
        },
    }

    live._plot_live_buffer(
        raw,
        500.0,
        Placeholder(),
        channel_indices=list(range(32)),
        params=params,
        render_inline=True,
    )

    assert captured["indices"] == [6, 32]
    assert captured["labels"][-1] == "Trigger"
    assert captured["buffer"].shape[0] == 500
    np.testing.assert_array_equal(raw[:, 32], np.tile([0.0, 1.0], 500))


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


def test_live_view_skips_signal_frames_but_keeps_full_resolution_history(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(live, "_plot_live_buffer", lambda *a, **k: calls.append(k))

    service = live.LiveViewService(
        Placeholder(), window_seconds=1.0, channel_indices=[0], max_update_seconds=60.0
    )
    first = np.arange(10, dtype=float)[:, None]
    second = np.arange(10, 20, dtype=float)[:, None]
    service.push(first, fs=10.0)
    service.push(second, fs=10.0)

    assert len(calls) == 1
    np.testing.assert_array_equal(service.vep_buffer[:, 0], np.arange(20, dtype=float))
