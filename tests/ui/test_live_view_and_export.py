"""Live-preview rendering/refresh, and export runs that do not overwrite each other."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("streamlit")

from cortipy.ui_streamlit.live import (  # noqa: E402
    DEFAULT_LIVE_SCALE,
    LIVE_SCALE_OPTIONS,
    LIVE_WINDOW_SECONDS,
    _channel_limits,
    _stacked_channel_figure,
    resolve_live_scale,
)
from cortipy.ui_streamlit.plot_windows import image_window_html  # noqa: E402
from cortipy.ui_streamlit.session import export_recording, next_run_index  # noqa: E402


class TestStreamingWindowRefreshes:
    """The pop-out window is rewritten on every chunk, but nothing told the open tab to
    reload it — so the "live" preview was a dead screenshot."""

    def figure(self):
        fig = matplotlib.figure.Figure()
        fig.add_subplot(111).plot([0, 1], [0, 1])
        return fig

    def test_a_streaming_window_carries_a_meta_refresh(self):
        html = image_window_html(self.figure(), "Live preview", auto_refresh=True, refresh_seconds=0.75)
        assert "http-equiv='refresh'" in html
        assert "0.75" in html

    def test_a_final_window_does_not_refresh(self):
        html = image_window_html(self.figure(), "Result", auto_refresh=False)
        assert "http-equiv='refresh'" not in html

    def test_the_refresh_interval_never_hits_zero(self):
        # A 0s meta-refresh would spin the browser.
        html = image_window_html(self.figure(), "x", auto_refresh=True, refresh_seconds=0.0)
        assert "0.20" in html


class TestScaling:
    def test_auto_is_the_default(self):
        assert DEFAULT_LIVE_SCALE == "Auto"
        assert resolve_live_scale(None) is None
        assert resolve_live_scale("Auto") is None

    def test_fixed_ranges_resolve_to_microvolts(self):
        assert resolve_live_scale("± 100 µV") == 100.0
        assert resolve_live_scale("± 25 µV") == 25.0

    def test_several_sensible_eeg_ranges_are_offered(self):
        fixed = [v for v in LIVE_SCALE_OPTIONS.values() if v]
        assert len(fixed) >= 5
        assert sorted(fixed) == fixed  # ascending, so the dropdown reads sensibly

    def test_an_unknown_label_falls_back_to_auto(self):
        assert resolve_live_scale("nonsense") is None

    def test_a_fixed_scale_is_symmetric_and_identical_for_every_channel(self):
        quiet = np.array([1.0, -1.0])
        loud = np.array([400.0, -400.0])
        assert _channel_limits(quiet, 100.0) == (-100.0, 100.0)
        assert _channel_limits(loud, 100.0) == (-100.0, 100.0)

    def test_auto_fits_each_channel_to_its_own_data(self):
        lo, hi = _channel_limits(np.array([-10.0, 20.0]), None)
        assert lo < -10.0 and hi > 20.0  # padded around the data

    def test_a_flat_channel_still_gets_a_usable_range(self):
        lo, hi = _channel_limits(np.zeros(10), None)
        assert hi > lo

    def test_all_nan_does_not_explode(self):
        lo, hi = _channel_limits(np.full(5, np.nan), None)
        assert hi > lo


class TestStackedFigure:
    def figure(self, n_ch=4, scale=None):
        fs, secs = 250, 10
        t = np.arange(fs * secs) / fs
        data = np.column_stack([np.sin(2 * np.pi * 10 * t) * (10 * (i + 1)) for i in range(n_ch)])
        labels = ["Fp1", "Fp2", "C3", "C4"][:n_ch]
        return _stacked_channel_figure(
            data, t, (0, secs), list(range(n_ch)), labels, fs=fs, scale=scale
        )

    def test_one_lane_per_channel(self):
        assert len(self.figure(n_ch=4).axes) == 4

    def test_each_lane_is_labelled_with_its_channel_and_min_max(self):
        texts = [t.get_text() for ax in self.figure(n_ch=2).axes for t in ax.texts]
        assert "Fp1" in texts and "Fp2" in texts
        assert any("min" in t and "max" in t and "µV" in t for t in texts)

    def test_a_fixed_scale_gives_every_lane_the_same_axis(self):
        # This is the point of a fixed scale: channels become comparable by eye.
        ylims = [ax.get_ylim() for ax in self.figure(n_ch=4, scale=100.0).axes]
        assert all(lim == (-100.0, 100.0) for lim in ylims)

    def test_auto_scale_gives_lanes_different_axes(self):
        ylims = [ax.get_ylim() for ax in self.figure(n_ch=4, scale=None).axes]
        assert len(set(ylims)) > 1

    def test_the_window_is_ten_seconds(self):
        assert LIVE_WINDOW_SECONDS == 10.0
        assert self.figure().axes[0].get_xlim() == (0.0, 10.0)

    def test_a_single_channel_still_renders(self):
        assert len(self.figure(n_ch=1).axes) == 1


class TestExportDoesNotOverwrite:
    """Every export used to land on the same filename, so a second recording into a dataset
    folder silently destroyed the first."""

    PARAMS = {
        "Method": "SSVEP",
        "Device": "UNICORN",
        "Parameters": {"fs": 250, "NumberEEGChannels": 2, "StimFreq": 12.0},
        "Channels": [{"Channel": "Ch 1", "Position": "O1"}, {"Channel": "Ch 2", "Position": "O2"}],
        "Metadata": {"Participant": {"Code": "01"}},
    }

    def data(self):
        return np.random.default_rng(0).standard_normal((300, 2)) * 20

    def test_repeated_exports_produce_separate_runs(self, tmp_path):
        for _ in range(3):
            export_recording(self.data(), self.PARAMS, tmp_path, "JSON-LD", "parquet")
        docs = sorted(p.name for p in (tmp_path / "jsonld_export").glob("*.jsonld"))
        assert docs == [
            "sbids_meta_01_ssvep_run-01.jsonld",
            "sbids_meta_01_ssvep_run-02.jsonld",
            "sbids_meta_01_ssvep_run-03.jsonld",
        ]

    def test_the_raw_payloads_are_separate_too(self, tmp_path):
        # A unique metadata file pointing at a shared raw file would still lose data.
        for _ in range(2):
            export_recording(self.data(), self.PARAMS, tmp_path, "JSON-LD", "parquet")
        raws = sorted(p.name for p in (tmp_path / "jsonld_export" / "raw_data").glob("*"))
        assert len(raws) == 2
        assert len(set(raws)) == 2

    def test_run_index_starts_at_one_in_an_empty_folder(self, tmp_path):
        assert next_run_index(tmp_path, "01_ssvep") == 1

    def test_run_index_is_scoped_per_stem(self, tmp_path):
        export_recording(self.data(), self.PARAMS, tmp_path, "JSON-LD", "parquet")
        root = tmp_path / "jsonld_export"
        assert next_run_index(root, "01_ssvep") == 2
        assert next_run_index(root, "02_ssvep") == 1  # different subject
        assert next_run_index(root, "01_alpha") == 1  # different task

    def test_a_typed_filename_is_honoured(self, tmp_path):
        params = dict(self.PARAMS)
        params["Parameters"] = dict(params["Parameters"], Filename="pilot subject A!")
        export_recording(self.data(), params, tmp_path, "JSON-LD", "parquet")
        export_recording(self.data(), params, tmp_path, "JSON-LD", "parquet")
        docs = sorted(p.name for p in (tmp_path / "jsonld_export").glob("*.jsonld"))
        # Sanitised (spaces -> _, '!' dropped) and run-indexed, not the default sub_task name.
        assert docs == ["sbids_meta_pilot_subject_A_run-01.jsonld", "sbids_meta_pilot_subject_A_run-02.jsonld"]


class TestLivePlotTypes:
    """A selectable plot for the live view, available for every method."""

    def data(self, n_ch=4):
        fs, secs = 100, 10
        t = np.arange(fs * secs) / fs
        return np.column_stack([np.sin(2 * np.pi * (5 + i) * t) * 10 for i in range(n_ch)]), t

    def test_all_four_types_are_offered(self):
        from cortipy.ui_streamlit.live import LIVE_PLOT_TYPES

        assert set(LIVE_PLOT_TYPES) == {
            "Stacked per-channel", "Overlaid signal", "FFT / spectrum", "Single channel"
        }

    def test_stacked_gives_one_axis_per_channel(self):
        from cortipy.ui_streamlit.live import build_live_figure, PLOT_STACKED

        data, t = self.data(4)
        fig = build_live_figure(PLOT_STACKED, data, t, (0, 10), [0, 1, 2, 3], ["a", "b", "c", "d"], fs=100)
        assert len(fig.axes) == 4

    def test_overlaid_and_fft_and_single_use_one_axis(self):
        from cortipy.ui_streamlit.live import (
            build_live_figure, PLOT_OVERLAID, PLOT_FFT, PLOT_SINGLE,
        )

        data, t = self.data(4)
        for ptype in (PLOT_OVERLAID, PLOT_FFT, PLOT_SINGLE):
            fig = build_live_figure(ptype, data, t, (0, 10), [0, 1, 2, 3], ["a", "b", "c", "d"], fs=100)
            assert len(fig.axes) == 1, ptype

    def test_fft_x_axis_stops_at_nyquist(self):
        from cortipy.ui_streamlit.live import build_live_figure, PLOT_FFT

        data, t = self.data(2)
        fig = build_live_figure(PLOT_FFT, data, t, (0, 10), [0, 1], ["a", "b"], fs=100)
        assert fig.axes[0].get_xlim()[1] == 50.0  # fs/2

    def test_single_channel_plots_the_first_selected(self):
        from cortipy.ui_streamlit.live import build_live_figure, PLOT_SINGLE

        data, t = self.data(3)
        fig = build_live_figure(PLOT_SINGLE, data, t, (0, 10), [2], ["a", "b", "Oz"], fs=100)
        assert "Oz" in fig.axes[0].get_title(loc="left")

    def test_an_unknown_type_falls_back_to_stacked(self):
        from cortipy.ui_streamlit.live import build_live_figure

        data, t = self.data(2)
        fig = build_live_figure("nonsense", data, t, (0, 10), [0, 1], ["a", "b"], fs=100)
        assert len(fig.axes) == 2  # stacked
