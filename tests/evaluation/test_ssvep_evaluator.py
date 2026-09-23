import numpy as np

from cortipy.evaluation.ssvep import SsvepEvaluator, _compute_ssvep_psd


def test_ssvep_reference_keeps_channels_when_number_eeg_channels_is_zero():
    params = {
        "Device": "ActiCHamp",
        "Parameters": {"NumberEEGChannels": 0, "ReferenceChannel": 1},
    }
    samples = 1000
    fs = 1000.0
    t = np.arange(samples) / fs
    data = np.column_stack(
        [
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 12 * t),
        ]
    )

    referenced = SsvepEvaluator()._apply_reference(params, data)
    psd = _compute_ssvep_psd(referenced, fs)

    assert referenced.shape == data.shape
    np.testing.assert_allclose(referenced[:, 0], 0.0, atol=1e-12)
    assert psd.psd.shape[0] == data.shape[1]
    assert psd.freq.size > 0


def _ssvep_params(positions, stim_hz=10.0, samples=2000, fs=250.0):
    t = np.arange(samples) / fs
    # Put the stimulus response only on the LAST channel, so picking the wrong column shows up.
    data = np.column_stack(
        [np.random.default_rng(i).standard_normal(samples) * 0.1 for i in range(len(positions))]
    )
    data[:, -1] += 10.0 * np.sin(2 * np.pi * stim_hz * t)
    return {
        "Method": "SSVEP",
        "Device": "Dummy",
        "ReportAnalyzer": True,  # no plotting
        "Parameters": {
            "fs": fs,
            "NumberEEGChannels": len(positions),
            "StimFreq": stim_hz,
            "PlotChannelLabel": "Oz",
        },
        "Channels": [
            {"Channel": f"Ch {i + 1}", "Position": pos, "Active": True}
            for i, pos in enumerate(positions)
        ],
        "data": data,
    }


def _evaluate(params):
    from cortipy.core.context import ModuleContext

    ctx = ModuleContext(params)
    SsvepEvaluator().evaluate(ctx)
    return ctx.params


def test_power_plot_selects_oz_when_present():
    # Regression: the label lookup stringified the Channels dicts, so it never matched and
    # the PSD always came from column 0 while still being titled "Oz".
    params = _evaluate(_ssvep_params(["Fp1", "Cz", "Oz"]))
    block = params["Parameters"]
    assert block["PlotChannelLabel"] == "Oz"
    assert block["ChannelIpsi"] == 3  # 1-based: Oz is the third column
    assert "PlotChannelFallback" not in params["Evaluation"]


def test_power_plot_reports_the_channel_it_actually_used_when_oz_is_absent():
    # The UNICORN montage has no Oz. The plot must not keep claiming "Oz".
    params = _evaluate(_ssvep_params(["Fp1", "Fp2", "O1"]))
    block = params["Parameters"]
    assert block["PlotChannelLabel"] == "O1"
    assert params["Evaluation"]["PlotChannelFallback"] == {"requested": "Oz", "used": "O1"}


def test_stim_freq_is_honoured_rather_than_defaulting_to_10hz():
    params = _evaluate(_ssvep_params(["Oz"], stim_hz=12.0))
    assert params["Parameters"]["StimFreq"] == 12.0


def test_power_plot_never_lands_on_the_reference_channel():
    # The reference channel is identically zero after referencing. The old fallback always
    # picked column 0 -- which IS the reference by default -- so the PSD was a flat line.
    params = _ssvep_params(["Fp1", "Fp2", "C3"])          # no Oz -> must fall back
    params["Parameters"]["ReferenceChannel"] = 1          # Fp1 == column 0
    out = _evaluate(params)

    assert out["Parameters"]["ChannelIpsi"] != 1, "plotted the reference channel"
    assert out["Parameters"]["PlotChannelLabel"] != "Fp1"

    # And the plotted channel actually carries signal.
    psd = out["Evaluation"]["PSD"]["dBpsdx"]
    plotted = psd[out["Parameters"]["ChannelIpsi"] - 1]
    assert np.ptp(plotted) > 1.0, "plotted channel is flat -> still the reference"
