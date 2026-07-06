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
