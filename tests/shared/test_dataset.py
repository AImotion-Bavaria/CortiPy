"""Tests for CortiDataset helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cortipy.shared.dataset import CortiDataset


def test_generate_eeg_samples_creates_synthetic_dataset():
    """generate_eeg_samples should synthesize realistic data into a CortiDataset."""
    ds = CortiDataset.generate_eeg_samples(
        sampling_rate=200.0,
        duration_s=1.5,
        channel_names=["Fp1", "Fp2", "Cz"],
        base_frequencies=(8.0, 12.0),
        noise=3.0,
        seed=7,
    )

    assert ds.data.shape == (int(200 * 1.5), 3)
    assert ds.raw.info["sfreq"] == pytest.approx(200.0)
    assert list(ds.channels["name"]) == ["Fp1", "Fp2", "Cz"]
    assert ds.metadata["generator"]["base_frequencies"] == [8.0, 12.0]
    assert ds.metadata["params"]["Parameters"]["RecordingTime"] == pytest.approx(1.5)
    assert np.var(ds.data, axis=0).min() > 0
