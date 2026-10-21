"""Unit tests for VEP trigger-locked segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from cortipy.shared.segmentation import seg_sig_fast, segment_at_triggers


def _triggered(n_samples: int, positions: list[int]) -> np.ndarray:
    """Samples x 2 array: a ramp EEG channel plus a trigger_adc-style impulse channel."""
    data = np.zeros((n_samples, 2))
    data[:, 0] = np.arange(n_samples)
    data[positions, 1] = 1.0
    return data


def test_epochs_start_on_the_trigger_sample():
    """t = 0 is the reversal itself, so the first epoch sample is the trigger sample."""
    segments = segment_at_triggers(_triggered(30, [2, 12]), fs=1000.0, max_time=0.005, trigger_channel=1)
    assert segments.shape == (2, 5, 2)
    np.testing.assert_array_equal(segments[:, 0, 0], [2.0, 12.0])
    np.testing.assert_array_equal(segments[:, 0, 1], [1.0, 1.0])


def test_complete_last_epoch_is_kept_and_incomplete_one_dropped():
    """Unlike seg_sig_fast, a complete final epoch is averaged; only a truncated one is dropped."""
    data = _triggered(20, [2, 12])
    assert segment_at_triggers(data, 1000.0, 0.008, 1).shape[0] == 2
    assert seg_sig_fast(data, 1000.0, 0.008, 1).shape[0] == 1
    assert segment_at_triggers(data, 1000.0, 0.009, 1).shape[0] == 1


def test_seg_sig_fast_starts_one_sample_later():
    """Documents the 1-sample offset segment_at_triggers removes for VEP."""
    data = _triggered(40, [2, 12, 22])
    sliced = segment_at_triggers(data, 1000.0, 0.005, 1)
    looped = seg_sig_fast(data, 1000.0, 0.005, 1)
    np.testing.assert_array_equal(looped[:, :-1, 0], sliced[: looped.shape[0], 1:, 0])


def test_no_triggers_returns_empty_stack():
    segments = segment_at_triggers(np.zeros((10, 3)), 1000.0, 0.004, 2)
    assert segments.shape == (0, 4, 3)


def test_invalid_trigger_channel_raises():
    with pytest.raises(IndexError):
        segment_at_triggers(np.zeros((10, 2)), 1000.0, 0.004, 2)
