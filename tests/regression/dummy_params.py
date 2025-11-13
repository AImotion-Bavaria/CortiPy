"""Deterministic dummy Params structures shared between regression tests."""

from __future__ import annotations

import numpy as np


def base_params(method: str, device: str, fs: float) -> dict:
    return {
        "Method": method,
        "Device": device,
        "Parameters": {"fs": fs},
        "ReportAnalyzer": True,
        "Channels": [{"Position": f"Ch{i+1}"} for i in range(64)],
    }


def dummy_params_alpha() -> dict:
    rng = np.random.default_rng(42)
    fs = 100
    duration = 20
    t = np.arange(0, duration, 1 / fs)
    signals = np.column_stack(
        [
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 12 * t + 0.4),
            0.5 * np.sin(2 * np.pi * 8 * t + 0.9),
            0.3 * np.sin(2 * np.pi * 15 * t),
        ]
    )
    params = base_params("Alpha", "ActiCHamp", fs)
    p = params["Parameters"]
    p.update(
        {
            "NumberEEGChannels": 4,
            "ReferenceChannel": 1,
            "TriggerChannel": 5,
            "Trigger": "Fixed",
            "TriggerTime": 2,
            "Start": "eyesClosed",
            "RecordingTime": duration,
        }
    )
    params["data"] = np.column_stack([signals, np.zeros_like(t)])
    return params


def dummy_params_vep() -> dict:
    rng = np.random.default_rng(7)
    fs = 200
    duration = 5
    samples = int(fs * duration)
    t = np.arange(samples) / fs
    trigger = (np.floor(t / 0.5) % 2 == 0).astype(float)
    signals = np.column_stack(
        [
            0.2 * np.sin(2 * np.pi * 6 * t),
            0.3 * np.sin(2 * np.pi * 12 * t + 0.5),
            0.1 * rng.standard_normal(samples),
            0.15 * rng.standard_normal(samples),
        ]
    )
    params = base_params("VEP", "ActiCHamp", fs)
    p = params["Parameters"]
    p.update(
        {
            "ReferenceChannel": 1,
            "TriggerChannel": signals.shape[1] + 1,
            "LivePlotCH": 2,
            "NumberAUXChannels": 0,
        }
    )
    params["data"] = np.column_stack([signals, trigger])
    return params


def dummy_params_ssvep() -> dict:
    rng = np.random.default_rng(99)
    fs = 250
    duration = 4
    t = np.arange(0, duration, 1 / fs)
    base_signal = np.sin(2 * np.pi * 12 * t)
    signals = np.tile(base_signal[:, None], (1, 8)) + 0.05 * rng.standard_normal((len(t), 8))
    params = base_params("SSVEP", "ActiCHamp", fs)
    p = params["Parameters"]
    p.update(
        {
            "ReferenceChannel": 1,
            "NumberAUXChannels": 0,
            "StimFreq": [8.6, 12, 10, 14],
            "NumberEEGChannels": 8,
        }
    )
    params["data"] = signals
    return params


def dummy_params_bera() -> dict:
    rng = np.random.default_rng(123)
    fs = 2000
    duration = 1
    samples = int(fs * duration)
    t = np.arange(samples) / fs
    trigger = (np.floor(t / 0.02) % 2 == 0).astype(float)
    signal = 0.2 * np.sin(2 * np.pi * 100 * t) + 0.05 * rng.standard_normal(samples)
    signals = np.column_stack([signal, 0.3 * np.sin(2 * np.pi * 120 * t), trigger])
    params = base_params("BERA", "ActiCHamp", fs)
    p = params["Parameters"]
    p.update(
        {
            "ReferenceChannel": 1,
            "TriggerChannel": signals.shape[1],
            "LivePlotCH": 2,
            "NumberAUXChannels": 0,
        }
    )
    params["data"] = signals
    return params


def dummy_params_p300() -> dict:
    rng = np.random.default_rng(321)
    fs = 256
    duration = 10
    t = np.arange(0, duration, 1 / fs)
    trigger = (np.floor(t / 1.0) % 2 == 0).astype(float)
    signals = np.column_stack(
        [
            0.1 * rng.standard_normal(len(t)),
            0.2 * np.sin(2 * np.pi * 5 * t),
            0.25 * np.sin(2 * np.pi * 7 * t),
        ]
    )
    params = base_params("P300", "ActiCHamp", fs)
    p = params["Parameters"]
    p.update(
        {
            "ReferenceChannel": 1,
            "TriggerChannel": signals.shape[1] + 1,
            "LivePlotCH": 2,
            "NumberAUXChannels": 0,
            "RecordingTime": duration,
        }
    )
    params["data"] = np.column_stack([signals, trigger])
    return params


def dummy_params_assr() -> dict:
    rng = np.random.default_rng(55)
    fs = 500
    duration = 5
    t = np.arange(0, duration, 1 / fs)
    ipsi = np.sin(2 * np.pi * 40 * t) + 0.01 * rng.standard_normal(len(t))
    contra = 0.5 * np.sin(2 * np.pi * 40 * t + 0.5) + 0.01 * rng.standard_normal(len(t))
    ref = 0.2 * np.sin(2 * np.pi * 5 * t)
    params = base_params("ASSR", "ActiCHamp", fs)
    p = params["Parameters"]
    p.update(
        {
            "ReferenceChannel": 3,
            "ChannelIpsi": 1,
            "ChannelContra": 2,
            "ASSRModulationFrequency": 40,
            "ASSRCarrierFrequency": 500,
        }
    )
    params["data"] = np.column_stack([ipsi, contra, ref])
    return params
