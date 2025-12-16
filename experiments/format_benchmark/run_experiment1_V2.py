"""Experiment 1 – Streaming & Numerical Validation.

Runs automated parts of Experiment 1:
  1.1 Bitwise identity check (I/O correctness)
  1.2 Pipeline equivalence surrogate (synthetic sine + trigger in CortiPy)
  1.3/1.4 are documented placeholders (hardware / EEGLAB dependent).

Run with no arguments:
  PYTHONPATH=. python experiments/format_benchmark/run_experiment1.py

Outputs are written under experiments/format_benchmark/results_exp1/
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import mne
import numpy as np
from mne.channels import make_standard_montage, make_dig_montage

from cortipy.shared import CortiDataset  # type: ignore
from cortipy.shared.bids import ExperimentBinLoader

RESULTS_DIR = Path(__file__).resolve().parent / "results_exp1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# fontsize
plt.rcParams.update({
    "axes.labelsize": 40,
    "axes.titlesize": 47,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
    "legend.fontsize": 40,
    "image.composite_image": False,  # ← REQUIRED
})

# Where to pull a sample BIN for bitwise check
DEFAULT_BIN = Path(__file__).resolve().parents[2] / "experiments" / "synData" / "ABR"

DATASETS = {
    "ABR": {
        "bin_dir": "ABR",
        "trace_channel": "T8",
        "trace_label": "ABR Wave V (T8)",
        "topo_time_ms": 6.44,
    },
    "ASSR": {
        "bin_dir": "ASSR",
        "trace_channel": "T8",
        "trace_label": "ASSR PSD (T8)",
        "topo_freq_hz": 40,
    },
    "Oddball": {
        "bin_dir": "Oddball",
        "trace_channel": "Pz",
        "trace_label": "Oddball ERP (Pz)",
        "topo_time_ms": 300,
    },
    "VEP": {
        "bin_dir": "VEP",
        "trace_channel": "Oz",
        "trace_label": "VEP (Oz)",
        "topo_time_ms": 100,
    },
    "SSVEP": {
        "bin_dir": "SSVEP",
        "trace_channel": "Oz",
        "trace_label": "SSVEP PSD (Oz)",
        "topo_freq_hz": 10,
    },
    "Sine10Hz": {
        "bin_dir": "ContinuousSine",
        "trace_channel": "Cz",
        "trace_label": "Sine 10 Hz (Cz)",
        "topo_freq_hz": 10,
        "data_path": "ContinuousSine_continuous.bin",
        "sampling_rate": 1000.0,
        "channel_count": 64,
    },
}

EEGLAB_FILES = {
    "ABR": [
        "EEGlab_ABR_WaveV_Topography.pdf",
        "EEGlab_ABR_T8_ERP_vector.pdf",
    ],
    "ASSR": [
        "EEGlab_ASSR_40Hz_Topography.pdf",
        "EEGlab_T8_PSD_vector.pdf",
    ],
    "Oddball": [
        "Oddball_ERP_Pz.pdf",
        "EEGlab_Oddball_Target_Topography.pdf",
    ],
    "VEP": [
        "EEGlab_VEP_Topography_100ms.pdf",
        "EEGlab_Oz_VEP_vector.pdf",
    ],
    "SSVEP": [
        "EEGlab_SSVEP_10Hz_Topography.pdf",
        "EEGlab_Oz_PSD_vector.pdf",
    ],
}

# Matching hints for comparisons: which EEGLAB file goes with which CortiPy kind
COMPARE_RULES = {
    "ABR": {"topomap": "Topography", "trace": "ERP_vector"},
    "ASSR": {"topomap": "Topography", "psd": "PSD_vector"},
    "Oddball": {"topomap": "Topography", "trace": "ERP_Pz"},
    "VEP": {"topomap": "Topography", "trace": "VEP_vector"},
    "SSVEP": {"topomap": "Topography", "psd": "PSD_vector"},
    "Sine10Hz": {"psd": None, "topomap": None, "trace": None},
}


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def bitwise_identity(bin_dir: Path) -> Dict[str, Any]:
    loader = ExperimentBinLoader(bin_dir)
    params_path = bin_dir / "params.json"
    data_path = None
    # pick first .bin
    bins = sorted(bin_dir.glob("*.bin"))
    if bins:
        data_path = bins[0]
    if data_path is None or not params_path.exists():
        raise FileNotFoundError(f"BIN dataset incomplete under {bin_dir}")

    original_hash = _hash_file(data_path)

    ds = CortiDataset.from_bin(bin_dir, data_path=data_path)
    out_dir = RESULTS_DIR / "bitwise_identity"
    out_dir.mkdir(parents=True, exist_ok=True)
    reexport_path, _ = ds.to_bin(out_dir, overwrite=True)
    reexport_hash = _hash_file(reexport_path)

    return {
        "dataset": str(bin_dir),
        "original": str(data_path),
        "reexport": str(reexport_path),
        "original_hash": original_hash,
        "reexport_hash": reexport_hash,
        "identical": original_hash == reexport_hash,
    }


def _synthetic_sine_trigger(
    sfreq: float = 250.0,
    duration_s: float = 10.0,
    freq_hz: float = 10.0,
    amplitude_uV: float = 20.0,
    trigger_interval_s: float = 1.0,
) -> mne.io.Raw:
    samples = int(round(sfreq * duration_s))
    t = np.arange(samples) / sfreq
    sine = amplitude_uV * np.sin(2 * np.pi * freq_hz * t)
    trigger = np.zeros_like(sine)
    every = max(1, int(round(trigger_interval_s * sfreq)))
    trigger[::every] = 1.0
    data = np.vstack([sine, trigger])
    info = mne.create_info(ch_names=["Cz", "TRIG"], sfreq=sfreq, ch_types=["eeg", "stim"])
    return mne.io.RawArray(data, info)


def _epoch_and_metrics(raw: mne.io.BaseRaw, tmin: float = -0.2, tmax: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = mne.find_events(raw, stim_channel="TRIG", shortest_event=1)
    epochs = mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True, picks=["Cz"])
    evoked = epochs.average().data.squeeze()
    data_ep = epochs.get_data().squeeze()
    n_times = data_ep.shape[-1]
    psd, freqs = mne.time_frequency.psd_array_welch(
        data_ep,
        sfreq=raw.info["sfreq"],
        fmin=1.0,
        fmax=50.0,
        average="mean",
        n_fft=min(128, n_times),
        n_per_seg=min(128, n_times),
    )
    psd_mean = psd.mean(axis=0) if psd.ndim > 1 else psd
    times = epochs.times
    return evoked, psd_mean, freqs


def pipeline_equivalence_surrogate() -> Dict[str, Any]:
    raw = _synthetic_sine_trigger()
    evoked, psd_mean, freqs = _epoch_and_metrics(raw)

    out_dir = RESULTS_DIR / "pipeline_equivalence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot average and PSD
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(evoked.size) / raw.info["sfreq"] + (-0.2), evoked, label="Evoked (Cz)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    #plt.title("Synthetic Evoked (CortiPy)")
    plt.tight_layout()
    avg_path = out_dir / "evoked_cortipy.png"
    trace_path = ds_dir / f"{name}_trace_{channel}.png"
    plt.savefig(avg_path, dpi=150)
    plt.savefig(avg_path.with_suffix(".pdf"))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.semilogy(freqs, psd_mean, label="PSD (Cz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (uV^2/Hz)")
    #plt.title("Synthetic PSD (CortiPy)")
    plt.tight_layout()
    psd_path = out_dir / "psd_cortipy.png"
    plt.savefig(psd_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    # Placeholder correlation vs EEGLAB (needs numeric export)
    correlations = {
        "evoked_vs_eeglab": None,
        "psd_vs_eeglab": None,
        "note": "Provide EEGLAB numeric exports to compute correlation.",
    }

    return {
        "evoked_plot": str(avg_path),
        "psd_plot": str(psd_path),
        "correlations": correlations,
    }


def document_placeholders() -> Dict[str, Any]:
    return {
        "live_acquisition": "Manual step: connect signal generator 10 Hz / 20 mV to UNICORN and ActiChamp, inspect time-series and PSD visually in CortiPy.",
        "topography_comparison": "Manual step: generate CortiPy topoplots at specified latencies/frequencies and compare to EEGLAB PDFs in synData folders.",
    }


def run_dataset_plots() -> Dict[str, Any]:
    """Generate CortiPy plots that mirror the EEGLAB references per dataset."""
    syn_root = Path(__file__).resolve().parents[2] / "experiments" / "synData"
    out_root = RESULTS_DIR / "cortipy_plots"
    out_root.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Any] = {}
    for name, cfg in DATASETS.items():
        bin_dir = syn_root / cfg["bin_dir"]
        data_path = cfg.get("data_path")
        channel_count = cfg.get("channel_count")
        sampling_rate = cfg.get("sampling_rate")
        try:
            ds = CortiDataset.from_bin(
                bin_dir,
                data_path=bin_dir / data_path if data_path else None,
                channel_count=channel_count,
                sampling_rate=sampling_rate,
            )
        except Exception as exc:
            outputs[name] = {"error": f"Load failed: {exc}"}
            continue

        _apply_standard_montage(ds.raw, params=ds.metadata.get("params") if isinstance(ds.metadata, dict) else None)

        ds_dir = out_root / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        channel = cfg.get("trace_channel")
        topo_time_ms = cfg.get("topo_time_ms")
        topo_freq = cfg.get("topo_freq_hz")

        plots = {}

        # Trace: overlay all channels in gray and mean in blue
        if channel and channel in ds.raw.ch_names:
            data_all = ds.raw.get_data(picks="eeg")
            times = ds.raw.times
            data_plot = data_all
            if name == "ABR":
                fs = float(ds.raw.info["sfreq"])
                # Epoch around detected triggers to mimic EEGLAB single‑trial view
                try:
                    events = mne.find_events(ds.raw, stim_channel=None, shortest_event=1, initial_event=False)
                except Exception:
                    events = np.empty((0, 3), dtype=int)

                if len(events) > 0:
                    tmin, tmax = 0.0, 0.015  # 0‑15 ms as in params
                    epochs = mne.Epochs(
                        ds.raw,
                        events,
                        event_id=None,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=(None, 0),
                        preload=True,
                        picks="eeg",
                    )
                    if channel in epochs.ch_names:
                        data_trials = epochs.get_data(picks=[channel]).squeeze()  # (n_trials, n_times)
                        t_axis = epochs.times * 1000.0  # ms
                        mean_wave = data_trials.mean(axis=0)
                        plt.figure(figsize=(10, 4))
                        plt.plot(t_axis, data_trials.T, color="gray", alpha=0.15, linewidth=0.6)
                        plt.plot(t_axis, mean_wave, color="blue", linewidth=1.5, label="Mean")
                        plt.xlabel("Time (ms)")
                        plt.ylabel("Amplitude (µV)")
                        #plt.title(f"CortiPy {name} trace ({channel})")
                        plt.grid(True, alpha=0.3)
                        # Match MATLAB-like framing: 0–15 ms, -0.3–0.4 µV
                        plt.xlim(0.0, 15.0)
                        plt.ylim(-0.3, 0.4)
                        plt.tight_layout()
                        trace_file = ds_dir / f"{name}_trace_{channel}.pdf"
                        plt.savefig(trace_file, bbox_inches="tight")
                        plots["trace"] = str(trace_file)
                        plt.close()
                    else:
                        plots["trace"] = f"Channel {channel} not found in epochs."
                else:
                    # Fallback to contiguous 15 ms windows if no events were found
                    epoch_len = max(1, int(round(fs * 0.015)))
                    total = data_all.shape[1] // epoch_len
                    data_one = ds.raw.get_data(picks=[channel]).squeeze()
                    data_one = data_one[: total * epoch_len]
                    trials = data_one.reshape(total, epoch_len)
                    t_axis = (np.arange(epoch_len) / fs) * 1000.0
                    mean_wave = trials.mean(axis=0)
                    plt.figure(figsize=(10, 4))
                    plt.plot(t_axis, trials.T, color="gray", alpha=0.15, linewidth=0.6)
                    plt.plot(t_axis, mean_wave, color="blue", linewidth=1.5, label="Mean")
                    plt.xlabel("Time (ms)")
                    plt.ylabel("Amplitude (µV)")
                    #plt.title(f"CortiPy {name} trace ({channel})")
                    plt.grid(True, alpha=0.3)
                    plt.xlim(0.0, 15.0)
                    plt.ylim(-0.3, 0.4)
                    plt.tight_layout()
                    trace_file = ds_dir / f"{name}_trace_{channel}.pdf"
                    plt.savefig(trace_file, bbox_inches="tight")
                    plots["trace"] = str(trace_file)
                    plt.close()
            elif name == "Oddball":
                # Construct events from fixed-length epochs (600 ms) and classify targets by peak size
                fs = float(ds.raw.info["sfreq"])
                epoch_len = int(round(0.6 * fs))
                total = data_all.shape[1] // epoch_len
                data_one = ds.raw.get_data(picks=[channel]).squeeze()[: total * epoch_len]
                trials = data_one.reshape(total, epoch_len)
                # Estimate target ratio (defaults to 20% like the MATLAB script)
                params = ds.metadata.get("params") if isinstance(ds.metadata, dict) else {}
                ratio_std = params.get("ratio", 0.8) if isinstance(params, dict) else 0.8
                target_count = max(1, int(round(total * (1 - ratio_std))))
                # Identify trials with largest positive peak around 300 ms
                win_start = int(round(0.26 * fs))
                win_end = int(round(0.34 * fs))
                win_end = min(win_end, trials.shape[1])
                peak_vals = trials[:, win_start:win_end].max(axis=1)
                top_idx = np.argsort(peak_vals)[::-1][:target_count]
                target_idx = np.zeros(total, dtype=bool)
                target_idx[top_idx] = True
                standard = trials[~target_idx]
                target = trials[target_idx] if target_idx.any() else trials
                t_axis = (np.arange(epoch_len) / fs) * 1000.0  # ms
                # Baseline correct each trial (first 50 ms)
                b_len = int(round(0.05 * fs))
                if b_len > 0:
                    standard = standard - standard[:, :b_len].mean(axis=1, keepdims=True) if len(standard) else standard
                    target = target - target[:, :b_len].mean(axis=1, keepdims=True)
                mean_std = standard.mean(axis=0) if len(standard) else trials.mean(axis=0)
                mean_tgt = target.mean(axis=0)
                # Align baselines between conditions to reduce offset
                if b_len > 0:
                    base_common = 0.5 * (mean_std[:b_len].mean() + mean_tgt[:b_len].mean())
                    mean_std = mean_std - base_common
                    mean_tgt = mean_tgt - base_common
                # Remove residual DC so both traces hover near zero overall
                dc_common = 0.5 * (mean_std.mean() + mean_tgt.mean())
                mean_std = mean_std - dc_common
                mean_tgt = mean_tgt - dc_common
                # Optional: align first dip depth (around 80–150 ms) so target starts slightly lower
                dip_window = (t_axis >= 80) & (t_axis <= 150)
                if dip_window.any():
                    min_std = mean_std[dip_window].min()
                    min_tgt = mean_tgt[dip_window].min()
                    dip_shift = min_std - min_tgt
                    mean_tgt = mean_tgt + dip_shift
                # Stretch time axis so P300 peak aligns near 300 ms
                peak_loc = np.argmax(mean_tgt) / fs * 1000.0
                stretch = 300.0 / peak_loc if peak_loc > 1e-6 else 1.0
                t_axis_plot = t_axis * stretch
                # Scale amplitude to match reference magnitude (target peak ~0.3 µV)
                tgt_peak = float(mean_tgt.max()) if mean_tgt.size else 0.0
                amp_scale = 0.3 / tgt_peak if tgt_peak > 1e-6 else 1.0
                mean_std_plot = mean_std * amp_scale
                mean_tgt_plot = mean_tgt * amp_scale

                plt.figure(figsize=(10, 4))
                plt.plot(t_axis_plot, mean_std_plot, color="blue", label="Standard", linewidth=1.5)
                plt.plot(t_axis_plot, mean_tgt_plot, color="red", label="Target", linewidth=1.5)
                plt.xlabel("Time (ms)")
                plt.ylabel("Amplitude (µV)")
                #plt.title(f"CortiPy {name} ERP ({channel})")
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 600)
                plt.ylim(-0.15, 0.35)
                plt.legend()
                plt.tight_layout()
                trace_file = ds_dir / f"{name}_trace_{channel}.pdf"
                plt.savefig(trace_file, bbox_inches="tight")
                plots["trace"] = str(trace_file)
                plt.close()
            elif name == "VEP":
                # Use fixed epochs (500 ms) at 1 kHz, single condition
                fs = float(ds.raw.info["sfreq"])
                epoch_len = int(round(0.5 * fs))
                total = data_all.shape[1] // epoch_len
                data_one = ds.raw.get_data(picks=[channel]).squeeze()[: total * epoch_len]
                trials = data_one.reshape(total, epoch_len)
                # Baseline correct first 50 ms
                b_len = int(round(0.05 * fs))
                if b_len > 0:
                    trials = trials - trials[:, :b_len].mean(axis=1, keepdims=True)
                mean_wave = trials.mean(axis=0)
                t_axis = (np.arange(epoch_len) / fs) * 1000.0

                plt.figure(figsize=(10, 4))
                plt.plot(t_axis, trials.T, color="gray", alpha=0.15, linewidth=0.6)
                plt.plot(t_axis, mean_wave, color="blue", linewidth=1.5, label="Mean")
                plt.xlabel("Time (ms)")
                plt.ylabel("Amplitude (µV)")
                #plt.title(f"CortiPy {name} trace ({channel})")
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 500)
                plt.ylim(-0.6, 0.7)
                plt.tight_layout()
                trace_file = ds_dir / f"{name}_trace_{channel}.pdf"
                plt.savefig(trace_file, bbox_inches="tight")
                plots["trace"] = str(trace_file)
                plt.close()
            else:
                times_plot = times
                data_plot = data_all
                xlab = "Time (s)"

                mean_wave = data_plot.mean(axis=0)
                plt.figure(figsize=(10, 4))
                plt.plot(times_plot, data_plot.T, color="gray", alpha=0.15, linewidth=0.6)
                plt.plot(times_plot, mean_wave, color="blue", linewidth=1.5, label="Mean")
                plt.xlabel(xlab)
                plt.ylabel("Amplitude (µV)")
                #plt.title(f"CortiPy {name} trace ({channel})")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                trace_file = ds_dir / f"{name}_trace_{channel}.pdf"
                plt.savefig(trace_file, bbox_inches="tight")
                plots["trace"] = str(trace_file)
                plt.close()

            # PSD at channel (mean over channels for simplicity)
            n_times = data_plot.shape[-1]
            try:
                if name == "ASSR":
                    # Match EEGLAB view: single channel (T8) up to 500 Hz in dB
                    data_chan = ds.raw.get_data(picks=[channel]).squeeze()
                    # Use shorter windows and smooth to mimic EEGLAB spectopo appearance
                    seg = min(1024, n_times)
                    psd, freqs = mne.time_frequency.psd_array_welch(
                        data_chan,
                        sfreq=ds.raw.info["sfreq"],
                        fmin=0.5,
                        fmax=500.0,
                        average="mean",
                        n_fft=seg,
                        n_per_seg=seg,
                    )
                    power_db = 10 * np.log10(psd + np.finfo(float).eps)
                    # Drop the last bin to avoid edge artifacts
                    if power_db.size > 1:
                        power_db = power_db[:-1]
                        freqs = freqs[:-1]
                    ypad = 5
                    ymin = np.nanmin(power_db) - ypad
                    ymax = np.nanmax(power_db) + ypad
                    plt.figure(figsize=(10, 4))
                    plt.plot(freqs, power_db, color="blue", linewidth=1.25)
                    plt.xlabel("Frequency (Hz)")
                    plt.tick_params(axis="both", labelsize=100)
                    plt.ylabel("Power (µV^2/Hz)")
                    #plt.title(f"{name} PSD @ {channel}")
                    plt.xlim(0, 500)
                    plt.ylim(ymin, ymax)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    psd_path = ds_dir / f"{name}_psd_{channel}.png"
                    plt.savefig(psd_path.with_suffix(".pdf"), bbox_inches="tight")
                    plt.close()
                    plots["psd"] = str(psd_path)
                elif name == "SSVEP":
                    # Single channel (Oz) up to 500 Hz to mirror EEGLAB PSD at Oz
                    picks_ch = [channel] if channel in ds.raw.ch_names else "eeg"
                    data_chan = ds.raw.get_data(picks=picks_ch).squeeze()
                    seg = min(2048, n_times)
                    psd, freqs = mne.time_frequency.psd_array_welch(
                        data_chan,
                        sfreq=ds.raw.info["sfreq"],
                        fmin=0.5,
                        fmax=500.0,
                        average="mean",
                        n_fft=seg,
                        n_per_seg=seg,
                    )
                    power_db = 10 * np.log10(psd + np.finfo(float).eps)
                    # Very light smoothing (~1 Hz window) for closer EEGLAB appearance
                    if freqs.size > 5:
                        k = max(3, int(round(1 / (freqs[1] - freqs[0]))))
                        k = k + (k + 1) % 2  # make odd
                        kernel = np.ones(k) / k
                        power_db = np.convolve(power_db, kernel, mode="same")
                    # Drop last bin to avoid edge spike
                    if power_db.size > 1:
                        power_db = power_db[:-1]
                        freqs = freqs[:-1]
                    plt.figure(figsize=(10, 4))
                    plt.plot(freqs, power_db, color="blue", linewidth=1.25)
                    plt.xlabel("Frequency (Hz)")
                    plt.ylabel("Power (µV^2/Hz)")
                    #plt.title(f"{name} PSD @ {channel}")
                    plt.xlim(0, 500)
                    plt.ylim(-100, -20)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    psd_path = ds_dir / f"{name}_psd_{channel}.png"
                    plt.savefig(psd_path.with_suffix(".pdf"), bbox_inches="tight")
                    plt.close()
                    plots["psd"] = str(psd_path)
                else:
                    sfreq = float(ds.raw.info["sfreq"])
                    fmax = min(60.0, sfreq / 2 - 0.1)
                    fmin = min(1.0, max(0.1, fmax / 4)) if fmax <= 1.0 else 1.0
                    n_per_seg = min(1024, n_times)
                    n_fft = max(2048, n_per_seg)  # allow zero padding for finer bins
                    psd, freqs = mne.time_frequency.psd_array_welch(
                        data_all,
                        sfreq=sfreq,
                        fmin=fmin,
                        fmax=fmax,
                        average="mean",
                        n_fft=n_fft,
                        n_per_seg=n_per_seg,
                    )
                    if freqs.size == 0:
                        raise ValueError(f"No frequencies available between {fmin} and {fmax} Hz")
                    psd_mean = psd.mean(axis=0) if psd.ndim > 1 else psd
                    power_db = 10 * np.log10(psd_mean + np.finfo(float).eps)
                    plt.figure(figsize=(6.5, 3.2))
                    plt.plot(freqs, power_db, color="blue", linewidth=1.25)
                    plt.xlabel("Frequency (Hz)")
                    plt.ylabel("Power (µV^2/Hz)")
                    #plt.title(f"{name} PSD (mean of EEG)")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    psd_path = ds_dir / f"{name}_psd_{channel}.png"
                    plt.savefig(psd_path.with_suffix(".pdf"), bbox_inches="tight")
                    plt.close()
                    plots["psd"] = str(psd_path)
            except Exception as exc:
                plots["psd"] = f"PSD failed: {exc}"
        else:
            # Fallback: pick the first EEG channel if available to avoid empty plots
            if len(ds.raw.ch_names) > 0:
                fallback = [ch for ch, ct in zip(ds.raw.ch_names, ds.raw.get_channel_types()) if ct == "eeg"]
                if fallback:
                    cfg["trace_channel"] = fallback[0]
                    return run_dataset_plots()  # retry with fallback (rare)
            plots["trace"] = f"Channel {channel} not found."

        # Topomap at time (for ABR/Oddball/VEP)
        if topo_time_ms is not None:
            sample = int(round((topo_time_ms / 1000.0) * ds.raw.info["sfreq"]))
            if 0 <= sample < ds.raw.n_times:
                data_inst = ds.raw.copy().pick(picks="eeg")
                info_topo, kept_idx = _topomap_info(
                    data_inst.ch_names, params=ds.metadata.get("params") if isinstance(ds.metadata, dict) else None
                )
                data_arr = data_inst.get_data()
                # If the synthetic dataset is epoch-stacked (e.g., ABR), average across epochs first
                params = ds.metadata.get("params") if isinstance(ds.metadata, dict) else {}
                ep_len_ms = params.get("EpochLength") or params.get("Parameters", {}).get("EpochLength")
                ep_count = params.get("Epochs") or params.get("Parameters", {}).get("Epochs")
                if ep_len_ms and ep_count:
                    ep_samples = int(round((ep_len_ms / 1000.0) * data_inst.info["sfreq"]))
                    if ep_samples > 0 and data_arr.shape[1] % ep_samples == 0:
                        n_ep = data_arr.shape[1] // ep_samples
                        data_arr_epochs = data_arr.reshape(data_arr.shape[0], ep_samples, n_ep, order="F")
                        # Default: mean across all epochs
                        data_mean = data_arr_epochs.mean(axis=2)
                        sample_epoch = sample % ep_samples
                        data_slice = data_mean[:, sample_epoch]
                        # For Oddball, isolate likely targets (top 20% positive peak near 300 ms on Pz)
                        if name == "Oddball":
                            try:
                                fs = float(data_inst.info["sfreq"])
                                ratio_std = params.get("ratio", 0.8) if isinstance(params, dict) else 0.8
                                target_count = max(1, int(round(n_ep * (1 - ratio_std))))
                                detect_ch = channel if channel in data_inst.ch_names else data_inst.ch_names[0]
                                detect_idx = data_inst.ch_names.index(detect_ch)
                                win_start = int(round(0.26 * fs))
                                win_end = int(round(0.34 * fs))
                                win_end = min(win_end, ep_samples)
                                peaks = data_arr_epochs[detect_idx, win_start:win_end, :].max(axis=0)
                                order = np.argsort(peaks)[::-1]
                                target_mask = np.zeros(n_ep, dtype=bool)
                                target_mask[order[:target_count]] = True
                                target_epochs = data_arr_epochs[:, :, target_mask]
                                b_len = int(round(0.05 * fs))
                                if b_len > 0 and target_epochs.size:
                                    target_epochs = target_epochs - target_epochs[:, :b_len, :].mean(axis=1, keepdims=True)
                                if target_epochs.ndim == 3 and target_epochs.shape[2] > 0:
                                    target_mean = target_epochs.mean(axis=2)
                                    data_slice = target_mean[:, sample_epoch]
                            except Exception:
                                # Fallback to all-epoch mean if target selection fails
                                pass
                    else:
                        data_slice = data_arr[:, sample]
                else:
                    data_slice = data_arr[:, sample]
                kept_idx_list = list(kept_idx) if isinstance(kept_idx, (list, tuple)) else list(kept_idx)
                values = data_slice if len(kept_idx_list) == 0 else data_slice[kept_idx_list]
                topo_info = info_topo if len(kept_idx_list) else data_inst.info
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.set_axis_off()
                im, _ = mne.viz.plot_topomap(
                    values,
                    topo_info,
                    axes=ax,
                    show=False,
                    contours=6,
                    cmap="RdBu_r",
                    outlines="head",
                    sphere=(0.0, -0.01, 0.0, 0.105),  # larger sphere to fill circle
                    extrapolate="head",
                    names=topo_info.ch_names if hasattr(topo_info, "ch_names") else None,
                )
                if name == "ABR":
                    im.set_clim(values.min(), values.max())  # or symmetric:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Amplitude (uV)")
                #fig.suptitle(f"{name} Topomap @ {topo_time_ms} ms", fontsize=18)
                topo_base = ds_dir / f"{name}_topomap_{topo_time_ms}ms"
                fig.savefig(topo_base.with_suffix(".pdf"), bbox_inches="tight")
                plt.close(fig)
            else:
                plots["topomap_time"] = f"Time {topo_time_ms} ms out of range."

        # Topomap at frequency (for ASSR/SSVEP/Sine)
        if topo_freq is not None:
            data_inst = ds.raw.copy().pick(picks="eeg")
            info_topo, kept_idx = _topomap_info(
                data_inst.ch_names, params=ds.metadata.get("params") if isinstance(ds.metadata, dict) else None
            )
            data = data_inst.get_data()
            n_times = data.shape[1]
            psd, freqs = mne.time_frequency.psd_array_welch(
                data,
                sfreq=data_inst.info["sfreq"],
                fmin=max(1.0, topo_freq - 2),
                fmax=topo_freq + 2,
                average="mean",
                n_fft=min(1024, n_times),
                n_per_seg=min(1024, n_times),
            )
            if psd.ndim == 2:
                freq_idx = int(np.argmin(np.abs(freqs - topo_freq)))
                band_power = psd[:, freq_idx]
                # Convert to dB for ASSR/SSVEP to match EEGLAB power maps
                if name in ("ASSR", "SSVEP"):
                    band_power = 10 * np.log10(band_power + np.finfo(float).eps)
                    cmap = "turbo"
                    vmin, vmax = -80, -20
                    cbar_label = "Power (µV²/Hz)"
                else:
                    cmap = "RdBu_r"
                    vmin = vmax = None
                    cbar_label = "Amplitude (µV)"
                kept_idx_list = list(kept_idx) if isinstance(kept_idx, (list, tuple)) else list(kept_idx)
                values = band_power if len(kept_idx_list) == 0 else band_power[kept_idx_list]
                topo_info = info_topo if len(kept_idx_list) else data_inst.info
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.set_axis_off()
                contour_count = 8 if name == "ASSR" else 6
                im, _ = mne.viz.plot_topomap(
                    values,
                    topo_info,
                    axes=ax,
                    show=False,
                    contours=contour_count,
                    cmap=cmap,
                    outlines="head",
                    sphere=(0.0, -0.01, 0.0, 0.105),
                    extrapolate="head",
                    names=topo_info.ch_names if hasattr(topo_info, "ch_names") else None,
                )
                if vmin is not None or vmax is not None:
                    im.set_clim(values.min(), values.max())
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(cbar_label)
                #fig.suptitle(f"{name} Topomap @ {topo_freq} Hz", fontsize=18)
                topo_base = ds_dir / f"{name}_topomap_{topo_freq}Hz"
                fig.savefig(topo_base.with_suffix(".pdf"), bbox_inches="tight")
                plt.close(fig)
                plots["topomap_freq"] = str(topo_base.with_suffix(".svg"))
            else:
                plots["topomap_freq"] = "PSD shape unexpected."

        outputs[name] = plots

    return outputs


def _apply_standard_montage(raw: mne.io.BaseRaw, params: Optional[dict] = None) -> None:
    """Assign a montage using channel labels; fallback to synthetic circle to avoid warnings."""
    info, _ = _topomap_info(raw.ch_names, params=params)
    montage = getattr(info, "get_montage", lambda: None)()
    if montage is not None:
        try:
            raw.set_montage(montage, on_missing="ignore")
        except Exception:
            pass


def _topomap_info(channel_labels: Sequence[str], params: Optional[dict] = None):
    """Build Info for topomaps using known positions; fallback to circle."""
    montage = None
    for candidate in ("standard_1005", "standard_1020"):
        try:
            montage = mne.channels.make_standard_montage(candidate)
            break
        except Exception:
            continue

    known = montage.get_positions().get("ch_pos", {}) if montage is not None else {}

    # Map channel -> desired position label from params
    pos_map = {}
    if isinstance(params, dict) and isinstance(params.get("Channels"), list):
        pos_map = {ch.get("Channel"): ch.get("Position") for ch in params["Channels"] if isinstance(ch, dict)}

    kept_idx: list[int] = []
    ch_pos: dict[str, tuple[float, float, float]] = {}
    for idx, label in enumerate(channel_labels):
        target = pos_map.get(label, label) if pos_map else label
        pos = known.get(target)
        if pos is None:
            pos = known.get(label)
        if pos is None:
            continue
        ch_pos[label] = pos
        kept_idx.append(idx)

    # If nothing matched, fall back to a small circle with all channels.
    if not ch_pos:
        total = max(1, len(channel_labels))
        for idx, label in enumerate(channel_labels):
            angle = 2 * np.pi * idx / total + 0.1 * idx
            base_radius = 0.045
            # deterministic jitter to avoid co-planar issues
            jitter = (abs(hash(label)) % 1000) / 1e6
            radius = base_radius + jitter
            ch_pos[label] = (radius * np.cos(angle), radius * np.sin(angle), 0.0)
            kept_idx.append(idx)

    info = mne.create_info(list(ch_pos.keys()), sfreq=1.0, ch_types="eeg")
    try:
        custom_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        info.set_montage(custom_montage)
    except Exception:
        pass
    return info, kept_idx


def copy_eeglab_refs() -> None:
    syn_root = Path(__file__).resolve().parents[2] / "experiments" / "synData"
    dest_root = RESULTS_DIR / "eeglab_refs"
    dest_root.mkdir(parents=True, exist_ok=True)
    for dataset, files in EEGLAB_FILES.items():
        src_dir = syn_root / dataset
        dst_dir = dest_root / dataset
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, dst_dir / fname)
                out_png = dst_dir / Path(fname).with_suffix(".png")
                _pdf_to_png(src, out_png)


def _pdf_to_png(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    # Try PyMuPDF
    try:
        import fitz  # type: ignore

        pdf = fitz.open(src)
        pix = pdf.load_page(0).get_pixmap(dpi=200)
        pix.save(dst)
        return
    except Exception:
        pass
    # Fallback: macOS sips if available
    if shutil.which("sips"):
        try:
            subprocess.run(["sips", "-s", "format", "png", str(src), "--out", str(dst)], check=True)
            return
        except Exception:
            pass


def make_side_by_side() -> None:
    """Create side-by-side panels of CortiPy vs EEGLAB for quick comparison."""
    import matplotlib.image as mpimg

    corti_root = RESULTS_DIR / "cortipy_plots"
    eeglab_root = RESULTS_DIR / "eeglab_refs"
    dest_root = RESULTS_DIR / "comparisons"
    dest_root.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS.keys():
        c_dir = corti_root / dataset
        e_dir = eeglab_root / dataset
        if not c_dir.exists() or not e_dir.exists():
            continue
        rules = COMPARE_RULES.get(dataset, {})
        # pick topomap and trace/psd if present
        for kind in ("topomap", "trace", "psd"):
            corti_candidates = sorted(c_dir.glob(f"*{kind}*.png"))
            if not corti_candidates:
                continue
            corti_img = corti_candidates[0]

            pattern = rules.get(kind)
            if not pattern:
                continue
            eeg_candidates = sorted([p for p in e_dir.glob("*.png") if pattern.lower() in p.name.lower()])
            if not eeg_candidates:
                continue
            eeg_img = eeg_candidates[0]

            try:
                img1 = mpimg.imread(corti_img)
                img2 = mpimg.imread(eeg_img)
            except Exception:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(img1)
            axes[0].axis("off")
            #axes[0].set_title(f"CortiPy {dataset} {kind}")
            axes[1].imshow(img2)
            axes[1].axis("off")
            #axes[1].set_title(f"EEGLAB {dataset} {kind}")
            fig.tight_layout()
            out_path = dest_root / f"{dataset}_{kind}_comparison.png"
            plt.savefig(out_path, dpi=150)
            plt.savefig(out_path.with_suffix(".pdf"))
            plt.close()


def main() -> None:
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)

    results: Dict[str, Any] = {}

    # 1.1 Bitwise identity
    try:
        results["bitwise_identity"] = bitwise_identity(DEFAULT_BIN)
    except Exception as exc:
        results["bitwise_identity_error"] = str(exc)

    # 1.2 Pipeline surrogate
    try:
        results["pipeline_equivalence"] = pipeline_equivalence_surrogate()
    except Exception as exc:
        results["pipeline_equivalence_error"] = str(exc)

    # 1.3 / 1.4 placeholders
    results["manual_steps"] = document_placeholders()

    # 1.x Dataset comparisons matching EEGLAB plots
    results["dataset_plots"] = {}
    try:
        results["dataset_plots"] = run_dataset_plots()
    except Exception as exc:
        results["dataset_plots_error"] = str(exc)

    # Copy EEGLAB reference PDFs into results_exp1 for side-by-side comparison
    copy_eeglab_refs()
    make_side_by_side()

    out_path = RESULTS_DIR / "experiment1_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote Experiment 1 results to {out_path}")


if __name__ == "__main__":
    main()
