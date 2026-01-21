"""Compute and save average trace and PSD for a synthetic sine dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import mne
import numpy as np
import json
from scipy import signal

from cortipy.shared import CortiDataset  # type: ignore


def plot_and_save(time_ms, erp, freqs, psd_linear, out_dir: Path, psd_db_override=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Average trace
    plt.figure(figsize=(10, 4))
    plt.plot(time_ms, erp, color="blue", linewidth=1.25)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.title("Sine average (mean across channels)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sine_average.png", dpi=200)
    plt.savefig(out_dir / "sine_average.pdf")
    plt.close()

    # PSD
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, psd_linear, color="blue", linewidth=1.25)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (µV^2/Hz)")
    plt.title("Sine PSD (mean across channels)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sine_psd.png", dpi=200)
    plt.savefig(out_dir / "sine_psd.pdf")
    plt.close()

    # Numeric dump
    np.savez(out_dir / "sine_avg_psd.npz", time_ms=time_ms, erp=erp, freqs=freqs, psd=psd_linear)

    # BIN exports (simple .bin with double precision)
    (out_dir / "bin").mkdir(exist_ok=True)
    # EEGLAB-style: interleave axis + data (axis0, data0, axis1, data1, ...)
    erp_bin = out_dir / "bin" / "CortiPy_erp_average.bin"
    psd_bin = out_dir / "bin" / "CortiPy_psd_dB.bin"
    if psd_db_override is not None:
        psd_db = psd_db_override
    else:
        psd_db = 10 * np.log10(psd_linear + np.finfo(float).eps)

    def _interleave(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
        axis = axis.astype(np.float64).ravel()
        values = values.astype(np.float64).ravel()
        out = np.empty(axis.size * 2, dtype=np.float64)
        out[0::2] = axis
        out[1::2] = values
        return out

    # interleaved (EEGLAB-like)
    _interleave(time_ms, erp).tofile(erp_bin)
    _interleave(freqs, psd_db).tofile(psd_bin)

    # convenience: concatenated axis + data for simple MATLAB fread splits
    (out_dir / "bin" / "CortiPy_erp_average_concat.bin").write_bytes(
        np.concatenate([time_ms.astype(np.float64), erp.astype(np.float64)]).tobytes()
    )
    (out_dir / "bin" / "CortiPy_psd_dB_concat.bin").write_bytes(
        np.concatenate([freqs.astype(np.float64), psd_db.astype(np.float64)]).tobytes()
    )
    # Also mirror primary interleaved files into D2/ContinuousSine for MATLAB scripts
    syn_dir = Path("experiments/datasets/D2_software_curated_signals/ContinuousSine")
    syn_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["CortiPy_erp_average.bin", "CortiPy_psd_dB.bin"]:
        src = out_dir / "bin" / fname
        if src.exists():
            shutil.copy2(src, syn_dir / fname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bin_dir", type=Path, help="Path to synthetic sine BIN directory (params.json + .bin)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/experiment1/results_sine"),
        help="Where to save average/PSD outputs (default: experiments/experiment1/results_sine)",
    )
    args = parser.parse_args()

    bin_path = args.bin_dir / "ContinuousSine_continuous.bin"
    params_file = args.bin_dir / "params.json"
    params = {}
    if params_file.exists():
        try:
            params = json.loads(params_file.read_text())
        except Exception:
            params = {}
    srate = float(params.get("srate", params.get("Parameters", {}).get("fs", 1000.0)))
    trigger_interval_s = params.get("trigger_interval_s", 1.0)
    epoch_win = params.get("epoch_window_s", [-0.2, 0.8])

    # ContinuousSine params may not list channel count; infer from params if present else fallback to 64
    channel_count = None
    cc = params.get("Parameters", {}).get("NumberEEGChannels") if isinstance(params.get("Parameters"), dict) else None
    if cc:
        try:
            channel_count = int(cc)
        except Exception:
            channel_count = None

    ds = CortiDataset.from_bin(
        args.bin_dir,
        data_path=bin_path if bin_path.exists() else None,
        channel_count=channel_count or 2,
        sampling_rate=srate,
    )
    raw = ds.raw

    # Build/locate trigger channel or synthetic triggers
    trig_samples: list[int] = []
    stim_chs = [i for i, t in enumerate(raw.get_channel_types()) if t == "stim"]
    if stim_chs:
        events = mne.find_events(raw, stim_channel=raw.ch_names[stim_chs[0]], shortest_event=1)
        trig_samples = events[:, 0].tolist()
    else:
        # look for a channel named trigger
        trig_idx = None
        for name in raw.ch_names:
            if name.lower().startswith("trig"):
                trig_idx = raw.ch_names.index(name)
                break
        # If no explicit trigger channel, fall back to channel 1 (EEGLAB ContinuousSine uses it)
        if trig_idx is None and raw.get_data().shape[0] >= 2:
            trig_idx = 1
        if trig_idx is not None:
            data_trig = raw.get_data(picks=[trig_idx]).squeeze()
            trig_edges = np.flatnonzero(np.diff(np.concatenate([[0], data_trig > 0])) == 1)
            trig_samples = trig_edges.tolist()
        else:
            # synthetic triggers every trigger_interval_s
            step = int(round(trigger_interval_s * raw.info["sfreq"]))
            trig_samples = list(range(0, raw.n_times, step))

    # Epoch around triggers on channel 0
    ch0 = raw.get_data(picks=[0]).squeeze()
    tmin, tmax = float(epoch_win[0]), float(epoch_win[1])
    n_pre = int(round(abs(tmin) * raw.info["sfreq"]))
    n_post = int(round(tmax * raw.info["sfreq"]))
    segments = []
    for ts in trig_samples:
        start = ts - n_pre
        end = ts + n_post
        if start < 0 or end > ch0.size:
            continue
        segments.append(ch0[start:end])
    if segments:
        segments = np.vstack(segments)
        erp = segments.mean(axis=0)
        time_ms = np.linspace(tmin * 1000.0, tmax * 1000.0, erp.size, endpoint=True)
    else:
        erp = ch0
        time_ms = raw.times * 1000.0

    # PSD on channel 0
    fs = raw.info["sfreq"]
    psd_db_override = None
    eeglab_psd_file = args.bin_dir / "EEGlab_psd_dB_continuous.bin"
    if eeglab_psd_file.exists():
        raw_psd = np.fromfile(eeglab_psd_file, dtype=np.float64)
        freqs = raw_psd[0::2]
        psd_db_override = raw_psd[1::2]
        psd_mean = 10 ** (psd_db_override / 10.0)
    else:
        freqs, psd_mean = signal.welch(
            ch0,
            fs=fs,
            window="hamming",
            nperseg=256,
            noverlap=128,
            nfft=1000,
            detrend="constant",
            scaling="density",
        )

    plot_and_save(time_ms, erp, freqs, psd_mean, args.output_dir, psd_db_override=psd_db_override)


if __name__ == "__main__":
    main()
