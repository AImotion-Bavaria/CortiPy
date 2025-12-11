"""Compute and save average trace and PSD for a synthetic sine dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from cortipy.shared import CortiDataset  # type: ignore


def compute_avg_psd(raw: mne.io.BaseRaw):
    """Return time vector, channel-mean trace, and channel-mean PSD."""
    data = raw.get_data(picks="eeg")
    time = raw.times
    avg = data.mean(axis=0)

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=raw.info["sfreq"],
        fmin=0.5,
        fmax=raw.info["sfreq"] / 2.0,
        average="mean",
        n_fft=min(4096, data.shape[-1]),
        n_per_seg=min(4096, data.shape[-1]),
    )
    psd_mean = psd.mean(axis=0) if psd.ndim > 1 else psd
    return time, avg, freqs, psd_mean


def plot_and_save(time, avg, freqs, psd, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Average trace
    plt.figure(figsize=(10, 4))
    plt.plot(time, avg, color="blue", linewidth=1.25)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("Sine average (mean across channels)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sine_average.png", dpi=200)
    plt.savefig(out_dir / "sine_average.pdf")
    plt.close()

    # PSD
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, psd, color="blue", linewidth=1.25)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (µV^2/Hz)")
    plt.title("Sine PSD (mean across channels)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sine_psd.png", dpi=200)
    plt.savefig(out_dir / "sine_psd.pdf")
    plt.close()

    # Numeric dump
    np.savez(out_dir / "sine_avg_psd.npz", time=time, average=avg, freqs=freqs, psd=psd)

    # BIN exports (simple .bin with double precision)
    (out_dir / "bin").mkdir(exist_ok=True)
    avg_bin = out_dir / "bin" / "sine_average.bin"
    psd_bin = out_dir / "bin" / "sine_psd.bin"
    avg.astype(np.float64).tofile(avg_bin)
    psd.astype(np.float64).tofile(psd_bin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bin_dir", type=Path, help="Path to synthetic sine BIN directory (params.json + .bin)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/format_benchmark/results_sine"),
        help="Where to save average/PSD outputs (default: experiments/format_benchmark/results_sine)",
    )
    args = parser.parse_args()

    bin_path = args.bin_dir / "ContinuousSine_continuous.bin"
    # ContinuousSine params may not list channel count; infer from params if present else fallback to 64
    channel_count = None
    params_file = args.bin_dir / "params.json"
    if params_file.exists():
        try:
            import json

            params = json.loads(params_file.read_text())
            cc = params.get("Parameters", {}).get("NumberEEGChannels")
            if cc:
                channel_count = int(cc)
        except Exception:
            pass
    ds = CortiDataset.from_bin(
        args.bin_dir,
        data_path=bin_path if bin_path.exists() else None,
        channel_count=channel_count or 64,
        sampling_rate=1000.0,
    )
    time, avg, freqs, psd = compute_avg_psd(ds.raw)
    plot_and_save(time, avg, freqs, psd, args.output_dir)


if __name__ == "__main__":
    main()
