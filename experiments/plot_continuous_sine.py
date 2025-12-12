"""Quick visualization of the ContinuousSine BIN dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from cortipy.shared import CortiDataset  # type: ignore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bin_dir",
        type=Path,
        nargs="?",
        default=Path("experiments/synData/ContinuousSine"),
        help="Path to the ContinuousSine dataset directory.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Explicit path to the .bin file (defaults to ContinuousSine_continuous.bin).",
    )
    parser.add_argument("--channel", type=int, default=0, help="Channel index to plot (default: 0)")
    parser.add_argument("--seconds", type=float, default=2.0, help="Seconds to show from start (default: 2.0)")
    parser.add_argument("--channel-count", type=int, default=64, help="Channel count fallback (default: 64)")
    parser.add_argument("--sampling-rate", type=float, default=1000.0, help="Sampling rate fallback (default: 1000.0)")
    args = parser.parse_args()

    data_path = args.data_path or (args.bin_dir / "ContinuousSine_continuous.bin")
    ds = CortiDataset.from_bin(
        args.bin_dir,
        data_path=data_path,
        channel_count=args.channel_count,
        sampling_rate=args.sampling_rate,
    )
    raw = ds.raw

    # Time-series snippet
    ch = max(0, min(args.channel, len(raw.ch_names) - 1))
    t = raw.times
    data = raw.get_data(picks=[ch]).squeeze()
    mask = t <= args.seconds
    plt.figure(figsize=(10, 3))
    plt.plot(t[mask], data[mask], linewidth=1)
    plt.title(f"Channel {raw.ch_names[ch]} (first {args.seconds} s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # PSD of that channel
    psd_obj = raw.compute_psd(fmax=raw.info["sfreq"] / 2.0, method="welch")
    psd, freqs = psd_obj.get_data(return_freqs=True)
    plt.figure(figsize=(10, 3))
    plt.semilogy(freqs, psd[ch], linewidth=1)
    plt.title(f"PSD @ {raw.ch_names[ch]}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (µV^2/Hz)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
