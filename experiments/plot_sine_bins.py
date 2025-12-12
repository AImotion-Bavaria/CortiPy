"""Plot the average and PSD signals stored as BIN files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bin_dir",
        type=Path,
        help="Directory containing sine_average.bin and sine_psd.bin (default: experiments/format_benchmark/results_sine/bin)",
    )
    parser.add_argument(
        "--time-bin",
        type=Path,
        default=None,
        help="Optional .bin file containing time vector (float64) matching sine_average length.",
    )
    parser.add_argument(
        "--avg-window-s",
        type=float,
        default=None,
        help="Time window (seconds) to show in the average plot. Omit for full length.",
    )
    parser.add_argument(
        "--freq-bin",
        type=Path,
        default=None,
        help="Optional .bin file containing frequency vector (float64) matching sine_psd length.",
    )
    args = parser.parse_args()

    avg_path = args.bin_dir / "sine_average.bin"
    psd_path = args.bin_dir / "sine_psd.bin"
    time_path = args.time_bin
    freq_path = args.freq_bin

    if not avg_path.exists() or not psd_path.exists():
        raise FileNotFoundError("Expected sine_average.bin and sine_psd.bin in the provided directory.")

    avg_raw = np.fromfile(avg_path, dtype=np.float64)
    psd_raw = np.fromfile(psd_path, dtype=np.float64)

    def _split_axis_data(arr: np.ndarray, fallback_axis: np.ndarray | None = None):
        if arr.size % 2 != 0:
            return fallback_axis if fallback_axis is not None else np.arange(arr.size), arr
        # heuristic: try concat then interleaved
        half = arr.size // 2
        axis_concat, data_concat = arr[:half], arr[half:]
        is_concat = np.all(np.diff(axis_concat) >= 0)
        if is_concat:
            return axis_concat, data_concat
        axis_inter = arr[0::2]
        data_inter = arr[1::2]
        return axis_inter, data_inter

    time, avg = _split_axis_data(
        avg_raw, fallback_axis=np.fromfile(time_path, dtype=np.float64) if time_path and time_path.exists() else None
    )
    freqs, psd = _split_axis_data(
        psd_raw, fallback_axis=np.fromfile(freq_path, dtype=np.float64) if freq_path and freq_path.exists() else None
    )

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    ax[0].plot(time, avg, color="blue", linewidth=1.25)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Sine average (from BIN)")
    ax[0].grid(True, alpha=0.3)
    # Highlight initial window if time is in seconds; else show first N samples
    if len(time) == len(avg) and args.avg_window_s:
        try:
            if time.max() > 1.0:
                mask = time <= args.avg_window_s
                ax[0].set_xlim(time[0], time[mask].max() if mask.any() else time.max())
            else:
                # time vector is likely samples; fall back to matching sample count
                sample_win = int(round(args.avg_window_s * len(time)))
                ax[0].set_xlim(time[0], time[min(len(time) - 1, sample_win)])
        except Exception:
            pass

    ax[1].plot(freqs, psd, color="blue", linewidth=1.25)
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Power")
    ax[1].set_title("Sine PSD (from BIN)")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
