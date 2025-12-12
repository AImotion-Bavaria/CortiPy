"""Compare EEGLAB sine ERP/PSD vs regenerated results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_axis_data(bin_path: Path, split: str = "interleaved"):
    raw = np.fromfile(bin_path, dtype=np.float64)
    if raw.size % 2 != 0:
        raise ValueError(f"{bin_path} length is not even; cannot split axis/data automatically.")
    if split == "interleaved":
        axis = raw[0::2]
        data = raw[1::2]
        return axis, data
    half = raw.size // 2
    return raw[:half], raw[half:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eeglab-erp",
        type=Path,
        default=Path("experiments/synData/ContinuousSine/EEGlab_erp_average.bin"),
        help="EEGLAB ERP bin (axis+data).",
    )
    parser.add_argument(
        "--eeglab-psd",
        type=Path,
        default=Path("experiments/synData/ContinuousSine/EEGlab_psd_dB_continuous.bin"),
        help="EEGLAB PSD bin (axis+data).",
    )
    parser.add_argument(
        "--new-erp",
        type=Path,
        default=Path("experiments/format_benchmark/results_sine/bin/CortiPy_erp_average.bin"),
        help="Regenerated ERP bin (axis+data).",
    )
    parser.add_argument(
        "--new-psd",
        type=Path,
        default=Path("experiments/format_benchmark/results_sine/bin/CortiPy_psd_dB.bin"),
        help="Regenerated PSD bin (axis+data, dB).",
    )
    parser.add_argument(
        "--split",
        choices=["interleaved", "half"],
        default="interleaved",
        help="Axis/data split mode (default: interleaved t0,a0,...).",
    )
    args = parser.parse_args()

    t_eeg, erp_eeg = load_axis_data(args.eeglab_erp, split=args.split)
    f_eeg, psd_eeg = load_axis_data(args.eeglab_psd, split=args.split)
    t_new, erp_new = load_axis_data(args.new_erp, split=args.split)
    f_new, psd_new = load_axis_data(args.new_psd, split=args.split)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex="col")

    axes[0, 0].plot(t_eeg, erp_eeg, color="blue", linewidth=1)
    axes[0, 0].set_title(f"EEGLAB ERP ({args.eeglab_erp.name})")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t_new, erp_new, color="blue", linewidth=1)
    axes[0, 1].set_title(f"New ERP ({args.new_erp.name})")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(f_eeg, psd_eeg, color="blue", linewidth=1)
    axes[1, 0].set_title(f"EEGLAB PSD ({args.eeglab_psd.name})")
    axes[1, 0].set_xlabel("Frequency")
    axes[1, 0].set_ylabel("Power (dB)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(f_new, psd_new, color="blue", linewidth=1)
    axes[1, 1].set_title(f"New PSD ({args.new_psd.name})")
    axes[1, 1].set_xlabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
