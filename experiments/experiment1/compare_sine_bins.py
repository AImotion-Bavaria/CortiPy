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


def summarize(name: str, axis_ref, data_ref, axis_new, data_new):
    length = min(len(data_ref), len(data_new))
    data_ref_c = data_ref[:length]
    data_new_c = data_new[:length]
    diff = data_new_c - data_ref_c

    def stats(label, arr):
        return f"{label}: min={arr.min():.6g}, max={arr.max():.6g}, mean={arr.mean():.6g}, std={arr.std():.6g}"

    print(f"\n== {name} ==")
    print(stats("EEGLAB data", data_ref_c))
    print(stats("CortiPy data", data_new_c))
    print(stats("Diff (new-ref)", diff))
    # relative scale at peak absolute of reference
    ref_peak_idx = np.argmax(np.abs(data_ref_c))
    ref_peak = data_ref_c[ref_peak_idx]
    new_at_peak = data_new_c[ref_peak_idx]
    ratio = new_at_peak / ref_peak if ref_peak != 0 else np.nan
    print(f"Ratio at ref peak idx {ref_peak_idx}: new/ref = {ratio:.6g}")
    # axis sanity
    ax_len = min(len(axis_ref), len(axis_new))
    ax_diff = np.abs(axis_ref[:ax_len] - axis_new[:ax_len])
    print(f"Axis diff: max={ax_diff.max():.6g}, mean={ax_diff.mean():.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eeglab-erp",
        type=Path,
        default=Path("experiments/datasets/D2_software_curated_signals/ContinuousSine/EEGlab_erp_average.bin"),
        help="EEGLAB ERP bin (axis+data).",
    )
    parser.add_argument(
        "--eeglab-psd",
        type=Path,
        default=Path("experiments/datasets/D2_software_curated_signals/ContinuousSine/EEGlab_psd_dB_continuous.bin"),
        help="EEGLAB PSD bin (axis+data).",
    )
    parser.add_argument(
        "--new-erp",
        type=Path,
        default=Path("experiments/experiment1/results_sine/bin/CortiPy_erp_average.bin"),
        help="Regenerated ERP bin (axis+data).",
    )
    parser.add_argument(
        "--new-psd",
        type=Path,
        default=Path("experiments/experiment1/results_sine/bin/CortiPy_psd_dB.bin"),
        help="Regenerated PSD bin (axis+data, dB).",
    )
    parser.add_argument(
        "--split",
        choices=["interleaved", "half"],
        default="interleaved",
        help="Axis/data split mode (default: interleaved t0,a0,...).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Only print statistics, do not show plots.")
    args = parser.parse_args()

    t_eeg, erp_eeg = load_axis_data(args.eeglab_erp, split=args.split)
    f_eeg, psd_eeg = load_axis_data(args.eeglab_psd, split=args.split)
    t_new, erp_new = load_axis_data(args.new_erp, split=args.split)
    f_new, psd_new = load_axis_data(args.new_psd, split=args.split)

    summarize("ERP", t_eeg, erp_eeg, t_new, erp_new)
    summarize("PSD (dB)", f_eeg, psd_eeg, f_new, psd_new)

    if args.no_plot:
        return

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
