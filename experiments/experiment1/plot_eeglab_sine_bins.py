"""Plot EEGLAB sine ERP/PSD BIN files that store axis + data concatenated."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _split_half(raw: np.ndarray):
    half = raw.size // 2
    return raw[:half], raw[half:]


def _split_fortran(raw: np.ndarray):
    arr = raw.reshape((-1, 2), order="F")
    return arr[:, 0], arr[:, 1]


def _split_interleaved(raw: np.ndarray):
    # axis, data alternating: t0,a0,t1,a1,...
    axis = raw[0::2]
    data = raw[1::2]
    return axis, data


def load_axis_data(bin_path: Path, split: str = "half"):
    raw = np.fromfile(bin_path, dtype=np.float64)
    if raw.size % 2 != 0:
        raise ValueError(f"{bin_path} length is not even; cannot split axis/data automatically.")

    if split == "interleaved":
        return _split_interleaved(raw)
    if split == "half":
        return _split_half(raw)
    if split == "fortran":
        return _split_fortran(raw)

    # auto mode: try both
    candidates = []
    for splitter, label in (
        (_split_interleaved, "interleaved"),
        (_split_half, "half"),
        (_split_fortran, "fortran"),
    ):
        try:
            ax, da = splitter(raw)
            mono = np.all(np.diff(ax) >= 0) or np.all(np.diff(ax) <= 0)
            candidates.append((label, ax, da, mono))
        except Exception:
            continue

    # Prefer monotonic axis with larger span
    def score(ax, mono):
        if not mono:
            return -np.inf
        def score(ax):
            return np.nanmax(ax) - np.nanmin(ax)
        return score(ax)

    if not candidates:
        raise ValueError(f"Could not split {bin_path}")

    best = max(candidates, key=lambda c: score(c[1], c[3]))
    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--erp-bin",
        type=Path,
        default=Path("experiments/datasets/D2_software_curated_signals/ContinuousSine/EEGlab_erp_average.bin"),
        help="Path to EEGLAB ERP BIN (axis + data).",
    )
    parser.add_argument(
        "--psd-bin",
        type=Path,
        default=Path("experiments/datasets/D2_software_curated_signals/ContinuousSine/EEGlab_psd_dB_continuous.bin"),
        help="Path to EEGLAB PSD BIN (axis + data, dB).",
    )
    parser.add_argument(
        "--split",
        choices=["interleaved", "half", "fortran", "auto"],
        default="interleaved",
        help="How to split axis/data (default: interleaved t0,a0,t1,a1...; use auto/half/fortran if needed).",
    )
    args = parser.parse_args()

    t_axis, erp = load_axis_data(args.erp_bin, split=args.split)
    f_axis, psd_db = load_axis_data(args.psd_bin, split=args.split)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(t_axis, erp, color="blue", linewidth=1.25)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title(f"ERP average ({args.erp_bin.name})")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(f_axis, psd_db, color="blue", linewidth=1.25)
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Power (dB)")
    ax[1].set_title(f"PSD (dB) ({args.psd_bin.name})")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
