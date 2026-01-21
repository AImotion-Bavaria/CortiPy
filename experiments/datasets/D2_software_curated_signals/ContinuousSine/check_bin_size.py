#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show file size and derived sample counts for the ContinuousSine binary."
    )
    parser.add_argument(
        "bin_file",
        nargs="?",
        default="experiments/datasets/D2_software_curated_signals/ContinuousSine/ContinuousSine_continuous.bin",
        help="Path to the binary file to inspect.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Number of channels stored sequentially. ContinuousSine uses 2 (SINE, TRIGGER).",
    )
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        default=8,
        help="Bytes per sample. ContinuousSine is written as double precision (8 bytes).",
    )
    args = parser.parse_args()

    path = Path(args.bin_file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    size_bytes = path.stat().st_size
    mib = size_bytes / (1024 * 1024)
    total_values = size_bytes // args.dtype_bytes
    samples_per_channel = (
        total_values // args.channels if args.channels > 0 else 0
    )

    print(f"File: {path}")
    print(f"Size: {size_bytes} bytes ({mib:.2f} MiB)")
    print(f"Values: {total_values} ({args.dtype_bytes}-byte samples)")
    if args.channels > 0:
        print(f"Channels: {args.channels}")
        print(f"Samples per channel: {samples_per_channel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
