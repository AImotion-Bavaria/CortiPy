"""Convert BIN -> SBIDS and run SSVEP evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cortipy.shared import CortiDataset  # type: ignore


def _json_safe(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _summarize_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Reduce evaluation payload to a concise summary to avoid huge dumps."""
    import numpy as np

    if not isinstance(evaluation, dict) or not evaluation:
        return {}

    summary: dict[str, Any] = {"keys": sorted(evaluation.keys())}

    def _mean(key: str) -> Any:
        arr = evaluation.get(key)
        try:
            val = float(np.nanmean(np.asarray(arr, dtype=float)))
            return val
        except Exception:
            return None

    for metric in ("SNR_2_45Hz", "SNR_max", "T2circ"):
        val = _mean(metric)
        if val is not None:
            summary[f"{metric}_mean"] = val
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bin_dir", type=Path, help="Path to BIN dataset directory (params.json + data.bin).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SBIDS JSON-LD path (default: sbids_meta_<dataset>.jsonld under bin_dir).",
    )
    parser.add_argument(
        "--export-format",
        default="parquet",
        choices=["parquet", "npz", "copy", "edf", "hdf5", "zarr"],
        help="Container to store raw data in SBIDS raw_data/ (default: %(default)s).",
    )
    parser.add_argument(
        "--stim-freq",
        type=float,
        nargs="+",
        default=None,
        help="Override stimulus frequencies (Hz) for SSVEP evaluation.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib plots during SSVEP evaluation.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    warnings.filterwarnings("ignore", message="Online software filter detected.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Not setting position of .* misc channel.*", category=RuntimeWarning)

    parser = _build_parser()
    args = parser.parse_args(argv)

    ds = CortiDataset.from_bin(args.bin_dir)

    # Only override StimFreq if the user provided it; otherwise let evaluate_ssvep
    # pull parameters from ds.metadata.
    params_override = None
    if args.stim_freq is not None:
        params_override = {"Parameters": {"StimFreq": list(args.stim_freq)}}

    evaluation = ds.evaluate_ssvep(params=params_override, show_plots=not args.no_plots)
    summary = _summarize_evaluation(evaluation)
    print("SSVEP evaluation completed. Summary:")
    print(json.dumps(summary, indent=2))

    output = args.output or (args.bin_dir / f"sbids_meta_{args.bin_dir.name}.jsonld")
    sbids_path = ds.to_sbids(output, export_format=args.export_format)
    print(f"Wrote SBIDS JSON-LD to {sbids_path}")

    if not args.no_plots:
        print("Close plot windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
