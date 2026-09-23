"""Signal units at the MNE boundary.

CortiPy carries EEG in **microvolts** everywhere: the device adapters emit µV (UNICORN
applies its LSB scale, ActiCHamp applies the per-channel resolution), and every evaluator,
plot axis and stored ``params["data"]`` assumes µV.

MNE's contract is the opposite — SI units, so EEG data must be **volts**.  Every
``mne.io.RawArray`` used to be built straight from the µV array, which stored a 50 µV
signal as 50 V.  That 1e6 error flowed into every BIDS/EDF/Parquet export.

Convert at the boundary and only at the boundary: :func:`to_volts` going into MNE,
:func:`to_microvolts` coming back out.  Non-voltage channels (``stim``, ``misc``) are never
scaled — a trigger of 1.0 must stay 1.0.

Reading older files
-------------------
CortiPy's own earlier exports hold µV magnitudes in a container that *claims* volts.
Scaling those by 1e6 on read would be just as wrong in the other direction.  So the reader
takes a hint (:func:`unit_hint_from_params`, written into new exports as ``SignalUnit``) and
falls back to :func:`looks_like_microvolts`, which asks whether the magnitudes are
physically possible for EEG expressed in volts.  Scalp EEG is well under 1 mV; anything
above that is µV data wearing a volt label.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DEFAULT_SIGNAL_UNIT",
    "MICROVOLTS_PER_VOLT",
    "VOLT_CHANNEL_TYPES",
    "looks_like_microvolts",
    "to_microvolts",
    "to_volts",
    "unit_hint_from_params",
]

MICROVOLTS_PER_VOLT = 1e6

# Channel types MNE stores in volts. `stim`, `misc`, `resp` etc. carry no voltage.
VOLT_CHANNEL_TYPES = frozenset(
    {"eeg", "eog", "ecg", "emg", "seeg", "ecog", "dbs", "bio", "ref_meg"}
)

DEFAULT_SIGNAL_UNIT = "uV"

# Above this (in volts) the data cannot be scalp EEG, so it must already be microvolts.
# 1 mV is ~20x the largest plausible EEG excursion, so this does not fire on real volt data.
_IMPLAUSIBLE_VOLTS = 1e-3


def _volt_mask(ch_types: Optional[Sequence[str]], n_channels: int) -> np.ndarray:
    """Boolean mask over channels selecting those measured in volts."""
    if not ch_types:
        return np.ones(n_channels, dtype=bool)
    mask = np.zeros(n_channels, dtype=bool)
    for idx in range(min(n_channels, len(ch_types))):
        mask[idx] = str(ch_types[idx]).strip().lower() in VOLT_CHANNEL_TYPES
    # Channels beyond the supplied types default to voltage.
    if len(ch_types) < n_channels:
        mask[len(ch_types) :] = True
    return mask


def _scaled(data: np.ndarray, ch_types: Optional[Sequence[str]], factor: float) -> np.ndarray:
    """Scale voltage channels of a (channels, samples) array by ``factor``."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or array.size == 0:
        return array.copy()
    mask = _volt_mask(ch_types, array.shape[0])
    out = array.copy()
    if mask.any():
        out[mask] = out[mask] * factor
    return out


def to_volts(data: np.ndarray, ch_types: Optional[Sequence[str]] = None) -> np.ndarray:
    """µV -> V for voltage channels of a (channels, samples) array. Use when entering MNE."""
    return _scaled(data, ch_types, 1.0 / MICROVOLTS_PER_VOLT)


def to_microvolts(data: np.ndarray, ch_types: Optional[Sequence[str]] = None) -> np.ndarray:
    """V -> µV for voltage channels of a (channels, samples) array. Use when leaving MNE."""
    return _scaled(data, ch_types, MICROVOLTS_PER_VOLT)


def looks_like_microvolts(
    data: np.ndarray, ch_types: Optional[Sequence[str]] = None
) -> bool:
    """True when a supposedly-volt array holds magnitudes only µV data could have.

    Guards against double-scaling CortiPy's own pre-fix exports, which wrote µV numbers
    into a volt-typed container.
    """
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or array.size == 0:
        return False
    mask = _volt_mask(ch_types, array.shape[0])
    if not mask.any():
        return False
    signal = array[mask]
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return False
    return float(np.nanmax(np.abs(finite))) > _IMPLAUSIBLE_VOLTS


def unit_hint_from_params(params: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Read the recorded ``SignalUnit`` ("uV" / "V"), or None when the export predates it."""
    if not isinstance(params, Mapping):
        return None
    block = params.get("Parameters") if isinstance(params.get("Parameters"), Mapping) else params
    value = block.get("SignalUnit") if isinstance(block, Mapping) else None
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"uv", "µv", "microvolt", "microvolts"}:
        return "uV"
    if text in {"v", "volt", "volts"}:
        return "V"
    return None
