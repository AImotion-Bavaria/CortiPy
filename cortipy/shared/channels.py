"""Channel-label <-> data-column mapping.

``params["Channels"]`` is a list of dicts (``{"Channel": "Ch 1", "Position": "Oz", ...}``)
whose i-th entry describes the i-th EEG column of ``params["data"]``.  Non-EEG columns
(AUX, trigger) trail the EEG block and are deliberately absent from the list.

Every evaluator used to carry its own label lookup and they disagreed with each other:
some stringified the dicts (never matching), some preferred ``Channel`` over ``Position``
(so anatomical lookups such as "Oz" failed), and each fell back to a different column.
Resolve through this module instead.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

__all__ = [
    "channel_entries",
    "channel_labels",
    "normalize_label",
    "resolve_channel_index",
    "resolve_plot_channel",
]

# Ordered by how well each stands in for an occipital reference site.
OCCIPITAL_PREFERENCE: Tuple[str, ...] = ("Oz", "POz", "O1", "O2", "PO3", "PO4", "Pz")


def normalize_label(label: Any) -> str:
    """Casefold and strip separators so "Ch 1", "ch1" and "CH-1" all compare equal."""
    text = str(label if label is not None else "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def _entry_labels(entry: Any) -> list[str]:
    """All names an entry may legitimately be addressed by (position first)."""
    if isinstance(entry, Mapping):
        keys = ("Position", "Channel", "label", "name")
        return [str(entry[key]) for key in keys if entry.get(key) not in (None, "")]
    if isinstance(entry, (list, tuple)) and entry:
        return [str(entry[0])]
    if entry in (None, ""):
        return []
    return [str(entry)]


def channel_entries(source: Any) -> list[Any]:
    """Montage entries from a params dict, or from a bare channel list.

    Callers hold either the whole params mapping or just ``params["Channels"]``; accept
    both so there is one lookup rather than one per caller.
    """
    if source is None:
        return []
    if isinstance(source, Mapping):
        for key in ("Channels", "ChannelLabels", "ChannelLabelsEEG"):
            entries = source.get(key)
            if entries:
                return list(entries)
        return []
    if isinstance(source, (list, tuple)):
        return list(source)
    return []


def channel_labels(source: Any, count: Optional[int] = None) -> list[str]:
    """Human-readable label per data column, padded with ``Ch N`` past the montage."""
    entries = channel_entries(source)
    labels: list[str] = []
    for idx, entry in enumerate(entries):
        names = _entry_labels(entry)
        labels.append(names[0] if names else f"Ch {idx + 1}")
    if count is not None:
        labels = labels[:count]
        while len(labels) < count:
            labels.append(f"Ch {len(labels) + 1}")
    return labels


def resolve_channel_index(
    source: Any,
    label: Any,
    *,
    count: Optional[int] = None,
) -> Optional[int]:
    """Resolve ``label`` to a 0-based data-column index, or None when absent.

    A positive integer is taken as a 1-based channel number.  Otherwise the label is
    matched against every name an entry answers to (``Position`` and ``Channel``).
    Returns None rather than guessing — callers decide what an unresolvable label means.
    """
    if label is None or (isinstance(label, str) and not label.strip()):
        return None

    if not isinstance(label, str):
        try:
            as_int = int(label)
        except (TypeError, ValueError):
            pass
        else:
            if as_int > 0:
                idx = as_int - 1
                return idx if count is None or idx < count else None
            return None
    else:
        stripped = label.strip()
        if stripped.isdigit():
            idx = int(stripped) - 1
            if idx >= 0:
                return idx if count is None or idx < count else None
            return None

    wanted = normalize_label(label)
    if not wanted:
        return None
    for idx, entry in enumerate(channel_entries(source)):
        if count is not None and idx >= count:
            break
        if any(normalize_label(name) == wanted for name in _entry_labels(entry)):
            return idx
    return None


def resolve_plot_channel(
    source: Any,
    requested: Any,
    count: int,
    *,
    preference: Sequence[str] = OCCIPITAL_PREFERENCE,
    exclude: Sequence[int] = (),
) -> Tuple[int, str, bool]:
    """Pick the column to plot, returning ``(index, label_actually_used, is_exact)``.

    When the requested electrode is not in the montage — Oz on a UNICORN cap, say — fall
    back to the nearest site in ``preference`` and report the substitution, so the caller
    can title the plot with the channel it really drew instead of the one it wished for.

    ``exclude`` holds column indices that must never be auto-selected. Pass the reference
    channel: it is identically zero after referencing, so auto-falling-back onto it (the
    old code always landed on column 0) produced a flat, meaningless spectrum. An
    explicitly requested channel is still honoured, excluded or not — that is the caller's
    decision to make.
    """
    count = max(int(count), 0)
    labels = channel_labels(source, count)
    if count == 0:
        return 0, str(requested or "Ch 1"), False

    banned = {int(i) for i in exclude}

    idx = resolve_channel_index(source, requested, count=count)
    if idx is not None and 0 <= idx < count:
        return idx, labels[idx], True

    for candidate in preference:
        alt = resolve_channel_index(source, candidate, count=count)
        if alt is not None and 0 <= alt < count and alt not in banned:
            return alt, labels[alt], False

    for alt in range(count):
        if alt not in banned:
            return alt, labels[alt], False

    return 0, labels[0], False
