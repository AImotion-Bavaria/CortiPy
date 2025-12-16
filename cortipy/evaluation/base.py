"""Base class for evaluation helpers and shared plotting utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt

from cortipy.core.context import ModuleContext

__all__ = ["EvaluatorBase", "save_new_figures"]


class EvaluatorBase(ABC):
    """Shared protocol for all evaluators."""

    @abstractmethod
    def evaluate(self, context: ModuleContext) -> None:
        """Mutate ``context.params`` with evaluation results."""


def save_new_figures(
    before_figs: Set[int],
    save_dir: Path,
    prefix: str,
    *,
    close: bool = True,
) -> Dict[str, List[str]]:
    """Persist matplotlib figures created after ``before_figs`` were present."""
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, List[str]] = {}
    new_figs = [num for num in plt.get_fignums() if num not in before_figs]
    counts: Dict[str, int] = {}
    for num in new_figs:
        fig = plt.figure(num)
        kind = _classify_figure(fig)
        title_slug = _figure_title_slug(fig)
        stem_base = f"{prefix}_{kind}"
        if title_slug:
            stem_base = f"{stem_base}_{title_slug}"
        counts[stem_base] = counts.get(stem_base, 0) + 1
        suffix = f"_{counts[stem_base]}" if counts[stem_base] > 1 else ""
        stem = f"{stem_base}{suffix}"
        png_path = save_dir / f"{stem}.png"
        pdf_path = save_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        saved.setdefault(kind, []).append(str(png_path))
        if close:
            plt.close(fig)
    return saved


def _classify_figure(fig: plt.Figure) -> str:
    titles: list[str] = []
    if getattr(fig, "_suptitle", None) is not None:
        titles.append(fig._suptitle.get_text())
    for ax in fig.axes:
        try:
            titles.append(ax.get_title())
        except Exception:
            continue
    text = " ".join(titles).lower()
    if "topomap" in text or "topography" in text:
        return "topomap"
    if "psd" in text or "spectrum" in text or "fft" in text:
        return "psd"
    if "trace" in text or "erp" in text or "evoked" in text or "average" in text:
        return "trace"
    return "figure"


def _figure_title_slug(fig: plt.Figure, max_len: int = 60) -> str:
    titles: list[str] = []
    if getattr(fig, "_suptitle", None) is not None:
        titles.append(fig._suptitle.get_text())
    for ax in fig.axes:
        try:
            t = ax.get_title()
            if t:
                titles.append(t)
                break
        except Exception:
            continue
    if not titles:
        return ""
    text = titles[0]
    return _slugify(text, max_len=max_len)


def _slugify(text: str, max_len: int = 60) -> str:
    """Create a filesystem-friendly slug from a title."""
    allowed = []
    for ch in text:
        if ch.isalnum():
            allowed.append(ch.lower())
        elif ch in {" ", "-", "_", "/"}:
            allowed.append("-")
    slug = "".join(allowed)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug
