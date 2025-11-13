"""Namespace package exposing the top-level ``shared`` module exports."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
__path__ = [str(_REPO_ROOT / "shared")]

_shared = import_module("shared")
__all__ = getattr(_shared, "__all__", [])
globals().update({name: getattr(_shared, name) for name in __all__})
