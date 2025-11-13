"""Namespace package exposing the top-level ``ui`` directory."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
__path__ = [str(_REPO_ROOT / "ui")]
