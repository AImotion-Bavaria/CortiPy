"""Bump the project version in a single place and propagate it to metadata files.

Usage:
    python scripts/bump_version.py --part patch
    python scripts/bump_version.py --new-version 0.2.0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT_FILE = ROOT / "cortipy" / "__init__.py"
CITATION_FILE = ROOT / "CITATION.cff"


def _read_current_version() -> str:
    text = INIT_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"(?P<ver>\d+\.\d+\.\d+)"', text)
    if not match:
        raise ValueError("Could not find __version__ in cortipy/__init__.py")
    return match.group("ver")


def _write_version(path: Path, pattern: str, version: str) -> None:
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(pattern, version, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected to replace version once in {path}, replaced {count} times")
    path.write_text(new_text, encoding="utf-8")


def _increment_version(current: str, part: str) -> str:
    major, minor, patch = [int(x) for x in current.split(".")]
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown part '{part}'")
    return f"{major}.{minor}.{patch}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump CortiPy version (semantic versioning).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--part", choices=["major", "minor", "patch"], help="Which part of the version to increment.")
    group.add_argument("--new-version", help="Set an explicit semantic version (e.g., 1.2.3).")
    args = parser.parse_args()

    current = _read_current_version()
    if args.new_version:
        if not re.fullmatch(r"\d+\.\d+\.\d+", args.new_version):
            raise ValueError("Version must follow semantic versioning (MAJOR.MINOR.PATCH)")
        new_version = args.new_version
    else:
        new_version = _increment_version(current, args.part)

    if new_version == current:
        print(f"Version unchanged: {current}")
        return

    print(f"Bumping version: {current} -> {new_version}")
    _write_version(INIT_FILE, r'__version__\s*=\s*"\d+\.\d+\.\d+"', f'__version__ = "{new_version}"')
    if CITATION_FILE.exists():
        _write_version(CITATION_FILE, r'version:\s*"\d+\.\d+\.\d+"', f'version: \"{new_version}\"')
    print("Updated cortipy/__init__.py and CITATION.cff")


if __name__ == "__main__":
    main()
