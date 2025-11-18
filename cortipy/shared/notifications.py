"""Helper functions for user feedback (audio/messages)."""

from __future__ import annotations

import sys
import time


def beep() -> None:
    """Emit a short console bell."""
    sys.stdout.write("\a")
    sys.stdout.flush()
    time.sleep(0.05)


def info_start_live() -> None:
    print("Now receiving data...")
    beep()


def info_end_live() -> None:
    print("Done receiving data...")
    beep()
