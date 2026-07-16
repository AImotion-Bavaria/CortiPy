"""Launch the Streamlit UI safely on macOS.

    python scripts/run_ui.py            # -> http://localhost:8501
    python scripts/run_ui.py --port 8600

Why this exists
---------------
numpy 1.26.x runs a subprocess at *import* of ``numpy.testing`` to probe CPU SVE support
(``_SUPPORTS_SVE = check_support_sve()`` calls ``subprocess.run('lscpu')``). ``bids`` is
imported lazily during a Streamlit render, i.e. inside a worker thread, after tornado has
already made the process fork-unsafe on macOS — so that fork segfaults and the page goes
blank. Importing ``numpy.testing`` here, before Streamlit loads, runs the probe once in a
clean process state and caches the result, so the later lazy import is a no-op.

Harmless everywhere else: on Windows/Linux there is no fork-safety problem, and the import
is cheap once cached. Plain ``streamlit run apps/streamlit_app.py`` still works on those
platforms.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Must happen before Streamlit (and therefore tornado) is imported.
import numpy.testing  # noqa: F401  # runs the SVE probe once, in the main thread

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "streamlit_app.py"


def main() -> None:
    port = "8501"
    args = sys.argv[1:]
    if "--port" in args:
        port = args[args.index("--port") + 1]

    from streamlit.web import cli as st_cli

    sys.argv = [
        "streamlit", "run", str(APP),
        "--server.port", port,
        "--browser.gatherUsageStats", "false",
    ]
    st_cli.main()


if __name__ == "__main__":
    main()
