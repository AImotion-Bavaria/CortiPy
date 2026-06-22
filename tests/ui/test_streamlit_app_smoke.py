from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

from cortipy.ui_streamlit.styles import GLOBAL_CSS


ROOT = Path(__file__).resolve().parents[2]


def load_streamlit_app():
    app_path = ROOT / "apps" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app_smoke", app_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_streamlit_app_imports() -> None:
    module = load_streamlit_app()

    assert callable(module.main)
    assert module.GENERAL_SCHEMA
    assert module.METHOD_SCHEMAS
    assert module.VIEW_OPTIONS == [
        "Session configuration",
        "Electrodes",
        "Live preview",
        "Preview",
        "Charts",
        "Saved sessions",
    ]


def test_global_css_keeps_pointer_cursor_override() -> None:
    assert "[data-baseweb=\"select\"]" in GLOBAL_CSS
    assert "[data-testid=\"stSegmentedControl\"]" in GLOBAL_CSS
    assert "--sidebar-width: 21rem" in GLOBAL_CSS
    assert "--app-top-padding: 1.9rem" in GLOBAL_CSS
    assert "[data-testid=\"stMainBlockContainer\"]" in GLOBAL_CSS
    assert "cursor: pointer !important" in GLOBAL_CSS
    assert "cursor: text !important" in GLOBAL_CSS


def test_ui_dependencies_are_bounded() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ui_deps = pyproject["project"]["optional-dependencies"]["ui"]

    assert "streamlit>=1.38,<1.50" in ui_deps
    assert "plotly>=5.24,<6" in ui_deps
    assert "streamlit-plotly-events==0.0.6" in ui_deps


def test_impedance_mapping_uses_measured_positive_values() -> None:
    module = load_streamlit_app()
    rows = [
        {"Channel": "GND", "Impedance": 0.0},
        {"Channel": "Ch 1", "Impedance": 0.0},
        {"Channel": "Ch 2", "Impedance": 0.0},
    ]

    assert module._has_measured_impedance([0.0, -1.0, 0.0]) is False
    assert module._has_measured_impedance([0.0, -1.0, 12000.0]) is True
    assert module._impedance_range_kohm([0.0, -1.0, 12000.0, 5000.0]) == (5.0, 12.0)
    assert module._impedance_range_kohm([0.0, -1.0]) is None

    mapped = module._map_impedances_to_channels(rows, [5000.0, 6000.0, 12000.0, -1.0])

    assert mapped[0]["Impedance"] == 5.0
    assert mapped[1]["Impedance"] == 12.0
    assert mapped[2]["Impedance"] == 0.0
