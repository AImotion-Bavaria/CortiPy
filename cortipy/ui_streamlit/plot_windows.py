"""Pop-out plot windows for the Streamlit UI.

The Streamlit page stays as a launcher/control surface while plots render in
separate browser windows. Plotly windows are editable and include export tools;
Matplotlib figures fall back to a static PNG window.
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import webbrowser
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

try:  # pragma: no cover - optional UI dependency
    import plotly.graph_objects as go
    import plotly.io as pio
except Exception:  # pragma: no cover
    go = None
    pio = None


WINDOW_DIR = Path.cwd() / ".plot_windows"


def safe_window_key(value: Any, fallback: str = "plot") -> str:
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("._-") or fallback


def _plotly_config(filename: str) -> dict[str, Any]:
    return {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "editable": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": 900,
            "width": 1400,
            "scale": 2,
        },
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawrect",
            "drawcircle",
            "eraseshape",
        ],
    }


def plotly_window_html(
    fig: "go.Figure",
    title: str,
    *,
    auto_refresh: bool = False,
    refresh_seconds: float = 1.0,
) -> str:
    if pio is None:
        return image_window_html(None, title)

    filename = safe_window_key(title)
    plot_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config=_plotly_config(filename),
    )
    refresh = (
        f"<meta http-equiv='refresh' content='{max(0.2, float(refresh_seconds)):.2f}'>"
        if auto_refresh
        else ""
    )
    title_html = html.escape(title)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh}
  <title>{title_html}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #17202a;
      background: #f7f9fb;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 10px 14px;
      border-bottom: 1px solid #d9e1e8;
      background: rgba(255, 255, 255, 0.96);
    }}
    .toolbar strong {{ margin-right: auto; }}
    button {{
      border: 1px solid #9aa9b5;
      border-radius: 6px;
      background: #ffffff;
      color: #17202a;
      padding: 6px 10px;
      font-size: 13px;
      cursor: pointer;
    }}
    button:hover {{ background: #eef3f7; }}
    .plot-wrap {{
      height: calc(100vh - 52px);
      min-height: 520px;
      padding: 8px 10px 14px;
      box-sizing: border-box;
    }}
    .plotly-graph-div {{ height: 100% !important; }}
  </style>
</head>
<body>
  <div class="toolbar">
    <strong>{title_html}</strong>
    <button onclick="downloadImage('png')">PNG</button>
    <button onclick="downloadImage('svg')">SVG</button>
    <button onclick="downloadCsv()">CSV</button>
    <button onclick="downloadHtml()">HTML</button>
  </div>
  <div class="plot-wrap">{plot_html}</div>
  <script>
    function graphDiv() {{
      return document.querySelector('.plotly-graph-div');
    }}
    function downloadImage(format) {{
      const gd = graphDiv();
      if (!gd || !window.Plotly) return;
      Plotly.downloadImage(gd, {{
        format: format,
        filename: {json.dumps(filename)},
        width: 1600,
        height: 900,
        scale: format === 'png' ? 2 : 1
      }});
    }}
    function downloadCsv() {{
      const gd = graphDiv();
      if (!gd || !gd.data) return;
      const rows = [['trace', 'x', 'y']];
      gd.data.forEach((trace, idx) => {{
        const name = trace.name || ('trace_' + (idx + 1));
        const xs = trace.x || [];
        const ys = trace.y || [];
        const n = Math.max(xs.length || 0, ys.length || 0);
        for (let i = 0; i < n; i++) {{
          rows.push([name, xs[i] ?? '', ys[i] ?? '']);
        }}
      }});
      const csv = rows.map(row => row.map(cell => {{
        const text = String(cell);
        return /[",\\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
      }}).join(',')).join('\\n');
      const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8' }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = {json.dumps(filename + ".csv")};
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 500);
    }}
    function downloadHtml() {{
      const blob = new Blob(['<!doctype html>\\n' + document.documentElement.outerHTML], {{
        type: 'text/html;charset=utf-8'
      }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = {json.dumps(filename + ".html")};
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 500);
    }}
  </script>
</body>
</html>"""


def image_window_html(fig: Optional[plt.Figure], title: str) -> str:
    title_html = html.escape(title)
    if fig is None:
        image_data = ""
    else:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
        image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    filename = safe_window_key(title) + ".png"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_html}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f9fb;
      color: #17202a;
    }}
    .toolbar {{
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 10px 14px;
      border-bottom: 1px solid #d9e1e8;
      background: #ffffff;
    }}
    .toolbar strong {{ margin-right: auto; }}
    a {{
      border: 1px solid #9aa9b5;
      border-radius: 6px;
      background: #ffffff;
      color: #17202a;
      padding: 6px 10px;
      font-size: 13px;
      text-decoration: none;
    }}
    .image-wrap {{ padding: 14px; }}
    img {{
      max-width: 100%;
      height: auto;
      background: #ffffff;
      border: 1px solid #d9e1e8;
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <strong>{title_html}</strong>
    <a download="{html.escape(filename)}" href="data:image/png;base64,{image_data}">PNG</a>
  </div>
  <div class="image-wrap"><img src="data:image/png;base64,{image_data}" alt="{title_html}"></div>
</body>
</html>"""


def _launcher_html(title: str, window_html: str, button_label: Optional[str] = None) -> str:
    encoded = base64.b64encode(window_html.encode("utf-8")).decode("ascii")
    label = html.escape(button_label or "Open plot window")
    title_text = html.escape(title)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #17202a;
      background: transparent;
    }}
    .launcher {{
      display: flex;
      gap: 10px;
      align-items: center;
      min-height: 38px;
    }}
    button {{
      border: 1px solid #8ea2b0;
      border-radius: 6px;
      background: #ffffff;
      color: #17202a;
      padding: 7px 11px;
      font-size: 13px;
      cursor: pointer;
    }}
    button:hover {{ background: #eef3f7; }}
    span {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="launcher">
    <button onclick="openPlot()">{label}</button>
    <span>{title_text}</span>
  </div>
  <script>
    const payload = {json.dumps(encoded)};
    function openPlot() {{
      const html = new TextDecoder().decode(Uint8Array.from(atob(payload), c => c.charCodeAt(0)));
      const w = window.open('', '_blank', 'popup=yes,width=1280,height=820,resizable=yes,scrollbars=yes');
      if (!w) return;
      w.document.open();
      w.document.write(html);
      w.document.close();
    }}
  </script>
</body>
</html>"""


def _target_context(target: Any):
    if target is None:
        return nullcontext()
    container = getattr(target, "container", None)
    if callable(container):
        return container()
    return nullcontext()


def render_plotly_window_launcher(
    fig: "go.Figure",
    title: str,
    key: str,
    *,
    target: Any = None,
    button_label: Optional[str] = None,
) -> None:
    if go is None or pio is None:
        return
    window_html = plotly_window_html(fig, title)
    launcher = _launcher_html(title, window_html, button_label)
    with _target_context(target):
        components.html(launcher, height=42)


def render_matplotlib_window_launcher(
    fig: plt.Figure,
    title: str,
    key: str,
    *,
    target: Any = None,
    button_label: Optional[str] = None,
) -> None:
    window_html = image_window_html(fig, title)
    launcher = _launcher_html(title, window_html, button_label)
    with _target_context(target):
        components.html(launcher, height=42)


def write_plotly_window(
    fig: "go.Figure",
    title: str,
    key: str,
    *,
    auto_refresh: bool = False,
    refresh_seconds: float = 1.0,
) -> Path:
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    path = WINDOW_DIR / f"{safe_window_key(key)}.html"
    path.write_text(
        plotly_window_html(fig, title, auto_refresh=auto_refresh, refresh_seconds=refresh_seconds),
        encoding="utf-8",
    )
    return path


def write_matplotlib_window(fig: plt.Figure, title: str, key: str) -> Path:
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    path = WINDOW_DIR / f"{safe_window_key(key)}.html"
    path.write_text(image_window_html(fig, title), encoding="utf-8")
    return path


def open_window_once(path: Path, key: str) -> None:
    state_key = f"_plot_window_opened_{safe_window_key(key)}"
    if st.session_state.get(state_key):
        return
    try:
        webbrowser.open(path.resolve().as_uri(), new=1)
        st.session_state[state_key] = True
    except Exception:
        st.session_state[state_key] = False
