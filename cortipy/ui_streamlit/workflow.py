"""Workflow dashboard and setup-progress widgets for the Streamlit UI."""

from __future__ import annotations

import html
from typing import Any, Callable, Dict, List, Sequence, Tuple

import streamlit as st

from cortipy.ui_streamlit.constants import DEVICE_EXTRA_LABELS

WORKFLOW_READINESS_ITEMS = (
    ("has_config", "Configuration"),
    ("has_participant", "Participant"),
    ("has_electrodes", "Electrodes"),
    ("is_valid", "Validation"),
    ("has_data", "Recorded data"),
)

PANE_BG = "#e8eef6"
PANE_BORDER = "#c4cedb"
PANE_TEXT = "#0f172a"
PANE_MUTED = "#475569"
INNER_PANE_BG = "#ffffff"
INNER_PANE_BORDER = "#d7e1ed"


def has_active_eeg_channels(params: Dict[str, Any]) -> bool:
    device = str(params.get("Device") or "")
    extras = {name.lower() for name in DEVICE_EXTRA_LABELS.get(device, [])}
    for channel in params.get("Channels", []) or []:
        if not isinstance(channel, dict):
            continue
        name = str(channel.get("Channel") or channel.get("Label") or "").lower()
        if name and name not in extras and bool(channel.get("Active", True)):
            return True
    return False


def render_workflow_progress(
    slot,
    steps: Sequence[Tuple[str, bool]],
    validation_issues: List[str],
) -> None:
    """Top-of-page progress for the staged session configuration.

    ``steps`` comes from the form itself (``session.config_progress_steps``), so the bar and
    the form can never disagree. It used to compute its own checklist and ticked "Sampling
    rate" and "Electrodes" before a device was chosen, because fs carries a schema default
    and the channel table is pre-populated — both were true by construction, so the bar
    claimed progress the operator had not made.
    """
    steps = list(steps)
    if not steps:
        return
    done = sum(1 for _, ok in steps if ok)
    ready = not validation_issues
    with slot:
        caption = (
            "Ready to run - press Start measurement"
            if ready and done == len(steps)
            else "Complete the required fields to run"
        )
        st.progress(done / len(steps), text=f"Setup {done}/{len(steps)} - {caption}")
        cols = st.columns(len(steps))
        for col, (label, ok) in zip(cols, steps):
            marker = "&#10003;" if ok else "&#9675;"
            color = "#0d9488" if ok else "#9ca3af"
            col.markdown(
                (
                    f"<span style='color:{color};font-weight:700;font-size:1.05rem'>{marker}</span> "
                    f"{html.escape(label)}"
                ),
                unsafe_allow_html=True,
            )


def _workflow_summary(params: Dict[str, Any]) -> Dict[str, str]:
    parameters = params.get("Parameters", {}) if isinstance(params, dict) else {}
    participant = (params.get("Metadata", {}) or {}).get("Participant", {}) or {}
    export_container = st.session_state.get("export_container", "JSON-LD")
    if export_container == "SBIDS":
        export_container = "JSON-LD"
    active_channels = sum(
        1
        for channel in params.get("Channels", []) or []
        if isinstance(channel, dict) and bool(channel.get("Active", True))
    )
    recording_time = parameters.get("RecordingTime") or 0
    try:
        recording_label = f"{int(float(recording_time))} s"
    except (TypeError, ValueError):
        recording_label = str(recording_time)
    return {
        "Method": str(params.get("Method") or "Not set"),
        "Device": str(params.get("Device") or "Not set"),
        "Time": recording_label,
        "Channels": str(active_channels or "Not set"),
        "Participant": str(participant.get("Code") or "Not set"),
        "Format": f"{export_container} + {st.session_state.get('export_raw_format', 'Parquet')}",
    }


def _workflow_flags(params: Dict[str, Any], validation_issues: List[str]) -> Dict[str, bool]:
    parameters = params.get("Parameters", {}) if isinstance(params, dict) else {}
    participant = (params.get("Metadata", {}) or {}).get("Participant", {}) or {}
    has_config = bool(params.get("Method") and params.get("Device") and parameters.get("fs"))
    return {
        "has_config": has_config,
        "has_participant": bool(participant.get("Code")),
        "has_electrodes": has_active_eeg_channels(params),
        "is_valid": not validation_issues,
        "has_data": bool(st.session_state.get("last_results") or st.session_state.get("imported_data") is not None),
    }


def _workflow_next_action(params: Dict[str, Any], validation_issues: List[str]) -> tuple[str, str, str, str]:
    flags = _workflow_flags(params, validation_issues)
    if not flags["has_config"] or not flags["has_participant"]:
        return (
            "Complete session setup",
            "Choose method/device, set the recording time, and fill participant details.",
            "Session configuration",
            "workflow_next_session",
        )
    if not flags["has_electrodes"]:
        return (
            "Prepare electrodes",
            "Review active channels, electrode labels, positions, and impedance mapping.",
            "Session configuration",  # the electrode table is a step of that page
            "workflow_next_electrodes",
        )
    if not flags["is_valid"]:
        return (
            "Resolve validation items",
            "Open Preview to see exactly which required fields still block the run.",
            "Preview",
            "workflow_next_preview",
        )
    if not flags["has_data"]:
        return (
            "Connect and record",
            "Use the sidebar to connect the device, then start measurement. Live preview helps verify signal quality.",
            "Live preview",
            "workflow_next_live",
        )
    return (
        "Review and export",
        "Inspect charts or reopen saved sessions, then export the recording from the sidebar.",
        "Charts",
        "workflow_next_charts",
    )


def _workflow_next_status(flags: Dict[str, bool]) -> str:
    if not flags["has_config"] or not flags["has_participant"]:
        return "Needs setup"
    if not flags["has_electrodes"]:
        return "Needs electrodes"
    if not flags["is_valid"]:
        return "Needs validation"
    if not flags["has_data"]:
        return "Ready to acquire"
    return "Ready to review"


def _workflow_status_tone(status: str) -> tuple[str, str, str]:
    normalized = status.lower()
    if "ready" in normalized:
        return "#d1fae5", "#065f46", "#34d399"
    if "need" in normalized or "missing" in normalized:
        return "#fff7ed", "#9a3412", "#fb923c"
    return "#e0f2fe", "#075985", "#38bdf8"


def _inject_workflow_styles() -> None:
    st.markdown(
        f"""
        <style>
        [data-testid="stVerticalBlockBorderWrapper"]:has(.workflow-pane-marker) {{
            background: {PANE_BG} !important;
            border-color: {PANE_BORDER} !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.workflow-pane-marker) > div {{
            background: {PANE_BG} !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.workflow-pane-marker) p,
        [data-testid="stVerticalBlockBorderWrapper"]:has(.workflow-pane-marker) label {{
            color: {PANE_TEXT};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _workflow_pane_marker() -> None:
    st.markdown("<span class='workflow-pane-marker' style='display:none'></span>", unsafe_allow_html=True)


def _render_workflow_banner(
    eyebrow: str,
    title: str,
    body: str,
    status: str,
    *,
    min_height: str = "0",
) -> None:
    bg, fg, border = _workflow_status_tone(status)
    eyebrow_html = (
        f"<div style='color:{fg};font-size:0.76rem;font-weight:800;text-transform:uppercase;'>"
        f"{html.escape(eyebrow)}</div>"
        if eyebrow
        else ""
    )
    st.markdown(
        (
            f"<div style='border:1px solid {border};border-left:6px solid {border};"
            f"border-radius:10px;background:{PANE_BG};padding:1.12rem 1.16rem;margin-bottom:1.1rem;"
            f"min-height:{min_height};'>"
            f"{eyebrow_html}"
            f"<div style='color:{PANE_TEXT};font-size:1.25rem;font-weight:800;line-height:1.25;margin-top:0.15rem;'>"
            f"{html.escape(title)}</div>"
            f"<div style='color:{PANE_MUTED};line-height:1.4;margin-top:0.45rem;'>{html.escape(body)}</div>"
            f"<div style='margin-top:0.72rem;'><span style='display:inline-block;border:1px solid {border};"
            f"border-radius:999px;background:{PANE_BG};color:{border};"
            f"font-size:0.78rem;font-weight:750;padding:0.13rem 0.58rem;'>{html.escape(status)}</span></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_spacer(height: str = "0.45rem") -> None:
    st.markdown(f"<div style='height:{height};'></div>", unsafe_allow_html=True)


def _render_summary_tile(label: str, value: str, status: str) -> None:
    bg, fg, border = _workflow_status_tone(status)
    st.markdown(
        (
            f"<div style='border:1px solid {border};border-radius:9px;background:{INNER_PANE_BG};"
            "padding:0.8rem 0.9rem;min-height:5.35rem;'>"
            f"<div style='color:{border};font-size:0.72rem;font-weight:800;text-transform:uppercase;'>"
            f"{html.escape(label)}</div>"
            f"<div style='color:{PANE_TEXT};font-size:1.35rem;font-weight:760;line-height:1.25;"
            f"margin-top:0.38rem;overflow-wrap:anywhere;'>{html.escape(value)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_check_item(label: str, ok: bool) -> None:
    color = "#0d9488" if ok else "#9ca3af"
    marker = "&#10003;" if ok else "&#9675;"
    st.markdown(
        (
            f"<span style='color:{color};font-weight:800'>{marker}</span> "
            f"<span style='font-size:0.9rem;line-height:1.45'>{html.escape(label)}</span>"
        ),
        unsafe_allow_html=True,
    )


def _render_check_block(checks: List[tuple[str, bool]], *, min_height: str = "5.7rem") -> None:
    items = []
    for label, ok in checks:
        color = "#0d9488" if ok else "#9ca3af"
        marker = "&#10003;" if ok else "&#9675;"
        items.append(
            "<div style='display:flex;align-items:flex-start;gap:0.45rem;'>"
            f"<span style='color:{color};font-weight:850;line-height:1.35;'>{marker}</span>"
            f"<span style='font-size:0.9rem;line-height:1.35;color:{PANE_TEXT};'>{html.escape(label)}</span>"
            "</div>"
        )
    st.markdown(
        (
            f"<div style='min-height:{min_height};display:flex;flex-direction:column;"
            "justify-content:flex-start;gap:0.65rem;margin-top:0.25rem;'>"
            f"{''.join(items)}</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_check_grid(items: List[tuple[str, bool]], columns: int = 2) -> None:
    cells = []
    for label, ok in items:
        color = "#0d9488" if ok else "#9ca3af"
        marker = "&#10003;" if ok else "&#9675;"
        cells.append(
            "<div style='display:flex;align-items:flex-start;gap:0.45rem;min-height:1.7rem;'>"
            f"<span style='color:{color};font-weight:850;line-height:1.3;'>{marker}</span>"
            f"<span style='font-size:0.9rem;line-height:1.3;color:{PANE_TEXT};'>{html.escape(label)}</span>"
            "</div>"
        )
    st.markdown(
        (
            f"<div style='display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));"
            "column-gap:1.1rem;row-gap:0.65rem;margin-top:0.55rem;'>"
            f"{''.join(cells)}</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_action_panel(title: str, body: str, status: str, *, min_height: str = "8.65rem") -> None:
    bg, fg, border = _workflow_status_tone(status)
    st.markdown(
        (
            f"<div style='border-left:7px solid {border};border-radius:10px;background:{PANE_BG};"
            "padding:1.05rem 1.12rem;margin-bottom:0.85rem;"
            f"min-height:{min_height};'>"
            f"<div style='display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;'>"
            "<div>"
            f"<div style='color:{PANE_TEXT};font-size:1.38rem;font-weight:850;line-height:1.18;'>"
            f"{html.escape(title)}</div>"
            f"<div style='color:{PANE_MUTED};font-size:0.98rem;line-height:1.45;margin-top:0.5rem;max-width:58rem;'>"
            f"{html.escape(body)}</div>"
            "</div>"
            f"<span style='display:inline-block;border:1px solid {border};border-radius:999px;"
            f"background:{PANE_BG};color:{border};font-size:0.78rem;font-weight:800;"
            f"padding:0.2rem 0.65rem;white-space:nowrap;'>{html.escape(status)}</span>"
            "</div>"
            "<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.65rem;margin-top:1rem;'>"
            f"<div style='border:1px solid {INNER_PANE_BORDER};border-radius:9px;background:{INNER_PANE_BG};"
            f"padding:0.48rem 0.55rem;color:{PANE_TEXT};font-weight:780;font-size:0.84rem;'>Setup</div>"
            f"<div style='border:1px solid {INNER_PANE_BORDER};border-radius:9px;background:{INNER_PANE_BG};"
            f"padding:0.48rem 0.55rem;color:{PANE_TEXT};font-weight:780;font-size:0.84rem;'>Preview</div>"
            f"<div style='border:1px solid {INNER_PANE_BORDER};border-radius:9px;background:{INNER_PANE_BG};"
            f"padding:0.48rem 0.55rem;color:{PANE_TEXT};font-weight:780;font-size:0.84rem;'>Record</div>"
            f"<div style='border:1px solid {INNER_PANE_BORDER};border-radius:9px;background:{INNER_PANE_BG};"
            f"padding:0.48rem 0.55rem;color:{PANE_TEXT};font-weight:780;font-size:0.84rem;'>Export</div>"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def _render_readiness_panel(completed: int, total: int, flags: Dict[str, bool], status: str) -> None:
    bg, fg, border = _workflow_status_tone(status)
    st.markdown(
        (
            f"<div style='border-left:7px solid {border};border-radius:10px;background:{PANE_BG};"
            "padding:1rem 1.08rem;margin-bottom:0.85rem;min-height:4.15rem;'>"
            "<div style='display:flex;justify-content:space-between;align-items:baseline;gap:1rem;'>"
            f"<div style='color:{PANE_TEXT};font-size:1.05rem;font-weight:820;'>Session readiness</div>"
            f"<div style='color:{border};font-size:0.9rem;font-weight:820;'>{completed}/{total}</div>"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )
    st.progress(completed / total, text=f"{completed}/{total} ready")
    _render_spacer("0.35rem")
    _render_check_grid([(label, flags[key]) for key, label in WORKFLOW_READINESS_ITEMS], columns=3)


def _render_session_panel(summary: Dict[str, str], summary_status: Dict[str, str]) -> None:
    cells = []
    for label, value in summary.items():
        bg, fg, border = _workflow_status_tone(summary_status[label])
        cells.append(
            f"<div style='border:1px solid {border};border-radius:10px;background:{INNER_PANE_BG};"
            "padding:0.78rem 0.86rem;min-height:4.9rem;'>"
            f"<div style='color:{border};font-size:0.72rem;font-weight:850;text-transform:uppercase;'>"
            f"{html.escape(label)}</div>"
            f"<div style='color:{PANE_TEXT};font-size:1.32rem;font-weight:800;line-height:1.2;"
            f"margin-top:0.38rem;overflow-wrap:anywhere;'>{html.escape(value)}</div>"
            "</div>"
        )
    st.markdown(
        (
            f"<div style='border:1px solid {PANE_BORDER};border-radius:12px;background:{PANE_BG};"
            "padding:1rem 1.05rem;margin-top:0.4rem;'>"
            "<div style='display:flex;justify-content:space-between;gap:1rem;align-items:baseline;margin-bottom:0.8rem;'>"
            f"<div style='color:{PANE_TEXT};font-size:1.02rem;font-weight:820;'>Session at a glance</div>"
            f"<div style='color:{PANE_MUTED};font-size:0.78rem;font-weight:760;text-transform:uppercase;'>"
            "Current editor state</div></div>"
            "<div style='display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:0.7rem;'>"
            f"{''.join(cells)}</div></div>"
        ),
        unsafe_allow_html=True,
    )


def _render_lane_header(index: int, title: str, status: str, body: str) -> None:
    bg, fg, border = _workflow_status_tone(status)
    st.markdown(
        (
            f"<div style='border-left:6px solid {border};border-radius:10px;background:{PANE_BG};"
            "padding:0.9rem 0.95rem;margin-bottom:0.82rem;min-height:7rem;'>"
            "<div style='display:flex;justify-content:space-between;gap:0.75rem;align-items:center;'>"
            f"<div style='color:{border};font-size:0.78rem;font-weight:850;'>Step {index:02d}</div>"
            f"<span style='border:1px solid {border};border-radius:999px;background:{PANE_BG};"
            f"color:{border};font-size:0.72rem;font-weight:780;padding:0.12rem 0.5rem;'>{html.escape(status)}</span>"
            "</div>"
            f"<div style='color:{PANE_TEXT};font-size:1.2rem;font-weight:850;line-height:1.2;margin-top:0.48rem;'>"
            f"{html.escape(title)}</div>"
            f"<div style='color:{PANE_MUTED};font-size:0.9rem;line-height:1.35;margin-top:0.45rem;'>"
            f"{html.escape(body)}</div></div>"
        ),
        unsafe_allow_html=True,
    )


def _phase_card(
    index: int,
    title: str,
    status: str,
    body: str,
    checks: List[tuple[str, bool]],
    actions: List[tuple[str, str]],
    key_prefix: str,
    set_active_view: Callable[[str], None],
) -> None:
    with st.container(border=True):
        _workflow_pane_marker()
        _render_lane_header(index, title, status, body)
        _render_check_block(checks, min_height="5.2rem")
        _render_spacer("0.65rem")
        action_cols = st.columns(len(actions))
        for button_idx, (label, page) in enumerate(actions):
            action_cols[button_idx].button(
                label,
                width="stretch",
                key=f"{key_prefix}_{button_idx}",
                on_click=set_active_view,
                args=(page,),
            )


def render_workflow_page(
    params: Dict[str, Any],
    validation_issues: List[str],
    set_active_view: Callable[[str], None],
) -> None:
    """Show a home page with workflow shortcuts for configuring, recording, and reviewing sessions."""
    _inject_workflow_styles()
    summary = _workflow_summary(params)
    flags = _workflow_flags(params, validation_issues)
    next_title, next_body, next_page, next_key = _workflow_next_action(params, validation_issues)
    next_status = _workflow_next_status(flags)
    completed = sum(1 for key, _ in WORKFLOW_READINESS_ITEMS if flags[key])
    total = len(WORKFLOW_READINESS_ITEMS)
    secondary_page = "Preview" if next_page != "Preview" else "Session configuration"

    top_left, top_right = st.columns([1.45, 1])
    with top_left:
        with st.container(border=True):
            _workflow_pane_marker()
            _render_action_panel(next_title, next_body, next_status)
            action_cols = st.columns([1.15, 0.85])
            action_cols[0].button(
                f"Open {next_page}",
                type="primary",
                width="stretch",
                key=next_key,
                on_click=set_active_view,
                args=(next_page,),
            )
            action_cols[1].button(
                f"Open {secondary_page}",
                width="stretch",
                key=f"{next_key}_secondary",
                on_click=set_active_view,
                args=(secondary_page,),
            )
    with top_right:
        with st.container(border=True):
            _workflow_pane_marker()
            _render_readiness_panel(completed, total, flags, next_status)

    summary_status = {
        "Method": "Ready" if summary["Method"] != "Not set" else "Needs setup",
        "Device": "Ready" if summary["Device"] != "Not set" else "Needs setup",
        "Time": "Ready" if summary["Time"] != "0 s" else "Needs setup",
        "Channels": "Ready" if summary["Channels"] != "Not set" else "Needs electrodes",
        "Participant": "Ready" if summary["Participant"] != "Not set" else "Needs setup",
        "Format": "Available",
    }
    _render_session_panel(summary, summary_status)

    _render_spacer("0.85rem")
    phase_cols = st.columns(3)
    with phase_cols[0]:
        prepare_status = "Ready" if flags["has_config"] and flags["has_participant"] and flags["has_electrodes"] else "Needs setup"
        _phase_card(
            1,
            "Prepare",
            prepare_status,
            "Define the run and make the channel table physically meaningful before acquisition.",
            [
                ("Method/device/fs selected", flags["has_config"]),
                ("Participant code entered", flags["has_participant"]),
                ("Active electrodes available", flags["has_electrodes"]),
            ],
            [("Configure", "Session configuration")],
            "workflow_prepare",
            set_active_view,
        )
    with phase_cols[1]:
        acquire_status = "Ready" if flags["is_valid"] and flags["has_electrodes"] else "Needs validation"
        _phase_card(
            2,
            "Acquire",
            acquire_status,
            "Check signal behavior, validate the params payload, then use the sidebar run controls.",
            [
                ("Electrodes ready", flags["has_electrodes"]),
                ("No validation blockers", flags["is_valid"]),
                ("Run controls stay in sidebar", True),
            ],
            [("Live preview", "Live preview"), ("Validate", "Preview")],
            "workflow_acquire",
            set_active_view,
        )
    with phase_cols[2]:
        review_status = "Ready" if flags["has_data"] else "After run/import"
        _phase_card(
            3,
            "Review",
            review_status,
            "Inspect plots, compare saved sessions, and export the selected container/format.",
            [
                ("Recording/import available", flags["has_data"]),
                (f"Export: {summary['Format']}", True),
                ("Saved sessions accessible", True),
            ],
            [("Charts", "Charts"), ("Sessions", "Saved sessions")],
            "workflow_review",
            set_active_view,
        )
