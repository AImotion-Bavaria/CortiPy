"""CSS used by the Streamlit frontend."""

GLOBAL_CSS = """
<style>
:root {
    --app-top-padding: 2.1rem;
    --sidebar-top-padding: 1.25rem;
    --sidebar-width: 21rem;
    --sidebar-bg: #cbd5e1;
    --sidebar-panel-bg: #dbe4ee;
    --expander-bg: #f8fafc;
    --expander-border: #9fb0c2;
    --expander-summary: #b7c4d2;
    --expander-summary-hover: #a7b7c8;
    --expander-text: #0f172a;
    --accent: #0d9488;
}
@media (prefers-color-scheme: dark) {
    :root {
        --sidebar-bg: #020617;
        --sidebar-panel-bg: #0b1220;
        --expander-bg: #0f172a;
        --expander-border: #334155;
        --expander-summary: #111827;
        --expander-summary-hover: #1f2937;
        --expander-text: #e5e7eb;
        --accent: #2dd4bf;
    }
}
section[data-testid="stSidebar"],
[data-testid="stSidebar"] {
    width: var(--sidebar-width) !important;
    min-width: var(--sidebar-width) !important;
    background-color: var(--sidebar-bg) !important;
}
section[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div:first-child {
    width: var(--sidebar-width) !important;
    background-color: var(--sidebar-bg) !important;
}
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    padding-top: var(--sidebar-top-padding) !important;
    background-color: var(--sidebar-bg) !important;
}
[data-testid="stMain"] .block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"] .main .block-container,
.main .block-container {
    padding-top: var(--app-top-padding) !important;
}
[data-testid="stAppViewContainer"] .main h1,
[data-testid="stMain"] h1,
.main h1 {
    font-size: 1.95rem;
    line-height: 1.4;
    padding-top: 0.15rem !important;
    margin-top: 0 !important;
    margin-bottom: 0.9rem;
    padding-bottom: 0.3rem;
    border-bottom: 3px solid var(--accent);
    display: inline-block;
}
[data-testid="stSidebar"] h1 {
    font-size: 1.9rem;
    line-height: 1.2;
    padding-top: 0 !important;
    margin-top: 0 !important;
    margin-bottom: 0.8rem;
}
[data-testid="stSidebar"] h3 {
    font-size: 1rem;
    line-height: 1.25;
    margin: -0.45rem -0.55rem 0.6rem -0.55rem;
    padding: 0.45rem 0.55rem;
    border-radius: 7px;
    background-color: var(--expander-summary);
    color: var(--expander-text);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 8px;
    border-color: var(--expander-border) !important;
    background-color: var(--sidebar-panel-bg) !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: var(--sidebar-panel-bg) !important;
}
.workflow-home {
    border: 1px solid var(--expander-border);
    border-radius: 10px;
    background: linear-gradient(135deg, #e8f1f5 0%, #f8fafc 58%, #fff7ed 100%);
    padding: 1rem;
    margin: 0.15rem 0 1rem 0;
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(18rem, 0.85fr);
    gap: 1rem;
    align-items: stretch;
}
.workflow-kicker {
    color: var(--accent);
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0;
    margin-bottom: 0.25rem;
}
.workflow-title {
    color: var(--expander-text);
    font-size: 1.65rem;
    font-weight: 760;
    line-height: 1.25;
    margin-bottom: 0.35rem;
}
.workflow-copy {
    color: #475569;
    max-width: 58rem;
    line-height: 1.45;
}
.workflow-summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.45rem;
}
.workflow-summary-tile {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.6);
    border-radius: 8px;
    padding: 0.55rem 0.65rem;
}
.workflow-summary-label {
    color: #64748b;
    font-size: 0.68rem;
    font-weight: 760;
    text-transform: uppercase;
    letter-spacing: 0;
}
.workflow-summary-value {
    color: #0f172a;
    font-size: 0.95rem;
    font-weight: 720;
    line-height: 1.25;
    margin-top: 0.15rem;
    overflow-wrap: anywhere;
}
.workflow-lane {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.55rem;
    margin: 0.4rem 0 1rem 0;
}
.workflow-node {
    border: 1px solid var(--expander-border);
    border-left-width: 4px;
    border-radius: 8px;
    background: #f8fafc;
    min-height: 5.3rem;
    padding: 0.55rem 0.6rem;
}
.workflow-node.is-ready {
    border-left-color: #0d9488;
}
.workflow-node.is-needed {
    border-left-color: #f59e0b;
}
.workflow-node.is-waiting {
    border-left-color: #64748b;
}
.workflow-node-num {
    width: 1.65rem;
    height: 1.65rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #e2e8f0;
    color: #0f172a;
    font-weight: 760;
    font-size: 0.72rem;
    margin-bottom: 0.4rem;
}
.workflow-node-title {
    color: #0f172a;
    font-weight: 720;
    line-height: 1.2;
}
.workflow-node-status {
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 0.2rem;
}
.workflow-section-title {
    color: #0f172a;
    font-size: 1.05rem;
    font-weight: 760;
    margin: 0.4rem 0 0.55rem 0;
}
.workflow-action {
    min-height: 6.3rem;
}
.workflow-action-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.45rem;
}
.workflow-index {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 760;
}
.workflow-badge {
    border-radius: 999px;
    padding: 0.12rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 720;
}
.workflow-badge.is-ready {
    background: #ccfbf1;
    color: #115e59;
}
.workflow-badge.is-needed {
    background: #fef3c7;
    color: #92400e;
}
.workflow-badge.is-waiting {
    background: #e2e8f0;
    color: #334155;
}
.workflow-action-title {
    color: #0f172a;
    font-size: 1rem;
    font-weight: 760;
    line-height: 1.25;
}
.workflow-action-copy {
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.35;
    margin-top: 0.25rem;
}
@media (prefers-color-scheme: dark) {
    .workflow-home {
        background: linear-gradient(135deg, #0b1220 0%, #111827 100%);
    }
    .workflow-copy {
        color: #cbd5e1;
    }
    .workflow-summary-tile,
    .workflow-node {
        background: rgba(15, 23, 42, 0.72);
        border-color: #334155;
    }
    .workflow-summary-label,
    .workflow-node-status,
    .workflow-index,
    .workflow-action-copy {
        color: #94a3b8;
    }
    .workflow-summary-value,
    .workflow-node-title,
    .workflow-section-title,
    .workflow-action-title {
        color: #e5e7eb;
    }
    .workflow-node-num,
    .workflow-badge.is-waiting {
        background: #1f2937;
        color: #e5e7eb;
    }
}
@media (max-width: 640px) {
    :root {
        --sidebar-width: 85vw;
        --app-top-padding: 1.25rem;
        --sidebar-top-padding: 1rem;
    }
    .workflow-home,
    .workflow-lane,
    .workflow-summary-grid {
        grid-template-columns: 1fr;
    }
}
div[data-testid="stExpander"] > details,
[data-testid="stExpander"] details {
    border-radius: 12px;
    border: 1px solid var(--expander-border);
    background-color: var(--expander-bg);
    color: var(--expander-text);
}
div[data-testid="stExpander"] > details > summary,
[data-testid="stExpander"] details summary {
    background-color: var(--expander-summary);
    color: var(--expander-text);
    border-radius: 11px 11px 0 0;
}
div[data-testid="stExpander"] > details > summary p,
[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] > details > summary span,
[data-testid="stExpander"] details summary span {
    color: var(--expander-text);
}
div[data-testid="stExpander"] > details > summary:hover,
[data-testid="stExpander"] details summary:hover {
    background-color: var(--expander-summary-hover);
}
div[data-testid="stExpander"] > details > div[role="group"],
[data-testid="stExpander"] details > div[role="group"] {
    padding-top: 0.5rem;
    color: var(--expander-text);
}
button,
[role="button"],
[data-baseweb="select"],
[data-baseweb="select"] *,
[data-testid="stSelectbox"],
[data-testid="stSelectbox"] *,
[data-testid="stMultiSelect"],
[data-testid="stMultiSelect"] *,
[data-testid="stCheckbox"],
[data-testid="stCheckbox"] *,
[data-testid="stRadio"] label,
[data-testid="stRadio"] label *,
[data-testid="stSegmentedControl"],
[data-testid="stSegmentedControl"] *,
[data-testid="stFileUploader"] button,
[data-testid="stDownloadButton"] button {
    cursor: pointer !important;
}
input,
textarea,
[contenteditable="true"] {
    cursor: text !important;
}
/* Space efficiency: short-value inputs (dropdowns / numbers / small text) shouldn't sprawl
   across the whole column. Cap their width in the main content area (sidebar unaffected). */
[data-testid="stMain"] [data-testid="stSelectbox"],
[data-testid="stMain"] [data-testid="stNumberInput"],
[data-testid="stMain"] [data-testid="stTextInput"] {
    max-width: 340px;
}
</style>
"""


def inject_global_styles(st_module) -> None:
    """Render global CSS into a Streamlit page."""
    st_module.markdown(GLOBAL_CSS, unsafe_allow_html=True)
