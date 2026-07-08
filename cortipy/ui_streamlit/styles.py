"""CSS used by the Streamlit frontend."""

GLOBAL_CSS = """
<style>
:root {
    --app-top-padding: 2.35rem;
    --sidebar-top-padding: 1.25rem;
    --sidebar-width: 21rem;
    --pane-bg: #e8eef6;
    --pane-border: #c4cedb;
    --pane-text: #0f172a;
    --pane-muted: #475569;
    --field-bg: #ffffff;
    --field-border: #d7e1ed;
    --field-disabled-text: #64748b;
    --sidebar-bg: #d7e0ea;
    --sidebar-panel-bg: var(--pane-bg);
    --expander-bg: var(--pane-bg);
    --expander-border: var(--pane-border);
    --expander-summary: #d3deea;
    --expander-summary-hover: #c7d4e2;
    --expander-text: var(--pane-text);
    --accent: #0d9488;
}
@media (prefers-color-scheme: dark) {
    :root {
        --sidebar-bg: #d7e0ea;
        --sidebar-panel-bg: var(--pane-bg);
        --expander-bg: var(--pane-bg);
        --expander-border: var(--pane-border);
        --expander-summary: #d3deea;
        --expander-summary-hover: #c7d4e2;
        --expander-text: var(--pane-text);
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
    margin-bottom: 1.15rem;
    padding-bottom: 0.4rem;
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
    margin: -0.25rem -0.45rem 0.85rem -0.45rem;
    padding: 0.58rem 0.65rem;
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
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--pane-border) !important;
    background-color: var(--pane-bg) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: var(--pane-bg) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] p,
[data-testid="stVerticalBlockBorderWrapper"] label,
[data-testid="stVerticalBlockBorderWrapper"] h2,
[data-testid="stVerticalBlockBorderWrapper"] h3,
[data-testid="stVerticalBlockBorderWrapper"] h4 {
    color: var(--pane-text);
}
@media (max-width: 640px) {
    :root {
        --sidebar-width: 85vw;
        --app-top-padding: 1.25rem;
        --sidebar-top-padding: 1rem;
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
    padding: 0.62rem 0.85rem !important;
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
    padding-top: 1rem;
    padding-bottom: 0.9rem;
    color: var(--expander-text);
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input {
    background-color: var(--field-bg) !important;
    color: var(--pane-text) !important;
    border-color: var(--field-border) !important;
}
[data-testid="stTextInput"] input:disabled,
[data-testid="stNumberInput"] input:disabled,
[data-testid="stTextArea"] textarea:disabled,
[data-testid="stDateInput"] input:disabled,
[data-testid="stTimeInput"] input:disabled {
    background-color: var(--field-bg) !important;
    color: var(--field-disabled-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--field-disabled-text) !important;
}
[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background-color: var(--field-bg) !important;
    border-color: var(--field-border) !important;
}
[data-baseweb="select"] input,
[data-baseweb="select"] span,
[data-baseweb="select"] svg {
    color: var(--pane-text) !important;
    fill: var(--pane-text) !important;
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
/* Keep compact fields readable without forcing the whole multi-page UI into a dense grid. */
[data-testid="stMain"] [data-testid="stSelectbox"],
[data-testid="stMain"] [data-testid="stNumberInput"],
[data-testid="stMain"] [data-testid="stTextInput"] {
    max-width: 420px;
}
</style>
"""


def inject_global_styles(st_module) -> None:
    """Render global CSS into a Streamlit page."""
    st_module.markdown(GLOBAL_CSS, unsafe_allow_html=True)
