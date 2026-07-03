"""CSS used by the Streamlit frontend."""

GLOBAL_CSS = """
<style>
:root {
    --app-top-padding: 2.1rem;
    --sidebar-width: 21rem;
    --expander-bg: #f8fafc;
    --expander-border: #e5e7eb;
    --expander-summary: #f1f5f9;
    --expander-summary-hover: #e2e8f0;
    --expander-text: #0f172a;
    --accent: #0d9488;
}
@media (prefers-color-scheme: dark) {
    :root {
        --expander-bg: #0f172a;
        --expander-border: #1f2937;
        --expander-summary: #111827;
        --expander-summary-hover: #152238;
        --expander-text: #e5e7eb;
        --accent: #2dd4bf;
    }
}
[data-testid="stSidebar"] {
    width: var(--sidebar-width) !important;
    min-width: var(--sidebar-width) !important;
}
[data-testid="stSidebar"] > div:first-child {
    width: var(--sidebar-width) !important;
}
[data-testid="stSidebarContent"] {
    padding-top: var(--app-top-padding) !important;
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
    margin-bottom: 0.35rem;
    color: var(--accent);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 8px;
}
@media (max-width: 640px) {
    :root {
        --sidebar-width: 85vw;
        --app-top-padding: 1.25rem;
    }
}
div[data-testid="stExpander"] > details {
    border-radius: 12px;
    border: 1px solid var(--expander-border);
    background-color: var(--expander-bg);
    color: var(--expander-text);
}
div[data-testid="stExpander"] > details > summary {
    background-color: var(--expander-summary);
    color: var(--expander-text);
}
div[data-testid="stExpander"] > details > summary p,
div[data-testid="stExpander"] > details > summary span {
    color: var(--expander-text);
}
div[data-testid="stExpander"] > details > summary:hover {
    background-color: var(--expander-summary-hover);
}
div[data-testid="stExpander"] > details > div[role="group"] {
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
</style>
"""


def inject_global_styles(st_module) -> None:
    """Render global CSS into a Streamlit page."""
    st_module.markdown(GLOBAL_CSS, unsafe_allow_html=True)
