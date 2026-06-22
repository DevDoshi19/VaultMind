"""
streamlit_app.py  —  VaultMind Phase 11 Frontend (Premium Redesign)
Run:  streamlit run streamlit_app.py
"""

import time
from base64 import b64encode
from html import escape
from pathlib import Path
from textwrap import dedent

import streamlit as st
import httpx
import os

ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PNG = ASSETS_DIR / "logo.png"
LOGO_DATA_URI = f"data:image/png;base64,{b64encode(LOGO_PNG.read_bytes()).decode('ascii')}"
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="VaultMind",
    page_icon=str(LOGO_PNG),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --vm-bg: #0A0B0F;
    --vm-bg-soft: #0F1118;
    --vm-panel: #111318;
    --vm-panel-soft: #161923;
    --vm-border: #1C1E27;
    --vm-border-strong: #2A2E3D;
    --vm-text: #F5F7FB;
    --vm-text-soft: #D7DBE7;
    --vm-muted: #9AA3B8;
    --vm-subtle: #6E7588;
    --vm-purple: #8B5CF6;
    --vm-purple-soft: #C4B5FD;
    --vm-red: #EF4444;
    --vm-green: #10B981;
    --vm-amber: #F59E0B;
}

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background:
        radial-gradient(circle at top center, rgba(139, 92, 246, 0.14), transparent 34%),
        var(--vm-bg);
    color: var(--vm-text);
    -webkit-font-smoothing: antialiased;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    right: 0.65rem !important;
    top: 0.55rem !important;
    background: transparent !important;
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 8.5rem !important;
    max-width: 920px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E0F14 0%, #0B0C11 100%) !important;
    border-right: 1px solid var(--vm-border);
}

section[data-testid="stSidebar"] > div:first-child {
    padding: 1.25rem 1.25rem 1.5rem;
}

[data-testid="collapsedControl"] {
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
    position: fixed !important;
    top: 1rem;
    left: 1rem;
    z-index: 1002;
}

[data-testid="collapsedControl"] button,
button[kind="header"] {
    width: 2.65rem;
    height: 2.65rem;
    border-radius: 14px !important;
    background: rgba(17, 19, 24, 0.96) !important;
    border: 1px solid rgba(139, 92, 246, 0.28) !important;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.32) !important;
}

[data-testid="collapsedControl"] button:hover,
button[kind="header"]:hover {
    border-color: rgba(196, 181, 253, 0.5) !important;
    background: rgba(23, 25, 34, 0.98) !important;
}

button[kind="header"] svg {
    color: #D8CCFF !important;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 0 !important;
    box-shadow: none !important;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
    display: none !important;
}

div[data-testid="stBottomBlockContainer"] {
    background: linear-gradient(180deg, rgba(10, 11, 15, 0) 0%, rgba(10, 11, 15, 0.92) 28%, rgba(10, 11, 15, 1) 100%) !important;
    padding: 1rem 0 1.35rem !important;
}

div[data-testid="stBottomBlockContainer"] > div {
    max-width: min(920px, calc(100vw - 2rem));
    margin: 0 auto;
}

@media (min-width: 1100px) {
    div[data-testid="stBottomBlockContainer"] > div {
        max-width: min(920px, calc(100vw - 22rem));
    }
}

[data-testid="stChatInput"] {
    background: rgba(18, 20, 28, 0.96) !important;
    border: 1px solid var(--vm-border) !important;
    border-radius: 18px !important;
    padding: 0.45rem 0.7rem !important;
    min-height: 4.1rem;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--vm-purple) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.12), 0 22px 48px rgba(0, 0, 0, 0.28) !important;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div:focus-within {
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: var(--vm-text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.97rem !important;
    line-height: 1.65 !important;
    min-height: 1.85rem !important;
    max-height: 13rem !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--vm-subtle) !important;
}

[data-testid="stChatInput"] button {
    background: rgba(139, 92, 246, 0.16) !important;
    border: 1px solid rgba(139, 92, 246, 0.24) !important;
    border-radius: 12px !important;
    color: var(--vm-purple-soft) !important;
}

[data-testid="stChatInput"] button:hover {
    background: rgba(139, 92, 246, 0.24) !important;
    border-color: rgba(196, 181, 253, 0.4) !important;
}

[data-testid="stExpander"] {
    background: rgba(14, 16, 22, 0.96) !important;
    border: 1px solid var(--vm-border) !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--vm-muted);
}

[data-testid="stExpanderDetails"] {
    color: var(--vm-text-soft);
}

.stButton > button {
    background: var(--vm-panel) !important;
    color: var(--vm-text-soft) !important;
    border: 1px solid var(--vm-border) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.95rem !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: #181B24 !important;
    color: var(--vm-text) !important;
    border-color: var(--vm-border-strong) !important;
}

.vm-brand {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.vm-brand-center {
    justify-content: center;
}

.vm-brand-center .vm-brand-copy {
    align-items: center;
    text-align: center;
}

.vm-brand-copy {
    display: flex;
    flex-direction: column;
    gap: 0.16rem;
}

.vm-logo {
    display: block;
    width: 2.7rem;
    height: 2.7rem;
    object-fit: cover;
    border-radius: 16px;
    border: 1px solid rgba(139, 92, 246, 0.18);
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.24);
}

.vm-logo-main {
    width: 3.15rem;
    height: 3.15rem;
}

.vm-wordmark {
    margin: 0;
    line-height: 1;
    letter-spacing: -0.035em;
}

.vm-wordmark .vault {
    color: #FFFFFF;
}

.vm-wordmark .mind {
    color: var(--vm-purple-soft);
}

.vm-wordmark-sidebar {
    font-size: 1.08rem;
    font-weight: 700;
}

.vm-wordmark-main {
    font-size: 1.72rem;
    font-weight: 700;
}

.vm-kicker {
    font-size: 0.72rem;
    color: var(--vm-subtle);
    letter-spacing: 0.03em;
}

.vm-hero {
    padding: 1.15rem 0 1.1rem;
}

.vm-subtitle {
    margin: 0.42rem 0 0;
    text-align: center;
    font-size: 0.98rem;
    font-weight: 400;
    color: var(--vm-muted);
}

.vm-divider {
    height: 1px;
    margin-bottom: 1.55rem;
    background: linear-gradient(90deg, rgba(139, 92, 246, 0.05), rgba(139, 92, 246, 0.24), rgba(28, 30, 39, 0.95), rgba(139, 92, 246, 0.05));
}

.vm-message-row {
    display: flex;
    margin-bottom: 1.15rem;
}

.vm-message-row.user {
    justify-content: flex-end;
}

.vm-user-bubble {
    max-width: min(78%, 720px);
    background: linear-gradient(180deg, rgba(24, 27, 36, 0.98) 0%, rgba(20, 22, 31, 0.98) 100%);
    border: 1px solid #242839;
    border-radius: 18px 18px 6px 18px;
    padding: 0.9rem 1.15rem;
    font-size: 0.95rem;
    color: var(--vm-text);
    line-height: 1.68;
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.16);
}

.vm-answer-wrap {
    margin-bottom: 0.8rem;
}

.vm-answer-card {
    background: linear-gradient(180deg, rgba(18, 20, 28, 0.98) 0%, rgba(15, 17, 24, 0.98) 100%);
    border: 1px solid var(--vm-border);
    border-left: 2px solid var(--vm-purple);
    border-radius: 6px 16px 16px 16px;
    padding: 1rem 1.25rem;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
}

.vm-answer-text {
    font-size: 0.96rem;
    color: var(--vm-text);
    line-height: 1.75;
}

.vm-thinking {
    background: linear-gradient(180deg, rgba(18, 20, 28, 0.98) 0%, rgba(15, 17, 24, 0.98) 100%);
    border: 1px solid var(--vm-border);
    border-left: 2px solid var(--vm-purple);
    border-radius: 6px 16px 16px 16px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.7rem;
}

.vm-thinking-label {
    font-size: 0.84rem;
    color: var(--vm-muted);
}

.vm-inline-panel {
    margin-bottom: 1.15rem;
}

.vm-inline-panel [data-testid="stExpander"] {
    background: rgba(14, 16, 22, 0.98) !important;
}

.vm-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    margin-top: 0.85rem;
    padding: 0.42rem 0.7rem;
    border: 1px solid #202433;
    border-radius: 999px;
    background: rgba(17, 19, 24, 0.92);
    font-size: 0.76rem;
    color: #B1B8CA;
}

.vm-status-indicator {
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 999px;
    background: var(--vm-purple);
    box-shadow: 0 0 0 6px rgba(139, 92, 246, 0.12);
}

.vm-dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: var(--vm-purple);
    animation: vm-pulse 1.2s infinite;
}

@keyframes vm-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.9); }
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #242938; border-radius: 999px; }

@media (max-width: 900px) {
    .vm-brand {
        gap: 0.7rem;
    }

    .vm-logo-main {
        width: 2.85rem;
        height: 2.85rem;
    }

    .vm-wordmark-main {
        font-size: 1.52rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def sanitize_text(text: str) -> str:
    return escape(text).replace("\n", "<br>")


def confidence_bar(score) -> str:
    if score is None:
        return ""

    pct = int(score * 100)
    if score >= 0.7:
        color, label = "#10B981", f"{score:.2f}"
    elif score >= 0.4:
        color, label = "#F59E0B", f"{score:.2f}"
    else:
        color, label = "#EF4444", f"{score:.2f}"

    return dedent(f"""
        <div style="display:flex;align-items:center;gap:0.7rem;margin-top:0.9rem;">
          <div style="flex:1;height:3px;background:#202432;border-radius:999px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:999px;transition:width 0.4s ease;"></div>
          </div>
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:{color};min-width:2.8rem;text-align:right;">{label}</span>
        </div>
    """).strip()


def format_meta_row(tokens: int, cost: float) -> str:
    return dedent(f"""
        <div style="display:flex;gap:1.4rem;margin-top:0.65rem;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#6E7588;">
          <span>{tokens:,} tok</span>
          <span>${cost:.6f}</span>
        </div>
    """).strip()


def brand_markup(location: str, subtitle: str) -> str:
    title_class  = "vm-wordmark-main"    if location == "main" else "vm-wordmark-sidebar"
    logo_class   = "vm-logo-main"        if location == "main" else ""
    wrapper_class = "vm-brand vm-brand-center" if location == "main" else "vm-brand"

    return dedent(f"""
        <div class="{wrapper_class}">
          <img src="{LOGO_DATA_URI}" alt="VaultMind logo" class="vm-logo {logo_class}">
          <div class="vm-brand-copy">
            <h1 class="vm-wordmark {title_class}">
              <span class="vault">Vault</span><span class="mind">Mind</span>
            </h1>
            <div class="vm-kicker">{subtitle}</div>
          </div>
        </div>
    """).strip()


def render_brand(location: str, subtitle: str) -> None:
    st.markdown(brand_markup(location, subtitle), unsafe_allow_html=True)


def render_user_message(content: str) -> None:
    st.markdown(
        dedent(f"""
            <div class="vm-message-row user">
              <div class="vm-user-bubble">{sanitize_text(content)}</div>
            </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def render_assistant_message(content: str, meta: dict) -> None:
    blocked    = meta.get("input_blocked", False)
    flagged    = meta.get("output_flagged", False)
    confidence = meta.get("confidence_score")
    tokens     = meta.get("total_tokens", 0)
    cost       = meta.get("estimated_cost", 0.0)

    border_color     = "#3B1A1A" if (blocked or flagged) else "#1C1E27"
    accent           = "#EF4444" if (blocked or flagged) else "#8B5CF6"
    confidence_markup = confidence_bar(confidence)

    st.markdown(
        dedent(f"""
            <div class="vm-answer-wrap">
              <div class="vm-answer-card" style="border-color:{border_color};border-left-color:{accent};">
                <div class="vm-answer-text">{sanitize_text(content)}</div>
                {confidence_markup}
              </div>
            </div>
        """).strip(),
        unsafe_allow_html=True,
    )
    st.markdown(format_meta_row(tokens, cost), unsafe_allow_html=True)


def render_chunk_count(count: int) -> None:
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
        f'color:#586075;margin-top:0.4rem;">{count} chunks retrieved</div>',
        unsafe_allow_html=True,
    )


PIPELINE_NODES = [
    ("01", "Input Guardrail",  "Pattern match + LLM safety"),
    ("02", "Classifier",       "Resume query detection"),
    ("03", "Hybrid Retriever", "ChromaDB · BM25 · RRF"),
    ("04", "Context Guard",    "Token budget ≤ 1,000"),
    ("05", "Generator",        "GPT-4o-mini · retry ×3"),
    ("06", "Output Guardrail", "PII redaction · safety"),
    ("07", "Confidence",       "Self-RAG score"),
]

STATUS_STYLES = {
    "idle":    ("color:#586075;", "○"),
    "running": ("color:#8B5CF6;animation:vm-pulse 1.2s infinite;", "◉"),
    "done":    ("color:#10B981;", "✓"),
    "blocked": ("color:#EF4444;", "✕"),
    "skipped": ("color:#2A2E3D;", "–"),
}


def render_pipeline(statuses: dict) -> None:
    st.markdown("""
        <style>
        .pipe-label {
          font-family:'Inter',sans-serif;font-size:0.64rem;font-weight:700;
          letter-spacing:0.1em;text-transform:uppercase;color:#848BA0;margin-bottom:0.7rem;
        }
        .pipe-row {
          display:flex;align-items:center;gap:0.65rem;
          padding:0.46rem 0;border-bottom:1px solid #151822;
        }
        .pipe-num { font-family:'JetBrains Mono',monospace;font-size:0.64rem;color:#636B80;min-width:1.5rem; }
        .pipe-name { font-family:'Inter',sans-serif;font-size:0.8rem;font-weight:500;flex:1; }
        .pipe-dot { font-size:0.68rem; }
        </style>
        <div class="pipe-label">Pipeline</div>
    """, unsafe_allow_html=True)

    for i, (num, name, _) in enumerate(PIPELINE_NODES):
        status     = statuses.get(i, "idle")
        style, dot = STATUS_STYLES[status]

        name_styles = {
            "idle":    "color:#747C90;",
            "running": "color:#D8CCFF;",
            "done":    "color:#F0F3FA;",
            "blocked": "color:#F87171;",
            "skipped": "color:#3A4051;",
        }
        name_style = name_styles.get(status, "color:#747C90;")

        st.markdown(
            f'<div class="pipe-row">'
            f'<span class="pipe-num">{num}</span>'
            f'<span class="pipe-name" style="{name_style}">{name}</span>'
            f'<span class="pipe-dot" style="{style}">{dot}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_process_panel(statuses: dict, status_text: str) -> None:
    render_pipeline(statuses)
    st.markdown(
        f"""
        <div class="vm-status-chip">
          <span class="vm-status-indicator"></span>
          <span>{escape(status_text)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# BACKEND RUNNER
# ─────────────────────────────────────────────
def _error_result(message: str) -> dict:
    # single place to build an error-state result dict — no duplication
    return {
        "answer":           message,
        "input_blocked":    True,
        "output_flagged":   False,
        "confidence_score": None,
        "total_tokens":     0,
        "estimated_cost":   0.0,
        "retrieval_status": "",
        "retrieved_chunks": 0,
    }


def run_vaultmind(question: str, pipeline_placeholder, status_placeholder) -> dict:

    # all nodes pulse while we wait — backend is a black box over HTTP
    statuses = {i: "running" for i in range(7)}
    with pipeline_placeholder.container():
        render_process_panel(statuses, "Running pipeline...")

    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/query",
            json={"question": question},
            timeout=60.0,
        )

        if response.status_code == 429:
            return _error_result("⚠️ Too many requests. Please wait a moment and try again.")

        response.raise_for_status()
        result = response.json()

    except httpx.ConnectError:
        return _error_result("⚠️ Cannot reach the VaultMind backend. Is it running on port 8000?")

    except httpx.HTTPStatusError as e:
        return _error_result(f"⚠️ Backend error ({e.response.status_code}). Please try again.")

    # reconstruct which nodes ran from the result fields
    blocked    = result.get("input_blocked", False)
    ret_status = result.get("retrieval_status", "")

    if blocked:
        statuses = {0:"blocked", 1:"skipped", 2:"skipped", 3:"skipped",
                    4:"skipped", 5:"skipped", 6:"skipped"}
    elif ret_status == "":
        # classifier said not relevant — retriever never ran
        statuses = {0:"done", 1:"done", 2:"skipped", 3:"skipped",
                    4:"done",  5:"done",  6:"done"}
    else:
        statuses = {i: "done" for i in range(7)}

    # animate nodes lighting up one by one
    for node_idx in range(7):
        tmp = {i: "idle" for i in range(7)}
        for j in range(node_idx):
            tmp[j] = statuses[j]
        tmp[node_idx] = statuses[node_idx]
        with pipeline_placeholder.container():
            render_process_panel(tmp, "Running pipeline...")
        time.sleep(0.06)

    with pipeline_placeholder.container():
        render_process_panel(statuses, "Complete")

    # Fix 3 — update status_placeholder text after pipeline finishes
    status_placeholder.markdown(
        '<span style="font-family:\'Inter\',sans-serif;font-size:0.76rem;color:#7F879B;">Complete</span>',
        unsafe_allow_html=True,
    )

    st.session_state.pipeline_statuses     = statuses
    st.session_state.pipeline_status_label = "Complete"
    return result


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("messages",               []),
    ("total_tokens_session",   0),
    ("total_cost_session",     0.0),
    ("query_count",            0),
    ("pipeline_statuses",      {i: "idle" for i in range(7)}),
    ("pipeline_status_label",  "Ready to answer"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    render_brand("sidebar", "Resume Intelligence · Phase 11")
    st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)

    pipeline_placeholder = st.empty()
    with pipeline_placeholder.container():
        render_process_panel(
            st.session_state.pipeline_statuses,
            st.session_state.pipeline_status_label,
        )

    status_placeholder = st.empty()

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style="font-family:'Inter',sans-serif;font-size:0.64rem;font-weight:700;
             letter-spacing:0.1em;text-transform:uppercase;color:#848BA0;margin-bottom:0.85rem;">
          Session
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style="background:#111318;border:1px solid #1C1E27;border-radius:10px;padding:0.9rem 0.8rem;">
              <div style="font-family:'Inter',sans-serif;font-size:0.62rem;font-weight:600;
                   letter-spacing:0.08em;text-transform:uppercase;color:#7E869A;margin-bottom:0.38rem;">Queries</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.18rem;font-weight:500;
                   color:#F5F7FB;">{st.session_state.query_count}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style="background:#111318;border:1px solid #1C1E27;border-radius:10px;padding:0.9rem 0.8rem;">
              <div style="font-family:'Inter',sans-serif;font-size:0.62rem;font-weight:600;
                   letter-spacing:0.08em;text-transform:uppercase;color:#7E869A;margin-bottom:0.38rem;">Tokens</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.18rem;font-weight:500;
                   color:#F5F7FB;">{st.session_state.total_tokens_session:,}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="background:#111318;border:1px solid #1C1E27;border-radius:10px;padding:0.9rem 0.8rem;">
          <div style="font-family:'Inter',sans-serif;font-size:0.62rem;font-weight:600;
               letter-spacing:0.08em;text-transform:uppercase;color:#7E869A;margin-bottom:0.38rem;">Session Cost</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.18rem;font-weight:500;
               color:#F5F7FB;">${st.session_state.total_cost_session:.5f}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style="font-family:'Inter',sans-serif;font-size:0.64rem;font-weight:700;
             letter-spacing:0.1em;text-transform:uppercase;color:#848BA0;margin-bottom:0.85rem;">
          Evaluation · RAGAS
        </div>
    """, unsafe_allow_html=True)

    for metric, score in [
        ("Faithfulness", 0.993),
        ("Relevancy",    0.906),
        ("Precision",    0.738),
        ("Recall",       0.821),
        ("Overall",      0.865),
    ]:
        pct   = int(score * 100)
        color = "#10B981" if score >= 0.85 else "#F59E0B" if score >= 0.7 else "#EF4444"
        st.markdown(f"""
            <div style="margin-bottom:0.6rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.28rem;">
                <span style="font-family:'Inter',sans-serif;font-size:0.74rem;color:#A6AEC2;">{metric}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:{color};font-weight:500;">{score:.3f}</span>
              </div>
              <div style="height:2px;background:#171B24;border-radius:999px;">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:999px;opacity:0.8;"></div>
              </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages               = []
        st.session_state.pipeline_statuses      = {i: "idle" for i in range(7)}
        st.session_state.pipeline_status_label  = "Ready to answer"
        st.rerun()

    st.markdown("""
        <div style="margin-top:1.55rem;padding-top:1rem;border-top:1px solid #151822;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.64rem;color:#70788C;line-height:1.95;">
            gpt-4o-mini · text-embedding-3-small<br>
            ChromaDB · BM25Okapi · RRF
          </div>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────
st.markdown(
    dedent(f"""
        <div class="vm-hero">
          {brand_markup("main", "Ask about Dev Doshi — experience, skills, projects, education.")}
        </div>
        <div class="vm-divider"></div>
    """).strip(),
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# RENDER CHAT HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user_message(msg["content"])
    else:
        meta        = msg.get("meta", {})
        chunk_count = meta.get("retrieved_chunks", 0)

        render_assistant_message(msg["content"], meta)

        if chunk_count > 0 and not meta.get("input_blocked", False):
            render_chunk_count(chunk_count)


# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
if question := st.chat_input("Message VaultMind..."):
    render_user_message(question)

    thinking_slot = st.empty()
    thinking_slot.markdown("""
        <div class="vm-thinking">
          <div style="display:flex;align-items:center;gap:0.7rem;">
            <div class="vm-dot"></div>
            <span class="vm-thinking-label">Processing your question...</span>
          </div>
        </div>
    """, unsafe_allow_html=True)

    result = run_vaultmind(question, pipeline_placeholder, status_placeholder)
    thinking_slot.empty()

    answer      = result.get("answer", "No answer returned.")
    blocked     = result.get("input_blocked", False)
    flagged     = result.get("output_flagged", False)
    confidence  = result.get("confidence_score")
    tokens      = result.get("total_tokens", 0)
    cost        = result.get("estimated_cost", 0.0)
    chunk_count = result.get("retrieved_chunks", 0)

    render_assistant_message(
        answer,
        {
            "input_blocked":    blocked,
            "output_flagged":   flagged,
            "confidence_score": confidence,
            "total_tokens":     tokens,
            "estimated_cost":   cost,
        },
    )

    if chunk_count > 0 and not blocked:
        render_chunk_count(chunk_count)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": {
            "input_blocked":    blocked,
            "output_flagged":   flagged,
            "confidence_score": confidence,
            "total_tokens":     tokens,
            "estimated_cost":   cost,
            "retrieved_chunks": chunk_count,
        },
    })

    st.session_state.total_tokens_session += tokens
    st.session_state.total_cost_session   += cost
    st.session_state.query_count          += 1

    st.rerun()