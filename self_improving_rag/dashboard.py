"""AutoRAG Dashboard — persistent multi-session chat UI"""
import json
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .analysis import compute_hyperparameter_importance
from .config import RAGConfig
from .evaluator import RAGEvaluator
from .pipeline import RAGPipeline
from .question_store import record_question, question_count
from .storage import RAGStorage
from .chat_store import (
    create_session, delete_session, get_session,
    list_sessions, save_messages, rename_session,
)

# ── Design tokens ─────────────────────────────────────────────────────────────
P      = "#4361ee"
P_D    = "#3451d1"
P_L    = "#eef0fd"
GREEN  = "#10b981"; GREEN_L = "#d1fae5"
RED    = "#ef233c"; RED_L   = "#fee2e2"
AMBER  = "#f59e0b"; AMBER_L = "#fef3c7"
BG     = "#f4f6fb"
CARD   = "#ffffff"
BORDER = "#e8eaf6"
TEXT   = "#1a1a2e"
MUTED  = "#6b7280"
SB_BG  = "#0f172a"   # sidebar background
SB_HVR = "#1e293b"   # sidebar hover
SB_ACT = "#1d4ed8"   # sidebar active

_USER_EMBED_MODEL = "all-MiniLM-L6-v2"
_USER_CHUNK_SIZE  = 512
_USER_OVERLAP     = 64

GLOBAL_CSS = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
#MainMenu,footer{{visibility:hidden}}
[data-testid="stToolbar"],[data-testid="stDecoration"],.stDeployButton{{display:none!important}}
.block-container{{padding:0!important;max-width:100%!important}}
html,body,[class*="css"]{{font-family:'Inter',-apple-system,sans-serif!important;background:{BG}!important}}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{{
    background:{SB_BG}!important;
    min-width:240px!important; max-width:260px!important;
    padding:0!important;
}}
section[data-testid="stSidebar"] .block-container{{
    padding:0!important; background:{SB_BG}!important;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{{
    gap:0!important;
}}
/* hide sidebar collapse toggle */
[data-testid="collapsedControl"]{{display:none!important}}
button[data-testid="stSidebarNavCollapseButton"]{{display:none!important}}

/* ── Sidebar buttons ── */
section[data-testid="stSidebar"] .stButton>button{{
    background:transparent!important; border:none!important; box-shadow:none!important;
    color:rgba(255,255,255,.75)!important; font-size:.82rem!important;
    font-weight:500!important; text-align:left!important; border-radius:7px!important;
    padding:.5rem .75rem!important; width:100%!important; transition:all .1s!important;
    white-space:nowrap!important; overflow:hidden!important; text-overflow:ellipsis!important;
}}
section[data-testid="stSidebar"] .stButton>button:hover{{
    background:{SB_HVR}!important; color:#fff!important;
}}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{{
    background:{SB_ACT}!important; color:#fff!important; font-weight:700!important;
}}

/* ── Main inputs ── */
.stTextInput>div>div>input,.stTextArea>div>div>textarea{{
    border-radius:8px!important;border-color:{BORDER}!important;
    font-family:'Inter',sans-serif!important;background:#fff!important}}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{{
    border-color:{P}!important;box-shadow:0 0 0 3px {P_L}!important}}
.stButton>button{{border-radius:8px!important;font-family:'Inter',sans-serif!important;
    font-weight:600!important;transition:all .15s!important}}
.stButton>button[kind="primary"]{{background:{P}!important;border-color:{P}!important;color:#fff!important}}
.stButton>button[kind="primary"]:hover{{background:{P_D}!important;
    box-shadow:0 4px 14px rgba(67,97,238,.3)!important}}
.stAlert>div{{border-radius:10px!important}}
[data-baseweb="tab"]{{border-radius:8px!important;font-weight:600!important}}
[data-testid="stChatMessageContent"] p{{margin:0}}
[data-testid="stFileUploaderDropzone"]{{border-radius:12px!important;border-color:{P}!important}}
</style>"""


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def _store(db_path: str) -> RAGStorage:
    return RAGStorage(db_path=db_path)


@st.cache_resource
def _embedder_singleton(model_name: str):
    """Load SentenceTransformer once, shared across all chat sessions."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_resource
def _pipeline(chroma_dir: str, cfg_json: str, collection_name: str) -> RAGPipeline:
    """One cached pipeline per (chroma_dir, config, collection). Embedder is shared."""
    cfg = RAGConfig.from_dict(json.loads(cfg_json))
    p = RAGPipeline(cfg, chroma_dir=chroma_dir, collection_name=collection_name)
    p._embedder = _embedder_singleton(cfg.embedding_model)
    return p


def _user_cfg() -> RAGConfig:
    return RAGConfig(
        embedding_model=_USER_EMBED_MODEL,
        chunk_size=_USER_CHUNK_SIZE,
        overlap=_USER_OVERLAP,
        top_k=8,
        similarity_threshold=0.0,
        groq_model="llama-3.3-70b-versatile",
    )


def _load_optuna_cfg(username: Optional[str] = None) -> RAGConfig:
    from .scheduler import load_current_config
    return load_current_config(username=username)


def _chat_collection(username: str, session_id: str) -> str:
    """Unique ChromaDB collection name per user per chat session."""
    safe = "".join(c for c in username if c.isalnum() or c == "_")[:10]
    return f"u_{safe}_{session_id}"


def _doc_count(chroma_dir: str, collection_name: str) -> int:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        for c in client.list_collections():
            if c.name == collection_name:
                return c.count()
        return 0
    except Exception:
        return 0


def _indexed_files(chroma_dir: str, collection_name: str) -> list[dict]:
    """Return list of {name, chunks} for every unique source file in the collection."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        col    = client.get_collection(collection_name)
        total  = col.count()
        if total == 0:
            return []
        counts: dict[str, int] = {}
        batch  = 1000
        offset = 0
        while offset < total:
            rows = col.get(limit=batch, offset=offset, include=["metadatas"])
            for m in rows.get("metadatas", []):
                src = os.path.basename(m.get("source", "") or "unknown")
                counts[src] = counts.get(src, 0) + 1
            offset += batch
        return [{"name": k, "chunks": v} for k, v in sorted(counts.items())]
    except Exception:
        return []


def _delete_file_from_kb(chroma_dir: str, collection_name: str, source_basename: str) -> int:
    """Delete all chunks belonging to source_basename. Returns chunk count removed."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        col    = client.get_collection(collection_name)
        total  = col.count()
        to_del = []
        batch  = 1000
        offset = 0
        while offset < total:
            rows = col.get(limit=batch, offset=offset, include=["metadatas"])
            for chunk_id, meta in zip(rows.get("ids", []), rows.get("metadatas", [])):
                if os.path.basename(meta.get("source", "") or "") == source_basename:
                    to_del.append(chunk_id)
            offset += batch
        if to_del:
            col.delete(ids=to_del)
        return len(to_del)
    except Exception:
        return 0


def _clear_kb(chroma_dir: str, collection_name: str) -> None:
    """Delete the entire collection from ChromaDB."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        client.delete_collection(collection_name)
    except Exception:
        pass


def _sample_topics(chroma_dir: str, collection_name: str, n: int = 6) -> list[str]:
    """Return n short text snippets from random chunks — used for topic hints."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        col    = client.get_collection(collection_name)
        total  = col.count()
        if total == 0:
            return []
        limit  = min(n * 4, total)
        rows   = col.get(limit=limit, include=["documents"])
        docs   = rows.get("documents", [])
        hints  = []
        for doc in docs[::max(1, len(docs) // n)]:
            first = doc.strip().split(".")[0].strip()
            if first and first not in hints:
                hints.append(first[:80])
            if len(hints) >= n:
                break
        return hints
    except Exception:
        return []


def _score_badge(label: str, val: float) -> str:
    if val >= 0.7:   c, bg = GREEN, GREEN_L
    elif val >= 0.5: c, bg = AMBER, AMBER_L
    else:            c, bg = RED,   RED_L
    return (f'<span style="display:inline-flex;align-items:center;padding:.18rem .6rem;'
            f'border-radius:99px;font-size:.68rem;font-weight:700;color:{c};background:{bg}">'
            f'{label} {val:.2f}</span>')


def _kpi(label, value, sub="", accent=P):
    return (f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;'
            f'padding:1.1rem 1.4rem;border-top:4px solid {accent}">'
            f'<div style="font-size:.62rem;font-weight:700;color:{MUTED};text-transform:uppercase;'
            f'letter-spacing:.07em;margin-bottom:.4rem">{label}</div>'
            f'<div style="font-size:1.75rem;font-weight:800;color:{TEXT};line-height:1">{value}</div>'
            f'<div style="font-size:.7rem;color:{MUTED};margin-top:.3rem">{sub}</div></div>')


def _hr():
    return f'<hr style="border:none;border-top:1px solid {BORDER};margin:1.25rem 0">'


def _pt():
    return {"paper_bgcolor": CARD, "plot_bgcolor": "#f8f9fe",
            "font": {"color": TEXT, "family": "Inter,sans-serif"},
            "margin": {"l": 40, "r": 20, "t": 44, "b": 40}}


# ─────────────────────────────────────────────────────────────────────────────
#  AUTH
# ─────────────────────────────────────────────────────────────────────────────
def _auth() -> bool:
    from .auth import ensure_store_initialized, register_user, verify_user

    if st.session_state.get("authenticated"):
        return True

    ensure_store_initialized()
    mode = st.session_state.get("auth_mode", "login")

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    # hide sidebar on auth screen
    st.markdown(f"""<style>
    section[data-testid="stSidebar"]{{display:none!important}}
    .block-container{{max-width:420px!important;margin:60px auto!important;padding:0 1rem!important}}
    div[data-testid="stForm"]{{background:{CARD};border:1px solid {BORDER};border-radius:20px;
        padding:2.5rem!important;box-shadow:0 8px 48px rgba(67,97,238,.1)}}
    </style>""", unsafe_allow_html=True)

    st.markdown(
        f'<div style="text-align:center;margin-bottom:1.5rem">'
        f'<div style="font-size:1.85rem;font-weight:800;color:{P}">Auto<span style="color:{TEXT}">RAG</span></div>'
        f'<div style="font-size:.85rem;color:{MUTED};margin-top:.25rem">'
        f'{"Create your account" if mode=="register" else "Sign in to continue"}</div></div>',
        unsafe_allow_html=True,
    )

    if mode == "login":
        with st.form("lf"):
            u  = st.text_input("Username")
            p  = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in →", use_container_width=True, type="primary")
        if ok:
            if verify_user(u.strip(), p):
                st.session_state.update(authenticated=True, auth_user=u.strip(), page="Chat")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.markdown(f'<p style="text-align:center;font-size:.82rem;color:{MUTED}">No account?</p>',
                    unsafe_allow_html=True)
        if st.button("Create an account →", use_container_width=True):
            st.session_state["auth_mode"] = "register"; st.rerun()
    else:
        with st.form("rf"):
            nu  = st.text_input("Username", help="3–32 chars, letters/numbers/_")
            np  = st.text_input("Password", type="password", help="Min 8 chars")
            cp  = st.text_input("Confirm password", type="password")
            ok  = st.form_submit_button("Create account →", use_container_width=True, type="primary")
        if ok:
            if not nu or not np:  st.error("Fill in all fields.")
            elif np != cp:        st.error("Passwords do not match.")
            else:
                err = register_user(nu.strip(), np)
                if err: st.error(err)
                else:
                    st.success("Account created! Sign in.")
                    st.session_state["auth_mode"] = "login"; st.rerun()
        st.markdown(f'<p style="text-align:center;font-size:.82rem;color:{MUTED}">Have account?</p>',
                    unsafe_allow_html=True)
        if st.button("← Back to sign in", use_container_width=True):
            st.session_state["auth_mode"] = "login"; st.rerun()

    return False


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR  (chat list + navigation)
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar(chroma_dir: str) -> str:
    """Render sidebar and return current page name."""
    user    = st.session_state.get("auth_user", "user")
    page    = st.session_state.get("page", "Chat")
    cur_sid = st.session_state.get("chat_session_id")

    with st.sidebar:
        # ── Brand + optimizer progress ────────────────────────────────────────
        _THRESHOLD = 10
        q_count    = question_count(user)
        filled     = min(q_count, _THRESHOLD)
        pct        = filled / _THRESHOLD          # 0.0 – 1.0
        bar_filled = int(pct * 10)                # 0–10 blocks
        bar_empty  = 10 - bar_filled
        bar_html   = (
            f'<span style="color:#4361ee">{"█" * bar_filled}</span>'
            f'<span style="color:rgba(255,255,255,.15)">{"█" * bar_empty}</span>'
        )

        if q_count == 0:
            status_line = "Ask questions to start training"
            status_color = "rgba(255,255,255,.25)"
        elif q_count < _THRESHOLD:
            status_line = f"{q_count}/{_THRESHOLD} · keep chatting"
            status_color = "#f59e0b"
        else:
            new_since = q_count - st.session_state.get("_last_opt_count", 0)
            if new_since >= _THRESHOLD:
                status_line = f"{q_count} questions · optimizer fires within 1 hr"
                status_color = "#10b981"
            else:
                status_line = f"{q_count} questions · waiting for {_THRESHOLD - new_since} more"
                status_color = "#4361ee"

        st.markdown(f"""
<div style="padding:1.1rem 1rem .8rem;border-bottom:1px solid rgba(255,255,255,.08)">
  <div style="font-size:1.05rem;font-weight:800;color:#fff;letter-spacing:-.3px">
    Auto<span style="opacity:.55">RAG</span>
  </div>
  <div style="font-size:.7rem;color:rgba(255,255,255,.4);margin-top:.1rem">{user}</div>
  <div style="margin-top:.6rem">
    <div style="font-size:.62rem;color:rgba(255,255,255,.3);
                text-transform:uppercase;letter-spacing:.06em;margin-bottom:.2rem">
      🔬 Self-Improvement
    </div>
    <div style="font-size:.75rem;font-family:monospace;letter-spacing:.5px">{bar_html}</div>
    <div style="font-size:.65rem;margin-top:.2rem;color:{status_color}">{status_line}</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div style="padding:.5rem .75rem .25rem">', unsafe_allow_html=True)

        # "Optimize Now" button — visible once 3+ questions collected
        if q_count >= 3:
            from .scheduler import OPTIMIZER_LOCK
            opt_running = OPTIMIZER_LOCK.locked()

            if opt_running:
                st.markdown(
                    f'<div style="padding:.4rem .75rem;font-size:.68rem;color:#f59e0b">'
                    f'⏳ Optimizer is running…</div>',
                    unsafe_allow_html=True,
                )
            elif st.button("⚡ Optimize Now", key="sb_opt_now", use_container_width=True,
                           help=f"Run optimizer using your {q_count} questions (~3 min)"):
                with st.spinner("Optimizing… 10 trials, ~16 s each. Please wait (~3 min)."):
                    try:
                        from .scheduler import nightly_job, OPTIMIZER_LOCK as _LOCK
                        if not _LOCK.acquire(blocking=False):
                            st.warning("Optimizer already running in background — try again shortly.")
                        else:
                            try:
                                result = nightly_job(questions=[], chroma_dir=chroma_dir,
                                                     username=user)
                                deployed = result.get("deployed", False)
                                best     = result.get("best_score")
                                st.session_state["_last_opt_count"] = q_count
                                if deployed:
                                    st.success(f"✅ New config deployed! Best score: {best:.3f}")
                                elif best is not None:
                                    st.info(f"Ran {result.get('n_trials',0)} trials — "
                                            f"best score {best:.3f}. Existing config kept (already optimal).")
                                else:
                                    st.warning("No completed trials. Check logs for details.")
                            finally:
                                _LOCK.release()
                    except Exception as exc:
                        msg = str(exc)
                        if "daily token limit" in msg or "API providers exhausted" in msg or "All OpenRouter" in msg:
                            st.warning(msg)
                        else:
                            st.error(f"Optimizer error: {exc}")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="padding:.5rem .75rem .25rem">', unsafe_allow_html=True)

        # New chat button
        if st.button("✏️  New Chat", key="sb_new", use_container_width=True):
            sid = create_session(user)
            st.session_state.update(chat_session_id=sid, page="Chat",
                                    chat_messages=[])
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Chat history list
        sessions = list_sessions(user)
        if sessions:
            st.markdown(
                f'<div style="padding:.4rem 1rem .2rem;font-size:.62rem;font-weight:700;'
                f'color:rgba(255,255,255,.3);text-transform:uppercase;letter-spacing:.08em">'
                f'Recent</div>',
                unsafe_allow_html=True,
            )

        for s in sessions:
            sid      = s["id"]
            title    = s.get("title", "New Chat")
            is_cur   = sid == cur_sid and page == "Chat"
            # Truncate title for display
            display  = title if len(title) <= 28 else title[:26] + "…"

            cols = st.columns([5, 1])
            with cols[0]:
                kind = "primary" if is_cur else "secondary"
                if st.button(f"💬 {display}", key=f"sb_{sid}", use_container_width=True,
                             type=kind):
                    # Load this session
                    data = get_session(user, sid)
                    msgs = data.get("messages", []) if data else []
                    st.session_state.update(
                        chat_session_id=sid,
                        page="Chat",
                        chat_messages=msgs,
                    )
                    st.rerun()
            with cols[1]:
                if st.button("🗑", key=f"del_{sid}", help="Delete"):
                    delete_session(user, sid)
                    if cur_sid == sid:
                        st.session_state.pop("chat_session_id", None)
                        st.session_state["chat_messages"] = []
                    st.rerun()

        # Knowledge base file list for the active chat
        if cur_sid:
            col_name  = _chat_collection(user, cur_sid)
            kb_files  = _indexed_files(chroma_dir, col_name)
            if kb_files:
                st.markdown(
                    f'<div style="padding:.4rem 1rem .2rem;margin-top:.5rem;'
                    f'font-size:.62rem;font-weight:700;color:rgba(255,255,255,.3);'
                    f'text-transform:uppercase;letter-spacing:.08em">📚 This Chat</div>',
                    unsafe_allow_html=True,
                )
                for f in kb_files:
                    icon = ("📄" if f["name"].endswith(".pdf") else
                            "📝" if f["name"].endswith((".docx", ".doc")) else
                            "📖" if f["name"].endswith(".epub") else "📃")
                    name_short = f["name"] if len(f["name"]) <= 22 else f["name"][:20] + "…"
                    st.markdown(
                        f'<div style="padding:.2rem 1rem;display:flex;justify-content:space-between;'
                        f'align-items:center">'
                        f'<span style="font-size:.72rem;color:rgba(255,255,255,.6);'
                        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                        f'{icon} {name_short}</span>'
                        f'<span style="font-size:.65rem;color:rgba(255,255,255,.3);'
                        f'margin-left:.4rem;flex-shrink:0">{f["chunks"]}c</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Divider + nav links
        st.markdown(f"""
<div style="border-top:1px solid rgba(255,255,255,.08);margin:.75rem 0"></div>
<div style="padding:.25rem .75rem">""", unsafe_allow_html=True)

        for label, pg, icon in [("Analytics", "Analytics", "📊"),
                                 ("Configuration", "Configuration", "⚙️")]:
            active = page == pg
            kind   = "primary" if active else "secondary"
            if st.button(f"{icon}  {label}", key=f"sb_{pg.lower()}", use_container_width=True, type=kind):
                st.session_state["page"] = pg; st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Logout at bottom
        st.markdown(f'<div style="position:absolute;bottom:0;width:100%;padding:.75rem;'
                    f'border-top:1px solid rgba(255,255,255,.08)">', unsafe_allow_html=True)
        if st.button("⎋  Logout", key="sb_logout", use_container_width=True):
            st.session_state.update(authenticated=False, auth_mode="login")
            st.session_state.pop("auth_user", None)
            st.session_state.pop("chat_session_id", None)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────────────────────────────────────
#  CHAT PAGE  (upload + persistent chat)
# ─────────────────────────────────────────────────────────────────────────────
def _chat_page(chroma_dir: str) -> None:
    user     = st.session_state.get("auth_user", "user")
    cfg      = _user_cfg()
    cfg_json = json.dumps(cfg.to_dict(), sort_keys=True)

    # Ensure there's always an active session
    if not st.session_state.get("chat_session_id"):
        sessions = list_sessions(user)
        if sessions:
            sid  = sessions[0]["id"]
            data = get_session(user, sid)
            st.session_state.update(
                chat_session_id=sid,
                chat_messages=data.get("messages", []) if data else [],
            )
        else:
            sid = create_session(user)
            st.session_state.update(chat_session_id=sid, chat_messages=[])

    sid             = st.session_state["chat_session_id"]
    col_name        = _chat_collection(user, sid)   # isolated per-chat KB
    messages        = st.session_state.get("chat_messages", [])
    doc_count       = _doc_count(chroma_dir, col_name)

    # ── Top bar ───────────────────────────────────────────────────────────────
    show_up = st.session_state.get("show_upload", doc_count == 0)
    top_left, top_right = st.columns([7, 3])
    with top_left:
        session_data = get_session(user, sid) or {}
        title        = session_data.get("title", "New Chat")
        st.markdown(
            f'<div style="padding:.85rem 1.5rem .4rem;font-size:1rem;font-weight:700;color:{TEXT}">'
            f'{title}</div>',
            unsafe_allow_html=True,
        )
    with top_right:
        doc_color = GREEN_L if doc_count else RED_L
        doc_text  = GREEN   if doc_count else RED
        st.markdown(
            f'<div style="padding:.9rem .5rem .3rem;text-align:right">'
            f'<span style="background:{doc_color};color:{doc_text};border-radius:99px;'
            f'padding:.2rem .65rem;font-size:.7rem;font-weight:700">'
            f'{"📚 " + str(doc_count) + " chunks" if doc_count else "⚠️ No docs"}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div style="height:1px;background:{BORDER};margin:0 1.5rem .5rem"></div>',
                unsafe_allow_html=True)

    # ── Chat messages ─────────────────────────────────────────────────────────
    _not_found_phrases = (
        "i don't know based on the provided context",
        "i do not know based on the provided context",
    )

    if doc_count == 0:
        # Empty state — no messages to show; upload expander appears below
        st.markdown(
            f'<div style="text-align:center;padding:4rem 1rem 2rem;color:{MUTED}">'
            f'<div style="font-size:2.5rem;margin-bottom:.75rem">📂</div>'
            f'<div style="font-size:1rem;font-weight:600;color:{TEXT};margin-bottom:.3rem">'
            f'No documents yet</div>'
            f'<div style="font-size:.85rem">Use the <b>📎 Attach files</b> panel below to upload PDFs, '
            f'DOCX, EPUB, TXT or paste a URL — then start chatting.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        for msg in messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])
                if msg.get("scores") and msg["role"] == "assistant":
                    s            = msg["scores"]
                    is_not_found = any(p in msg["content"].lower() for p in _not_found_phrases)
                    if is_not_found:
                        st.markdown(
                            f'<div style="background:{AMBER_L};border:1px solid {AMBER};'
                            f'border-radius:10px;padding:.5rem .85rem;font-size:.78rem;color:#92400e">'
                            f'⚠️ Not found in your documents.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            _score_badge("Faithfulness", s.get("faithfulness", 0)) + " " +
                            _score_badge("Relevance",    s.get("relevance",    0)) + " " +
                            _score_badge("Composite",    s.get("composite",    0)) +
                            f' <span style="font-size:.67rem;color:{MUTED};margin-left:.35rem">'
                            f'⏱ {s.get("retrieval_ms","?")} ms · {s.get("generation_ms","?")} ms</span>',
                            unsafe_allow_html=True,
                        )
                if msg.get("chunks"):
                    with st.expander(f"📎 {len(msg['chunks'])} sources"):
                        for i, chunk in enumerate(msg["chunks"], 1):
                            src   = os.path.basename(chunk.get("metadata", {}).get("source", "")) or "—"
                            score = chunk.get("score", 0)
                            st.markdown(
                                f'<div style="margin-bottom:.6rem;padding:.75rem;background:{BG};'
                                f'border-radius:9px;border-left:3px solid {P}">'
                                f'<div style="font-size:.66rem;font-weight:700;color:{MUTED};margin-bottom:.25rem">'
                                f'#{i} · {score:.3f} · {src}</div>'
                                f'<div style="font-size:.81rem;color:{TEXT};line-height:1.6">'
                                f'{chunk["text"][:400]}{"…" if len(chunk["text"])>400 else ""}'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )

        # Welcome state (docs indexed but no messages yet)
        if not messages:
            st.markdown(
                f'<div style="text-align:center;padding:3.5rem 1rem;color:{MUTED}">'
                f'<div style="font-size:2rem;margin-bottom:.6rem">💬</div>'
                f'<div style="font-size:.95rem;font-weight:600;color:{TEXT};margin-bottom:.25rem">'
                f'Knowledge base ready</div>'
                f'<div style="font-size:.82rem">'
                f'{doc_count:,} chunks indexed · Ask anything about your documents.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Floating upload panel + FAB ───────────────────────────────────────────
    # Upload panel renders just above the chat input when open
    if show_up:
        with st.container():
            st.markdown(
                f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;'
                f'padding:1rem 1.2rem;margin:.5rem 0 .5rem">',
                unsafe_allow_html=True,
            )
            _upload_widget(chroma_dir, cfg, col_name)
            st.markdown('</div>', unsafe_allow_html=True)

    # Hidden real button — triggered by the FAB via JS
    if st.button("__FAB__", key="upload_fab_real"):
        st.session_state["show_upload"] = not show_up
        st.rerun()

    # Inject FAB: hides the real button, renders a fixed floating one instead
    _fab_icon  = "✕" if show_up else "📎"
    _fab_label = "Close" if show_up else "Upload"
    st.markdown(f"""
<script>
(function() {{
  function mountFAB() {{
    // hide the real trigger button
    var btns = Array.from(document.querySelectorAll('button'));
    var trigger = btns.find(function(b) {{ return b.innerText.trim() === '__FAB__'; }});
    if (!trigger) {{ setTimeout(mountFAB, 150); return; }}
    trigger.parentElement.style.display = 'none';

    // remove stale FAB
    var old = document.getElementById('__upload_fab__');
    if (old) old.remove();

    // create FAB
    var fab = document.createElement('button');
    fab.id = '__upload_fab__';
    fab.innerHTML = '{_fab_icon} {_fab_label}';
    fab.style.cssText = [
      'position:fixed','bottom:80px','right:20px','z-index:9999',
      'background:#3b82f6','color:#fff','border:none','border-radius:28px',
      'padding:10px 20px','font-size:14px','font-weight:600','cursor:pointer',
      'box-shadow:0 4px 20px rgba(59,130,246,0.45)',
      'transition:transform .1s,box-shadow .1s',
      'display:flex','align-items:center','gap:6px'
    ].join(';');
    fab.onmouseenter = function() {{ fab.style.transform='scale(1.05)'; fab.style.boxShadow='0 6px 24px rgba(59,130,246,0.55)'; }};
    fab.onmouseleave = function() {{ fab.style.transform='scale(1)'; fab.style.boxShadow='0 4px 20px rgba(59,130,246,0.45)'; }};
    fab.onclick = function() {{ trigger.click(); }};
    document.body.appendChild(fab);
  }}
  mountFAB();
  setTimeout(mountFAB, 400);
}})();
</script>
""", unsafe_allow_html=True)

    # ── Chat input ────────────────────────────────────────────────────────────
    chat_placeholder = ("Upload documents first…" if doc_count == 0
                        else "Ask anything about your documents…")
    if question := st.chat_input(chat_placeholder):
        if doc_count == 0:
            st.warning("Click the 📎 Upload button at the bottom-right to upload documents first.")
        else:
            messages.append({"role": "user", "content": question})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Searching and generating…"):
                    try:
                        p      = _pipeline(chroma_dir, cfg_json, col_name)
                        result = p.query(question)
                        chunks = result.get("chunks", [])
                        answer = result["answer"]

                        _not_found_phrases = (
                            "i don't know based on the provided context",
                            "i do not know based on the provided context",
                            "not in the context",
                            "not mentioned in the context",
                            "does not provide",
                            "not provided in the context",
                        )
                        is_not_found = any(ph in answer.lower() for ph in _not_found_phrases)

                        ev  = RAGEvaluator()
                        raw = ev.evaluate(question, answer, result.get("context", ""))
                        scores = {
                            **raw,
                            "retrieval_ms":  result.get("retrieval_ms", "?"),
                            "generation_ms": result.get("generation_ms", "?"),
                        }

                        try:
                            record_question(
                                username=user,
                                session_id=sid,
                                collection_name=col_name,
                                question=question,
                                composite_score=raw.get("composite", 0.0),
                            )
                        except Exception:
                            pass

                        st.markdown(answer)

                        if is_not_found:
                            hints = _sample_topics(chroma_dir, col_name)
                            hints_html = ""
                            if hints:
                                items = "".join(
                                    f'<li style="margin:.15rem 0;line-height:1.4">"{h}…"</li>'
                                    for h in hints
                                )
                                hints_html = (
                                    f'<div style="margin-top:.5rem;font-size:.75rem;color:#78350f">'
                                    f'<b>Your documents contain passages like:</b>'
                                    f'<ul style="margin:.3rem 0 0 1rem;padding:0">{items}</ul>'
                                    f'Try asking questions about those topics.</div>'
                                )
                            st.markdown(
                                f'<div style="background:{AMBER_L};border:1px solid {AMBER};'
                                f'border-radius:10px;padding:.6rem .9rem;margin-top:.4rem;'
                                f'font-size:.8rem;color:#92400e">'
                                f'<b>⚠️ Not found in your documents.</b> '
                                f'The retrieved passages do not contain a direct answer. '
                                f'Try rephrasing or check that your PDF is text-based (not scanned).'
                                f'{hints_html}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                _score_badge("Faithfulness", scores.get("faithfulness", 0)) + " " +
                                _score_badge("Relevance",    scores.get("relevance",    0)) + " " +
                                _score_badge("Composite",    scores.get("composite",    0)) +
                                f' <span style="font-size:.67rem;color:{MUTED};margin-left:.35rem">'
                                f'⏱ {scores.get("retrieval_ms","?")} ms · '
                                f'{scores.get("generation_ms","?")} ms</span>',
                                unsafe_allow_html=True,
                            )
                        if chunks:
                            with st.expander(f"📎 {len(chunks)} sources"):
                                for i, chunk in enumerate(chunks, 1):
                                    src   = os.path.basename(chunk.get("metadata", {}).get("source", "")) or "—"
                                    score = chunk.get("score", 0)
                                    st.markdown(
                                        f'<div style="margin-bottom:.6rem;padding:.75rem;background:{BG};'
                                        f'border-radius:9px;border-left:3px solid {P}">'
                                        f'<div style="font-size:.66rem;font-weight:700;color:{MUTED};margin-bottom:.25rem">'
                                        f'#{i} · {score:.3f} · {src}</div>'
                                        f'<div style="font-size:.81rem;color:{TEXT};line-height:1.6">'
                                        f'{chunk["text"][:400]}{"…" if len(chunk["text"])>400 else ""}'
                                        f'</div></div>',
                                        unsafe_allow_html=True,
                                    )

                        messages.append({
                            "role":    "assistant",
                            "content": answer,
                            "scores":  scores,
                            "chunks":  chunks,
                        })

                        new_title = None
                        if len(messages) == 2:
                            new_title = question[:55] + ("…" if len(question) > 55 else "")

                        st.session_state["chat_messages"] = messages
                        save_messages(user, sid, messages, title=new_title)

                    except Exception as exc:
                        err_str = str(exc)
                        if "daily token limit" in err_str.lower() or "tokens per day" in err_str.lower():
                            st.warning(err_str)
                        else:
                            st.error(f"Error: {err_str}")
                        messages.append({"role": "assistant", "content": f"Error: {err_str}"})
                        st.session_state["chat_messages"] = messages
                        save_messages(user, sid, messages)


# ─────────────────────────────────────────────────────────────────────────────
#  UPLOAD WIDGET
# ─────────────────────────────────────────────────────────────────────────────
def _upload_widget(chroma_dir: str, cfg: RAGConfig, collection_name: str) -> None:
    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        uploaded_files = st.file_uploader(
            "Files (PDF, DOCX, EPUB, TXT, MD)",
            type=["pdf", "txt", "md", "docx", "epub"],
            accept_multiple_files=True,
            key="ul_files",
        )
    with right:
        urls_text = st.text_area("URLs (one per line)", height=100,
                                 placeholder="https://example.com/article",
                                 key="ul_urls",
                                 label_visibility="visible")
        with st.expander("⚙️ Chunk settings"):
            oa, ob = st.columns(2)
            with oa: st.number_input("Chunk size", 128, 4096, _USER_CHUNK_SIZE, 64, key="ul_cs")
            with ob: st.number_input("Overlap",    0,   512,  _USER_OVERLAP,    16,  key="ul_ov")

    ch_size = int(st.session_state.get("ul_cs", _USER_CHUNK_SIZE))
    ov_size = int(st.session_state.get("ul_ov", _USER_OVERLAP))
    urls    = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]

    if st.button("⚡  Index into Knowledge Base", type="primary", use_container_width=True, key="ul_btn"):
        # Read all file bytes eagerly before Streamlit can reset the uploader
        file_items = []
        for f in (uploaded_files or []):
            try:
                file_items.append((f.name, f.read()))
            except Exception as exc:
                st.error(f"Could not read **{f.name}**: {exc}")

        url_items = [(u, None) for u in urls]
        all_items = file_items + url_items

        if not all_items:
            st.warning("Upload at least one file or paste a URL first.")
            return

        pipeline = RAGPipeline(cfg, chroma_dir=chroma_dir, collection_name=collection_name)
        total, failed = 0, 0
        prog = st.progress(0, text="Starting…")

        for i, (label, content) in enumerate(all_items):
            prog.progress(i / len(all_items), text=f"Processing {i+1}/{len(all_items)}: {label[:40]}…")
            is_url = content is None
            try:
                if is_url:
                    from .file_processor import load_from_url as _url
                    docs = _url(label, ch_size, ov_size)
                else:
                    from .file_processor import load_from_uploaded_bytes as _ub
                    docs = _ub(label, content, ch_size, ov_size)

                if docs:
                    pipeline.index_documents(docs)
                    total += len(docs)
                    # Show first 120 chars of extracted text so user can verify
                    preview = docs[0]["text"][:120].replace("\n", " ")
                    st.success(
                        f"✅ **{label}** — {len(docs)} chunks indexed\n\n"
                        f"*Preview: {preview}…*"
                    )
                else:
                    failed += 1
                    ext = label.rsplit(".", 1)[-1].lower() if "." in label else ""
                    hint = ""
                    if ext == "pdf":
                        hint = (" This PDF appears to be a **scanned image** — "
                                "`pypdf` can only read text-based PDFs. "
                                "Try running it through an OCR tool (e.g. Adobe, "
                                "Smallpdf) to convert it to a searchable PDF first.")
                    st.error(f"⚠️ **{label}** — 0 chunks extracted.{hint}")

            except Exception as exc:
                failed += 1
                st.error(f"❌ **{label}** failed: {exc}")

        prog.progress(1.0, text="Done!")

        if total:
            _pipeline.clear()
            new_count = _doc_count(chroma_dir, collection_name)
            if failed:
                st.warning(
                    f"Indexed **{total} chunks** from {len(all_items)-failed} file(s). "
                    f"**{failed} file(s) failed** — see errors above."
                )
            else:
                st.success(
                    f"All {len(all_items)} file(s) indexed — "
                    f"knowledge base now has **{new_count:,} chunks**. "
                    f"Ask your questions below! 👇"
                )
            st.rerun()
        elif failed:
            st.error(
                "No chunks were indexed. All files failed. "
                "Common reasons:\n"
                "- **Scanned/image PDF** — not text-based, pypdf returns blank\n"
                "- **Password-protected PDF** — remove protection first\n"
                "- **Corrupted file** — try re-downloading\n"
                "- **DOCX with only images** — add actual text content"
            )

    # ── Knowledge Base file list with delete buttons ───────────────────────
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    kb_files = _indexed_files(chroma_dir, collection_name)
    if kb_files:
        total_chunks = sum(f["chunks"] for f in kb_files)
        hdr_col, clear_col = st.columns([4, 1])
        with hdr_col:
            st.markdown(
                f'<div style="font-size:.72rem;font-weight:700;color:{MUTED};'
                f'text-transform:uppercase;letter-spacing:.06em;padding:.4rem 0 .25rem">'
                f'📚 This Chat — {len(kb_files)} file(s) · {total_chunks:,} chunks</div>',
                unsafe_allow_html=True,
            )
        with clear_col:
            if st.button("🗑 Clear all", key="kb_clear_all", help="Remove all documents from this chat"):
                _clear_kb(chroma_dir, collection_name)
                _pipeline.clear()
                st.rerun()

        for f in kb_files:
            icon = ("📄" if f["name"].endswith(".pdf") else
                    "📝" if f["name"].endswith((".docx", ".doc")) else
                    "📖" if f["name"].endswith(".epub") else
                    "🌐" if f["name"].startswith("http") else "📃")
            name_col, chunk_col, del_col = st.columns([5, 1, 1])
            with name_col:
                st.markdown(
                    f'<div style="font-size:.8rem;color:{TEXT};padding:.35rem 0;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                    f'{icon} {f["name"]}</div>',
                    unsafe_allow_html=True,
                )
            with chunk_col:
                st.markdown(
                    f'<div style="font-size:.72rem;color:{MUTED};padding:.4rem 0;text-align:right">'
                    f'{f["chunks"]:,} chunks</div>',
                    unsafe_allow_html=True,
                )
            with del_col:
                if st.button("✕", key=f"del_kb_{f['name']}", help=f"Remove {f['name']}"):
                    removed = _delete_file_from_kb(chroma_dir, collection_name, f["name"])
                    _pipeline.clear()
                    st.success(f"Removed {removed} chunks from {f['name']}")
                    st.rerun()
    else:
        st.markdown(
            f'<div style="padding:.5rem .9rem;background:{AMBER_L};'
            f'border:1px solid {AMBER};border-radius:8px;font-size:.78rem;color:#92400e">'
            f'📭 No documents for this chat yet. Upload files above and click Index.</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def _analytics(storage: RAGStorage) -> None:
    st.markdown(f'<div style="padding:2rem 2.5rem;background:{BG}">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1.35rem;font-weight:800;color:{TEXT};margin-bottom:1rem">Analytics</div>',
                unsafe_allow_html=True)

    sn      = st.text_input("Study name", value="nightly", key="an_sn")
    results = storage.get_query_results(study_name=sn, limit=500)
    trials  = storage.get_trials(study_name=sn)

    t1, t2, t3 = st.tabs(["📈 Score Trends", "🔧 Hyperparameters", "📋 Query History"])

    with t1:
        if results:
            composites = [r["composite"]    for r in results if r["composite"]    is not None]
            faith_vals = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
            rel_vals   = [r["relevance"]    for r in results if r["relevance"]    is not None]
            fig = go.Figure()
            if composites:
                fig.add_trace(go.Scatter(y=composites[::-1], mode="lines+markers", name="Composite",
                    line=dict(color=P, width=2.5), marker=dict(size=5)))
            if faith_vals:
                fig.add_trace(go.Scatter(y=faith_vals[::-1], mode="lines", name="Faithfulness",
                    line=dict(color=GREEN, width=1.5, dash="dot")))
            if rel_vals:
                fig.add_trace(go.Scatter(y=rel_vals[::-1], mode="lines", name="Relevance",
                    line=dict(color=AMBER, width=1.5, dash="dot")))
            fig.update_layout(title="Score over Time", yaxis=dict(range=[0, 1]), **_pt())
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet. Chat to generate query history.")

    with t2:
        if trials:
            imp = compute_hyperparameter_importance(storage, sn)
            if imp:
                idf = pd.DataFrame({"Parameter": list(imp.keys()), "Importance": list(imp.values())}) \
                        .sort_values("Importance", ascending=False)
                fig = px.bar(idf, x="Importance", y="Parameter", orientation="h",
                             color="Importance", color_continuous_scale=["#e8eaf6", P])
                fig.update_layout(title="Hyperparameter Importance", **_pt())
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No optimization trials yet.")

    with t3:
        if results:
            df   = pd.DataFrame(results)
            cols = [c for c in ["question","faithfulness","relevance","composite"] if c in df.columns]
            st.dataframe(df[cols].head(200), column_config={
                "question":     st.column_config.TextColumn("Question", width="large"),
                "faithfulness": st.column_config.ProgressColumn("Faithfulness", min_value=0, max_value=1),
                "relevance":    st.column_config.ProgressColumn("Relevance",    min_value=0, max_value=1),
                "composite":    st.column_config.ProgressColumn("Composite",    min_value=0, max_value=1),
            }, use_container_width=True, hide_index=True)
        else:
            st.info("No queries found.")
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
def _configuration() -> None:
    from .auth import change_password as _cp, list_users

    st.markdown(f'<div style="padding:2rem 2.5rem;background:{BG}">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1.35rem;font-weight:800;color:{TEXT};margin-bottom:1.5rem">Configuration</div>',
                unsafe_allow_html=True)

    t1, t2 = st.tabs(["⚙️ RAG Config", "👤 Account"])

    with t1:
        cfg   = _load_optuna_cfg(username=st.session_state.get("auth_user"))
        items = list(cfg.to_dict().items())
        cols  = st.columns(4)
        for i, (k, v) in enumerate(items):
            with cols[i % 4]:
                st.markdown(
                    f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;'
                    f'padding:1rem 1.25rem;margin-bottom:.75rem">'
                    f'<div style="font-size:.62rem;font-weight:700;color:{MUTED};text-transform:uppercase;'
                    f'letter-spacing:.07em;margin-bottom:.4rem">{k.replace("_"," ")}</div>'
                    f'<div style="font-size:1rem;font-weight:700;color:{TEXT}">{v}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0">', unsafe_allow_html=True)
        st.json(cfg.to_dict())

    with t2:
        user = st.session_state.get("auth_user", "")
        st.markdown(f'<div style="font-size:.95rem;font-weight:700;color:{TEXT};margin-bottom:.75rem">'
                    'Change Password</div>', unsafe_allow_html=True)
        with st.form("cpw"):
            op  = st.text_input("Current password", type="password")
            np  = st.text_input("New password",     type="password")
            np2 = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Update password", type="primary"):
                if np != np2: st.error("Passwords do not match.")
                else:
                    err = _cp(user, op, np)
                    st.error(err) if err else st.success("Password updated.")

        if user == os.environ.get("AUTH_USERNAME", "admin"):
            st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0">'
                        f'<div style="font-size:.95rem;font-weight:700;color:{TEXT};margin-bottom:.6rem">'
                        'Users</div>', unsafe_allow_html=True)
            for u in list_users():
                badge_html = (
                    f'<span style="background:{P_L};color:{P};border-radius:99px;'
                    f'padding:.15rem .55rem;font-size:.68rem;font-weight:700">admin</span>'
                    if u == user else
                    f'<span style="background:#f3f4f6;color:{MUTED};border-radius:99px;'
                    f'padding:.15rem .55rem;font-size:.68rem;font-weight:700">user</span>'
                )
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:.65rem;padding:.55rem 0;'
                    f'border-bottom:1px solid {BORDER}">'
                    f'<div style="width:26px;height:26px;border-radius:50%;background:{P};'
                    f'color:#fff;display:flex;align-items:center;justify-content:center;'
                    f'font-size:.7rem;font-weight:700">{u[0].upper()}</div>'
                    f'<span style="font-size:.875rem;font-weight:500;color:{TEXT}">{u}</span>'
                    f'{badge_html}</div>',
                    unsafe_allow_html=True,
                )
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY
# ─────────────────────────────────────────────────────────────────────────────
def _apply_hf_token() -> None:
    """Pick the first available HF_TOKEN1/2/3, fall back to HF_TOKEN."""
    for i in range(1, 10):
        t = os.environ.get(f"HF_TOKEN{i}", "").strip()
        if t:
            os.environ["HF_TOKEN"] = t
            os.environ["HUGGINGFACE_HUB_TOKEN"] = t
            return
    # plain HF_TOKEN already set — nothing to do


def run_dashboard(db_path: str = "./rag_results.db", chroma_dir: str = "./chroma_db") -> None:
    st.set_page_config(page_title="AutoRAG", page_icon="🤖",
                       layout="wide", initial_sidebar_state="expanded")
    from dotenv import load_dotenv
    load_dotenv()
    _apply_hf_token()

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    if not _auth():
        st.stop()

    page    = _sidebar(chroma_dir)
    storage = _store(db_path)

    if   page == "Chat":          _chat_page(chroma_dir)
    elif page == "Analytics":     _analytics(storage)
    elif page == "Configuration": _configuration()
