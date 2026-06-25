"""Persistent chat session storage — one JSON file per user."""
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

_STORE_DIR = "chat_sessions"


def _user_file(username: str) -> str:
    os.makedirs(_STORE_DIR, exist_ok=True)
    return os.path.join(_STORE_DIR, f"{username}.json")


def _load(username: str) -> Dict[str, dict]:
    path = _user_file(username)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(username: str, sessions: Dict[str, dict]) -> None:
    with open(_user_file(username), "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, default=str)


# ── Public API ────────────────────────────────────────────────────────────────

def list_sessions(username: str) -> List[dict]:
    """Return all sessions sorted newest-first."""
    sessions = _load(username)
    return sorted(sessions.values(), key=lambda s: s.get("updated_at", ""), reverse=True)


def create_session(username: str) -> str:
    """Create a new empty session and return its ID."""
    sessions = _load(username)
    sid = uuid.uuid4().hex[:10]
    now = datetime.now().isoformat()
    sessions[sid] = {
        "id": sid,
        "title": "New Chat",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    _save(username, sessions)
    return sid


def get_session(username: str, sid: str) -> Optional[dict]:
    return _load(username).get(sid)


def save_messages(username: str, sid: str, messages: List[dict], title: Optional[str] = None) -> None:
    """Persist the message list for a session. Strips large chunk texts to save space."""
    sessions = _load(username)
    if sid not in sessions:
        return
    sessions[sid]["messages"] = [_slim(m) for m in messages]
    sessions[sid]["updated_at"] = datetime.now().isoformat()
    if title:
        sessions[sid]["title"] = title[:60]
    _save(username, sessions)


def delete_session(username: str, sid: str) -> None:
    sessions = _load(username)
    sessions.pop(sid, None)
    _save(username, sessions)


def rename_session(username: str, sid: str, title: str) -> None:
    sessions = _load(username)
    if sid in sessions:
        sessions[sid]["title"] = title[:60]
        _save(username, sessions)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slim(msg: dict) -> dict:
    """Reduce stored chunk size to keep files manageable."""
    m = {k: v for k, v in msg.items() if k != "chunks"}
    if msg.get("chunks"):
        m["chunks"] = [
            {
                "text":     c.get("text", "")[:400],
                "metadata": c.get("metadata", {}),
                "score":    c.get("score", 0),
            }
            for c in msg["chunks"]
        ]
    return m
