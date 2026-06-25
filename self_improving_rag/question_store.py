"""
Collects real questions asked through the chat UI for use in nightly optimization.

Each user's questions are stored separately so the optimizer tunes on THEIR
documents and topics, not a mix of every user's questions.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_STORED = 200   # cap per user to avoid unbounded growth


def _store_path(username: str) -> str:
    os.makedirs("data", exist_ok=True)
    safe = "".join(c for c in username if c.isalnum() or c == "_")[:32]
    return f"data/questions_{safe}.json"


def _load(username: str) -> List[Dict]:
    path = _store_path(username)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _save(username: str, records: List[Dict]) -> None:
    with open(_store_path(username), "w") as f:
        json.dump(records, f, indent=2)


def record_question(
    username: str,
    session_id: str,
    collection_name: str,
    question: str,
    composite_score: float,
) -> None:
    """Save a question asked in the chat UI to that user's question bank."""
    records = _load(username)
    records.append({
        "username":        username,
        "session_id":      session_id,
        "collection_name": collection_name,
        "question":        question,
        "score":           round(composite_score, 4),
        "timestamp":       datetime.utcnow().isoformat(),
    })
    _save(username, records[-_MAX_STORED:])


def get_questions_for_optimization(username: str, max_n: int = 20) -> List[Dict]:
    """
    Return up to max_n questions for the optimizer.
    Prioritises low-scoring questions (most room to improve).
    """
    records = _load(username)
    if not records:
        return []
    seen, unique = set(), []
    for r in records:
        key = r["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    unique.sort(key=lambda r: r["score"])
    return unique[:max_n]


def best_collection(username: str) -> Optional[str]:
    """Return the collection_name this user chats with most — best optimizer target."""
    records = _load(username)
    if not records:
        return None
    counts: Dict[str, int] = {}
    for r in records:
        col = r.get("collection_name", "")
        if col:
            counts[col] = counts.get(col, 0) + 1
    return max(counts, key=counts.__getitem__) if counts else None


def question_count(username: str) -> int:
    """Total questions recorded for this user."""
    return len(_load(username))
