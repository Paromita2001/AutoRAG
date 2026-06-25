"""
Collects real questions asked through the chat UI for use in nightly optimization.

Instead of tuning on hardcoded RAG questions, the optimizer uses questions
the user actually asked against their own documents — making tuning relevant
to their specific content and topics.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_STORE_PATH = "data/user_questions.json"
_MAX_STORED  = 200   # cap to avoid unbounded growth


def _load() -> List[Dict]:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(_STORE_PATH):
        return []
    try:
        with open(_STORE_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save(records: List[Dict]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(_STORE_PATH, "w") as f:
        json.dump(records, f, indent=2)


def record_question(
    username: str,
    session_id: str,
    collection_name: str,
    question: str,
    composite_score: float,
) -> None:
    """Save a question that was asked in the chat UI."""
    records = _load()
    records.append({
        "username":        username,
        "session_id":      session_id,
        "collection_name": collection_name,
        "question":        question,
        "score":           round(composite_score, 4),
        "timestamp":       datetime.utcnow().isoformat(),
    })
    # Keep newest _MAX_STORED records
    _save(records[-_MAX_STORED:])


def get_questions_for_optimization(max_n: int = 20) -> List[Dict]:
    """
    Return up to max_n questions for the optimizer to use.
    Prioritises low-scoring questions (hardest to answer = most room to improve).
    Falls back to all questions if fewer than 5 exist.
    """
    records = _load()
    if not records:
        return []
    # deduplicate by question text (case-insensitive)
    seen, unique = set(), []
    for r in records:
        key = r["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    # sort by score ascending — low scores first (most improvement potential)
    unique.sort(key=lambda r: r["score"])
    return unique[:max_n]


def best_collection() -> Optional[str]:
    """
    Return the collection_name that appears most in stored questions
    — i.e. the one the user chats with most, best target for optimization.
    """
    records = _load()
    if not records:
        return None
    counts: Dict[str, int] = {}
    for r in records:
        col = r.get("collection_name", "")
        if col:
            counts[col] = counts.get(col, 0) + 1
    return max(counts, key=counts.__getitem__) if counts else None


def question_count() -> int:
    return len(_load())
