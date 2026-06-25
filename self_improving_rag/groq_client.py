"""
Groq API key rotation pool.

Loads GROQ_API_KEY1 / GROQ_API_KEY2 / GROQ_API_KEY3 from environment.
On a 429 rate-limit error, automatically switches to the next key and retries.
Falls back to GROQ_API_KEY if the numbered keys are not set.
"""

import logging
import os
import time
from threading import Lock
from typing import List, Optional

from groq import Groq

logger = logging.getLogger(__name__)


def _load_keys() -> List[str]:
    keys = []
    for i in range(1, 10):
        k = os.environ.get(f"GROQ_API_KEY{i}", "").strip()
        if k:
            keys.append(k)
    if not keys:
        fallback = os.environ.get("GROQ_API_KEY", "").strip()
        if fallback:
            keys.append(fallback)
    return keys


class GroqKeyPool:
    """Round-robin key pool with automatic 429 failover."""

    def __init__(self, keys: Optional[List[str]] = None, retry_delay: float = 2.0):
        self._keys = keys or _load_keys()
        if not self._keys:
            raise ValueError(
                "No Groq API keys found. Set GROQ_API_KEY1 / GROQ_API_KEY2 / ... in .env"
            )
        self._index = 0
        self._lock = Lock()
        self._retry_delay = retry_delay
        self._clients = [Groq(api_key=k) for k in self._keys]
        logger.info("GroqKeyPool: loaded %d key(s)", len(self._keys))

    @property
    def current(self) -> Groq:
        with self._lock:
            return self._clients[self._index]

    def _rotate(self) -> Groq:
        with self._lock:
            self._index = (self._index + 1) % len(self._clients)
            logger.warning(
                "Rotated to Groq key %d/%d", self._index + 1, len(self._clients)
            )
            return self._clients[self._index]

    def chat_completions_create(self, retries: int = 6, **kwargs):
        """
        Call chat.completions.create with automatic key rotation on 429.
        Tries every key up to `retries` total attempts.
        """
        last_exc = None
        for attempt in range(retries):
            client = self.current
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                msg = str(exc)
                is_rate_limit = "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower()
                if is_rate_limit:
                    logger.warning(
                        "Rate limit on key %d (attempt %d/%d) — rotating",
                        self._index + 1, attempt + 1, retries,
                    )
                    self._rotate()
                    wait = self._retry_delay * (2 ** min(attempt, 4))
                    time.sleep(wait)
                    last_exc = exc
                else:
                    raise
        raise last_exc


_DEFAULT_POOL: Optional[GroqKeyPool] = None
_POOL_LOCK = Lock()


def get_pool() -> GroqKeyPool:
    """Return the process-level shared key pool (lazy init).

    Loads .env automatically if keys are not already in the environment.
    Raises ValueError only when making real API calls with no keys found.
    """
    global _DEFAULT_POOL
    with _POOL_LOCK:
        if _DEFAULT_POOL is None:
            # Try loading .env if keys aren't set yet
            if not os.environ.get("GROQ_API_KEY1") and not os.environ.get("GROQ_API_KEY"):
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                except ImportError:
                    pass
            _DEFAULT_POOL = GroqKeyPool()
    return _DEFAULT_POOL
