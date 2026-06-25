import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import instructor
from groq import Groq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _cache_key(question: str, answer: str, context: str) -> str:
    raw = f"{question}|||{context}|||{answer}"
    return hashlib.md5(raw.encode()).hexdigest()


class FaithfulnessScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class RelevanceScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class RAGEvaluator:
    """LLM-as-Judge evaluator.

    Production: uses GroqKeyPool (3 keys) with automatic 429 failover.
    Tests: inject ev._instructor_client = mock to bypass all real API calls.
    """

    def __init__(self, groq_api_key: Optional[str] = None, max_workers: int = 5):
        self._explicit_key = groq_api_key
        self._instructor_client = None  # tests set this directly; production leaves it None
        self._pool = None               # lazy: created on first real API call
        self.max_workers = max_workers
        self._cache: Dict[str, dict] = {}

    # ── Internal: pick instructor client for this call ────────────────────────

    def _get_instructor(self):
        """Return the active instructor client.

        - If a test has injected _instructor_client, use that.
        - Otherwise use the shared GroqKeyPool (rotates on 429).
        """
        if self._instructor_client is not None:
            return self._instructor_client
        return instructor.from_groq(self._get_pool().current)

    def _get_pool(self):
        if self._pool is None:
            from .groq_client import GroqKeyPool, get_pool
            if self._explicit_key:
                self._pool = GroqKeyPool(keys=[self._explicit_key])
            else:
                self._pool = get_pool()
        return self._pool

    def _call_with_rotation(self, response_model, messages: list):
        """Call instructor with automatic key rotation on 429."""
        if self._instructor_client is not None:
            # Test path — mock is already set, one call only.
            return self._instructor_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_model=response_model,
                messages=messages,
                temperature=0.0,
            )

        # Production path — retry across all keys.
        pool = self._get_pool()
        retries = max(6, len(pool._keys) * 2)
        last_exc = None
        for attempt in range(retries):
            client = instructor.from_groq(pool.current)
            try:
                return client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    response_model=response_model,
                    messages=messages,
                    temperature=0.0,
                )
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "rate_limit" in msg.lower():
                    import time
                    pool._rotate()
                    time.sleep(2.0 * (2 ** min(attempt, 3)))
                    last_exc = exc
                else:
                    raise
        raise last_exc

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        try:
            result = self._call_with_rotation(
                response_model=FaithfulnessScore,
                messages=[{
                    "role": "user",
                    "content": (
                        "Rate how faithfully this answer is grounded in the context.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Answer:\n{answer}\n\n"
                        "Score 0 (fully hallucinated) to 1 (fully grounded in context)."
                    ),
                }],
            )
            return float(result.score)
        except Exception as exc:
            logger.error("Faithfulness evaluation failed: %s", exc)
            return 0.5

    def evaluate_relevance(self, question: str, answer: str) -> float:
        try:
            result = self._call_with_rotation(
                response_model=RelevanceScore,
                messages=[{
                    "role": "user",
                    "content": (
                        "Rate how relevant this answer is to the question.\n\n"
                        f"Question:\n{question}\n\n"
                        f"Answer:\n{answer}\n\n"
                        "Score 0 (completely irrelevant) to 1 (perfectly answers the question)."
                    ),
                }],
            )
            return float(result.score)
        except Exception as exc:
            logger.error("Relevance evaluation failed: %s", exc)
            return 0.5

    def evaluate(self, question: str, answer: str, context: str) -> dict:
        key = _cache_key(question, answer, context)
        if key in self._cache:
            logger.debug("Cache hit for judge evaluation")
            return self._cache[key]

        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(question, answer)
        composite = 0.6 * faithfulness + 0.4 * relevance
        result = {"faithfulness": faithfulness, "relevance": relevance, "composite": composite}
        self._cache[key] = result
        return result

    def evaluate_batch(self, items: List[Dict]) -> List[dict]:
        """Evaluate a list of {question, answer, context} dicts concurrently."""
        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.evaluate, item["question"], item["answer"], item["context"]): i
                for i, item in enumerate(items)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Batch eval index %d failed: %s", idx, exc)
                    results[idx] = {"faithfulness": 0.5, "relevance": 0.5, "composite": 0.5}
        return results

    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
