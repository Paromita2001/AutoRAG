import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import instructor
from groq import Groq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _OpenRouterFallback(Exception):
    """Sentinel: Groq TPD exhausted, caller should use OpenRouter scoring."""


class _SchemaFallback(Exception):
    """Sentinel: model returned score embedded in text — extracted score carried here."""
    def __init__(self, score: float):
        self.score = score


def _extract_score_from_error(msg: str) -> Optional[float]:
    """Pull a 0–1 float out of an instructor failed_generation error string."""
    # Try explicit {"score": X} first
    m = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', msg)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    # "score is 0.8" / "score: 0.8" in natural language
    m = re.search(r'score[^0-9]{0,10}([0-9]\.[0-9]+)', msg, re.IGNORECASE)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    # Any bare float in the message
    m = re.search(r'\b(0\.[0-9]+|1\.0+)\b', msg)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


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
        self._tpd_hits: set = set()     # persists across calls — tracks which key orgs hit daily limit

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
                max_tokens=120,
            )

        # Production path — retry across all keys.
        pool = self._get_pool()
        retries = max(6, len(pool._keys) * 2)
        last_exc = None
        for attempt in range(retries):
            client = instructor.from_groq(pool.current)
            try:
                return client.chat.completions.create(
                    model="llama-3.3-70b-versatile",   # confirmed working on Groq free tier; smaller models decommissioned
                    response_model=response_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=120,                  # score + brief reason — no need for more
                )
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "rate_limit" in msg.lower():
                    import time
                    if "tokens per day" in msg.lower() or "tpd" in msg.lower():
                        self._tpd_hits.add(pool._index)
                        logger.warning(
                            "[evaluator] TPD hit on key %d/%d (%d/%d keys exhausted)",
                            pool._index + 1, len(pool._keys),
                            len(self._tpd_hits), len(pool._keys),
                        )
                        if len(self._tpd_hits) >= len(pool._keys):
                            # All orgs exhausted — fall back to OpenRouter
                            logger.warning("[evaluator] All keys TPD exhausted — trying OpenRouter")
                            from .openrouter_client import available as _or_avail
                            if _or_avail():
                                raise _OpenRouterFallback()
                            raise RuntimeError(
                                "⏳ Groq daily token limit reached on all keys. "
                                "Add OPENROUTER_API_KEY to .env for automatic fallback."
                            ) from exc
                        # More keys to try — rotate immediately, no sleep needed
                        pool._rotate()
                        last_exc = exc
                        continue
                    pool._rotate()
                    time.sleep(2.0 * (2 ** min(attempt, 3)))
                    last_exc = exc
                elif "400" in msg and ("tool_use_failed" in msg or "tool call validation" in msg.lower()):
                    # Model wrote score in plain text instead of JSON field.
                    # Extract score from the failed_generation payload rather than discarding it.
                    score = _extract_score_from_error(msg)
                    if score is not None:
                        logger.warning(
                            "[evaluator] Schema mismatch — extracted score %.2f from error text", score
                        )
                        raise _SchemaFallback(score)
                    # Can't extract — try next key with a slightly rephrased prompt
                    logger.warning("[evaluator] Schema mismatch with no extractable score — rotating key")
                    pool._rotate()
                    last_exc = exc
                else:
                    raise
        raise last_exc

    # ── Public API ────────────────────────────────────────────────────────────

    _EVAL_SYSTEM_PROMPT = (
        "You are an objective evaluator of AI-generated answers. "
        "Be strict and precise. Return only the JSON score object requested."
    )

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        prompt = (
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}\n\n"
            "Rate how faithfully this answer is grounded in the context. "
            "Score 0.0 (fully hallucinated) to 1.0 (fully grounded)."
        )
        try:
            result = self._call_with_rotation(
                response_model=FaithfulnessScore,
                messages=[
                    {"role": "system", "content": self._EVAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return float(result.score)
        except _SchemaFallback as sf:
            return sf.score
        except _OpenRouterFallback:
            from .openrouter_client import evaluate_score
            return evaluate_score(prompt)
        except Exception as exc:
            logger.error("Faithfulness evaluation failed: %s", exc)
            return 0.5

    def evaluate_relevance(self, question: str, answer: str) -> float:
        prompt = (
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            "Rate how relevant this answer is to the question. "
            "Score 0.0 (completely irrelevant) to 1.0 (perfectly answers it)."
        )
        try:
            result = self._call_with_rotation(
                response_model=RelevanceScore,
                messages=[
                    {"role": "system", "content": self._EVAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return float(result.score)
        except _SchemaFallback as sf:
            return sf.score
        except _OpenRouterFallback:
            from .openrouter_client import evaluate_score
            return evaluate_score(prompt)
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
