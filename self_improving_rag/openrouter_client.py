"""
OpenRouter fallback client.

Used automatically when Groq's 100k daily token limit (TPD) is exhausted.
OpenRouter free models are slower but have no token-per-day cap.

Get a free key at: https://openrouter.ai  (no credit card needed)
Add to .env:  OPENROUTER_API_KEY=sk-or-v1-...
"""
import json
import logging
import os
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ── Model lists ───────────────────────────────────────────────────────────────
# Ranked by quality for RAG generation. All are free (:free suffix).
GENERATION_MODELS: List[str] = [
    "meta-llama/llama-3.3-70b-instruct:free",   # primary: 70B, reliable
    "openai/gpt-oss-20b:free",     # fallback: smaller, lower load
    "openai/gpt-oss-20b:free",                   # last resort: GPT-family
]

EVAL_MODELS: List[str] = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-20b:free",
]

# ── System prompts ────────────────────────────────────────────────────────────
GENERATION_SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Answer questions using ONLY the context provided by the user. "
    "Rules:\n"
    "1. Answer directly if the context clearly states it.\n"
    "2. If partially answered, share what you found and note gaps. "
    "Start with: 'Based on your documents:'\n"
    "3. If the context has nothing relevant, say exactly: "
    "'I don't know based on the provided context.'\n"
    "Never use training knowledge. Never invent facts."
)

FALLBACK_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. "
    "Answer the question using your general knowledge. "
    "Be accurate, concise, and honest about uncertainty."
)

EVAL_SYSTEM_PROMPT = (
    "You are an objective evaluator of AI-generated answers. "
    "Be strict and precise. Respond only with the JSON score requested."
)

# ── Singleton client ──────────────────────────────────────────────────────────
_client = None
_checked = False


def _get_client():
    global _client, _checked
    if _checked:
        return _client
    _checked = True

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        logger.info("[openrouter] OPENROUTER_API_KEY not set — fallback disabled")
        return None
    try:
        from openai import OpenAI
        _client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "AutoRAG",
            },
        )
        logger.info("[openrouter] Fallback client ready")
    except ImportError:
        logger.warning("[openrouter] openai package not installed — run: pip install openai")
    return _client


def available() -> bool:
    return _get_client() is not None


def chat_completions_create(models: List[str], **kwargs) -> Any:
    """
    Drop-in for GroqKeyPool.chat_completions_create.
    Injects a system prompt and tries each free model in order on 429.
    """
    client = _get_client()
    if client is None:
        raise RuntimeError(
            "OpenRouter fallback unavailable. "
            "Add OPENROUTER_API_KEY to .env (free at openrouter.ai)"
        )

    kwargs.pop("model", None)

    # Inject system prompt if not already present.
    # Use fallback (general knowledge) prompt when no Context block is in the user message.
    messages = kwargs.get("messages", [])
    if messages and messages[0].get("role") != "system":
        user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        system = FALLBACK_SYSTEM_PROMPT if "Context:" not in user_text else GENERATION_SYSTEM_PROMPT
        kwargs["messages"] = [{"role": "system", "content": system}] + messages

    last_exc: Optional[Exception] = None
    for model in models:
        try:
            logger.info("[openrouter] Trying %s", model)
            resp = client.chat.completions.create(model=model, **kwargs)
            logger.info("[openrouter] Success with %s", model)
            return resp
        except Exception as exc:
            msg = str(exc)
            if "404" in msg or "No endpoints found" in msg.lower():
                # Model removed or unavailable on OpenRouter — skip silently
                logger.warning("[openrouter] %s not available (404) — skipping", model)
                last_exc = exc
                continue
            if "429" in msg or "rate_limit" in msg.lower():
                # Extract retry_after_seconds from the error payload if present
                retry_wait = 5
                try:
                    import json as _json
                    raw = getattr(exc, "response", None)
                    body = raw.json() if raw is not None else _json.loads(msg.split(" - ", 1)[-1])
                    retry_wait = int(
                        body.get("error", {}).get("metadata", {}).get("retry_after_seconds", 5)
                    )
                except Exception:
                    pass
                logger.warning(
                    "[openrouter] %s rate-limited — waiting %ds then trying next model",
                    model, retry_wait,
                )
                import time as _time
                _time.sleep(min(retry_wait, 30))
                last_exc = exc
                continue
            raise

    raise last_exc or RuntimeError("All OpenRouter free models rate-limited")


def evaluate_score(prompt: str) -> float:
    """
    Ask OpenRouter for a 0–1 score without instructor/structured output.
    Uses a system prompt for consistent scoring. Falls back to 0.5 on failure.
    """
    client = _get_client()
    if client is None:
        return 0.5

    user_msg = prompt + '\n\nReply with ONLY this JSON on one line: {"score": <0.0-1.0>}'

    for model in EVAL_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=30,
            )
            content = resp.choices[0].message.content
            if not content:
                logger.warning("[openrouter] %s returned empty content — skipping", model)
                continue
            text = content.strip()

            # Try strict JSON first
            try:
                return float(json.loads(text)["score"])
            except Exception:
                pass

            # Regex fallback
            m = re.search(r"\b(1\.0+|0\.\d+|[01])\b", text)
            if m:
                return max(0.0, min(1.0, float(m.group(1))))

        except Exception as exc:
            logger.warning("[openrouter] eval failed with %s: %s", model, exc)
            continue

    logger.warning("[openrouter] all eval models failed — returning 0.5")
    return 0.5
