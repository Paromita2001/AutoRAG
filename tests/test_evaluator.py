from unittest.mock import MagicMock
import pytest
from self_improving_rag.evaluator import RAGEvaluator, FaithfulnessScore, RelevanceScore


def _make_evaluator():
    ev = RAGEvaluator(groq_api_key="fake_key")
    return ev


def _mock_instructor(faithfulness=0.9, relevance=0.8):
    client = MagicMock()

    def _create(*args, response_model=None, **kwargs):
        if response_model is FaithfulnessScore:
            return FaithfulnessScore(score=faithfulness, reason="grounded")
        if response_model is RelevanceScore:
            return RelevanceScore(score=relevance, reason="relevant")
        raise ValueError(f"Unknown response_model: {response_model}")

    client.chat.completions.create.side_effect = _create
    return client


def test_evaluate_faithfulness_returns_float():
    ev = _make_evaluator()
    ev._instructor_client = _mock_instructor(faithfulness=0.85)
    score = ev.evaluate_faithfulness("answer", "context")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_evaluate_relevance_returns_float():
    ev = _make_evaluator()
    ev._instructor_client = _mock_instructor(relevance=0.75)
    score = ev.evaluate_relevance("question", "answer")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_composite_formula():
    ev = _make_evaluator()
    ev._instructor_client = _mock_instructor(faithfulness=1.0, relevance=0.0)
    result = ev.evaluate("q", "a", "ctx")
    assert abs(result["composite"] - 0.6) < 0.001


def test_composite_formula_both_high():
    ev = _make_evaluator()
    ev._instructor_client = _mock_instructor(faithfulness=1.0, relevance=1.0)
    result = ev.evaluate("q", "a", "ctx")
    assert abs(result["composite"] - 1.0) < 0.001


def test_evaluate_returns_all_keys():
    ev = _make_evaluator()
    ev._instructor_client = _mock_instructor()
    result = ev.evaluate("question", "answer", "context")
    assert "faithfulness" in result
    assert "relevance" in result
    assert "composite" in result


def test_faithfulness_fallback_on_exception():
    ev = _make_evaluator()
    ev._instructor_client = MagicMock()
    ev._instructor_client.chat.completions.create.side_effect = RuntimeError("boom")
    score = ev.evaluate_faithfulness("a", "ctx")
    assert score == 0.5


def test_relevance_fallback_on_exception():
    ev = _make_evaluator()
    ev._instructor_client = MagicMock()
    ev._instructor_client.chat.completions.create.side_effect = RuntimeError("boom")
    score = ev.evaluate_relevance("q", "a")
    assert score == 0.5
