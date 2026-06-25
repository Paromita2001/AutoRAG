"""
Integration tests — wire RAGPipeline + RAGEvaluator + RAGStorage together.
No real Groq / instructor calls; clients are swapped for mocks.
SentenceTransformer is loaded once at module scope (real model, slow but accurate).
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from self_improving_rag.config import RAGConfig
from self_improving_rag.evaluator import FaithfulnessScore, RAGEvaluator, RelevanceScore
from self_improving_rag.pipeline import RAGPipeline
from self_improving_rag.storage import RAGStorage


@pytest.fixture(scope="module")
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def storage(tmp_path):
    return RAGStorage(db_path=str(tmp_path / "integration.db"))


def _groq_mock():
    mock = MagicMock()
    mock.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="RAG stands for Retrieval-Augmented Generation."))
    ]
    return mock


def _instructor_mock():
    client = MagicMock()

    def _create(*args, response_model=None, **kwargs):
        if response_model is FaithfulnessScore:
            return FaithfulnessScore(score=0.85, reason="grounded in context")
        if response_model is RelevanceScore:
            return RelevanceScore(score=0.9, reason="directly answers the question")
        raise ValueError(f"Unknown model: {response_model}")

    client.chat.completions.create.side_effect = _create
    return client


def _make_pipeline(embedder, chroma_dir):
    cfg = RAGConfig(embedding_model="all-MiniLM-L6-v2", similarity_threshold=0.0)
    pipeline = RAGPipeline(cfg, chroma_dir=chroma_dir)
    pipeline._embedder = embedder
    pipeline._groq_client = _groq_mock()
    # Stub out a real collection with a mock
    col = MagicMock()
    col.query.return_value = {
        "documents": [["RAG combines retrieval with generation."]],
        "metadatas": [[{"source": "test.txt", "chunk_index": 0}]],
        "distances": [[0.05]],
    }
    pipeline._collection = col
    return pipeline


def test_pipeline_query_end_to_end(embedder, tmp_path):
    pipeline = _make_pipeline(embedder, str(tmp_path / "chroma"))
    result = pipeline.query("What is RAG?")
    assert result["answer"] == "RAG stands for Retrieval-Augmented Generation."
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["score"] > 0


def test_evaluator_scores_pipeline_output(embedder, tmp_path):
    pipeline = _make_pipeline(embedder, str(tmp_path / "chroma2"))
    result = pipeline.query("What is RAG?")

    evaluator = RAGEvaluator(groq_api_key="fake")
    evaluator._instructor_client = _instructor_mock()
    scores = evaluator.evaluate(result["question"], result["answer"], result["context"])

    assert 0 <= scores["faithfulness"] <= 1
    assert 0 <= scores["relevance"] <= 1
    assert abs(scores["composite"] - (0.6 * 0.85 + 0.4 * 0.9)) < 0.001


def test_pipeline_output_saved_to_storage(embedder, tmp_path, storage):
    pipeline = _make_pipeline(embedder, str(tmp_path / "chroma3"))
    result = pipeline.query("What is RAG?")
    evaluator = RAGEvaluator(groq_api_key="fake")
    evaluator._instructor_client = _instructor_mock()
    scores = evaluator.evaluate(result["question"], result["answer"], result["context"])

    row_id = storage.save_query_result(
        question=result["question"],
        answer=result["answer"],
        context=result["context"],
        faithfulness=scores["faithfulness"],
        relevance=scores["relevance"],
        composite=scores["composite"],
        config=result["config"],
        study_name="integration",
    )
    assert row_id >= 1
    rows = storage.get_query_results(study_name="integration")
    assert len(rows) == 1
    assert rows[0]["composite"] == pytest.approx(scores["composite"])


def test_multiple_questions_stored(embedder, tmp_path, storage):
    pipeline = _make_pipeline(embedder, str(tmp_path / "chroma4"))
    evaluator = RAGEvaluator(groq_api_key="fake")
    evaluator._instructor_client = _instructor_mock()

    questions = ["What is RAG?", "How does retrieval work?", "What is an embedding?"]
    for q in questions:
        result = pipeline.query(q)
        scores = evaluator.evaluate(result["question"], result["answer"], result["context"])
        storage.save_query_result(
            question=result["question"],
            answer=result["answer"],
            context=result["context"],
            faithfulness=scores["faithfulness"],
            relevance=scores["relevance"],
            composite=scores["composite"],
            config=result["config"],
            study_name="multi",
        )

    rows = storage.get_query_results(study_name="multi")
    assert len(rows) == 3


def test_best_config_after_storage(embedder, tmp_path, storage):
    pipeline_low = _make_pipeline(embedder, str(tmp_path / "chroma5"))
    pipeline_low._groq_client.chat.completions.create.return_value.choices[0].message.content = "low"
    evaluator = RAGEvaluator(groq_api_key="fake")

    # Low score run
    low_inst = MagicMock()
    low_inst.chat.completions.create.side_effect = lambda *a, response_model=None, **kw: (
        FaithfulnessScore(score=0.3, reason="r") if response_model is FaithfulnessScore
        else RelevanceScore(score=0.3, reason="r")
    )
    evaluator._instructor_client = low_inst
    result = pipeline_low.query("q")
    scores = evaluator.evaluate(result["question"], result["answer"], result["context"])
    storage.save_query_result(
        question="q", answer=result["answer"], context=result["context"],
        faithfulness=scores["faithfulness"], relevance=scores["relevance"],
        composite=scores["composite"], config={"chunk_size": 256}, study_name="best_test",
    )

    # High score run — different answer so cache key differs from low run
    pipeline_low._groq_client.chat.completions.create.return_value.choices[0].message.content = "high"
    high_inst = _instructor_mock()
    evaluator._instructor_client = high_inst
    result2 = pipeline_low.query("q")
    scores2 = evaluator.evaluate(result2["question"], result2["answer"], result2["context"])
    storage.save_query_result(
        question="q", answer=result2["answer"], context=result2["context"],
        faithfulness=scores2["faithfulness"], relevance=scores2["relevance"],
        composite=scores2["composite"], config={"chunk_size": 512}, study_name="best_test",
    )

    best = storage.get_best_config("best_test")
    assert best["chunk_size"] == 512
