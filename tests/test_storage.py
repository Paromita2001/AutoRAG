import os
import tempfile
import pytest
from self_improving_rag.storage import RAGStorage


@pytest.fixture
def storage(tmp_path):
    return RAGStorage(db_path=str(tmp_path / "test.db"))


def test_save_and_retrieve_query(storage):
    row_id = storage.save_query_result(
        question="What is RAG?",
        answer="RAG is retrieval-augmented generation.",
        context="Some context about RAG.",
        faithfulness=0.9,
        relevance=0.8,
        composite=0.86,
        config={"chunk_size": 512},
        study_name="test_study",
    )
    assert row_id >= 1
    results = storage.get_query_results(study_name="test_study")
    assert len(results) == 1
    assert results[0]["question"] == "What is RAG?"
    assert results[0]["composite"] == pytest.approx(0.86)


def test_save_trial(storage):
    row_id = storage.save_trial(
        study_name="test_study",
        trial_number=0,
        params={"chunk_size": 512, "top_k": 5},
        score=0.75,
    )
    assert row_id >= 1
    trials = storage.get_trials("test_study")
    assert len(trials) == 1
    assert trials[0]["score"] == pytest.approx(0.75)


def test_get_best_config(storage):
    storage.save_query_result("q1", "a1", "ctx", 0.6, 0.6, 0.6, {"chunk_size": 256}, "s")
    storage.save_query_result("q2", "a2", "ctx", 0.9, 0.9, 0.9, {"chunk_size": 512}, "s")
    best = storage.get_best_config("s")
    assert best["chunk_size"] == 512


def test_get_best_config_no_results(storage):
    assert storage.get_best_config("nonexistent") is None


def test_filter_by_study_name(storage):
    storage.save_query_result("q1", "a", "c", 0.5, 0.5, 0.5, {}, "study_a")
    storage.save_query_result("q2", "a", "c", 0.5, 0.5, 0.5, {}, "study_b")
    results_a = storage.get_query_results(study_name="study_a")
    assert len(results_a) == 1
    assert results_a[0]["question"] == "q1"


def test_context_snippet_truncated(storage):
    long_ctx = "x" * 1000
    storage.save_query_result("q", "a", long_ctx, 0.5, 0.5, 0.5, {})
    results = storage.get_query_results()
    assert len(results[0]["context_snippet"]) <= 500


def test_multiple_trials_ordered(storage):
    for i in range(3):
        storage.save_trial("s", i, {"k": i}, float(i) * 0.1)
    trials = storage.get_trials("s")
    assert [t["trial_number"] for t in trials] == [0, 1, 2]


def test_limit_respected(storage):
    for i in range(10):
        storage.save_query_result(f"q{i}", "a", "c", 0.5, 0.5, 0.5, {})
    results = storage.get_query_results(limit=3)
    assert len(results) == 3
