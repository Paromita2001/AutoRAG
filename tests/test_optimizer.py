from unittest.mock import MagicMock, patch
import pytest
from self_improving_rag.config import RAGConfig
from self_improving_rag.optimizer import RAGOptimizer, SEARCH_SPACE


@pytest.fixture
def storage(tmp_path):
    from self_improving_rag.storage import RAGStorage
    return RAGStorage(db_path=str(tmp_path / "opt.db"))


def _mock_pipeline_query(question):
    return {
        "question": question,
        "answer": "Test answer",
        "context": "Test context",
        "chunks": [],
        "config": {},
    }


def _mock_evaluator():
    ev = MagicMock()
    ev.evaluate.return_value = {"faithfulness": 0.8, "relevance": 0.7, "composite": 0.76}
    return ev


def test_search_space_keys():
    expected = {"embedding_model", "chunk_size", "overlap", "top_k", "temperature", "similarity_threshold"}
    assert set(SEARCH_SPACE.keys()) == expected


def test_optimizer_runs_and_returns_best_config(storage):
    opt = RAGOptimizer(
        questions=["q1", "q2"],
        storage=storage,
        chroma_dir="/tmp/chroma_test",
        study_name="opt_test",
        n_trials=2,
    )
    opt._evaluator = _mock_evaluator()

    with patch("self_improving_rag.optimizer.RAGPipeline") as MockPipeline:
        mock_p = MagicMock()
        mock_p.query.side_effect = _mock_pipeline_query
        MockPipeline.return_value = mock_p
        result = opt.run()

    assert result["best_config"] is not None
    assert isinstance(result["best_config"], RAGConfig)
    assert result["best_score"] is not None
    assert result["n_trials"] >= 1


def test_optimizer_stores_trials(storage):
    opt = RAGOptimizer(
        questions=["q1"],
        storage=storage,
        chroma_dir="/tmp/chroma_test",
        study_name="trial_test",
        n_trials=2,
    )
    opt._evaluator = _mock_evaluator()

    with patch("self_improving_rag.optimizer.RAGPipeline") as MockPipeline:
        mock_p = MagicMock()
        mock_p.query.side_effect = _mock_pipeline_query
        MockPipeline.return_value = mock_p
        opt.run()

    trials = storage.get_trials("trial_test")
    assert len(trials) >= 1


def test_best_config_property(storage):
    opt = RAGOptimizer(
        questions=["q"],
        storage=storage,
        chroma_dir="/tmp",
        n_trials=1,
    )
    assert opt.best_config is None
    opt._evaluator = _mock_evaluator()
    with patch("self_improving_rag.optimizer.RAGPipeline") as MockPipeline:
        mock_p = MagicMock()
        mock_p.query.side_effect = _mock_pipeline_query
        MockPipeline.return_value = mock_p
        opt.run()
    assert opt.best_config is not None
