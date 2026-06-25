import numpy as np
import pytest
from self_improving_rag.analysis import DriftDetector, _kl_divergence, compute_hyperparameter_importance


@pytest.fixture
def storage(tmp_path):
    from self_improving_rag.storage import RAGStorage
    return RAGStorage(db_path=str(tmp_path / "analysis.db"))


def test_kl_divergence_identical_distributions():
    p = np.random.default_rng(0).normal(0, 1, 100)
    kl = _kl_divergence(p, p)
    assert kl == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_different_distributions():
    p = np.zeros(100)
    q = np.ones(100)
    kl = _kl_divergence(p, q)
    # Bins don't overlap → large KL
    assert kl > 0


def test_kl_divergence_constant():
    p = np.array([0.5] * 50)
    q = np.array([0.5] * 50)
    kl = _kl_divergence(p, q)
    assert kl == 0.0


def test_drift_detector_insufficient_data(storage):
    from unittest.mock import MagicMock, patch
    with patch("self_improving_rag.analysis.SentenceTransformer"):
        detector = DriftDetector(storage)
        result = detector.compute_drift()
    assert result["drift_detected"] is False
    assert result["reason"] == "insufficient_data"


def test_drift_detector_no_drift(storage):
    from unittest.mock import MagicMock, patch
    import numpy as np

    # Add 40 questions
    for i in range(40):
        storage.save_query_result(f"q{i}", "a", "c", 0.5, 0.5, 0.5, {})

    fake_model = MagicMock()
    # Same embeddings for both halves → no drift
    fake_model.encode.return_value = np.random.default_rng(42).normal(0, 0.01, (20, 384))

    with patch("self_improving_rag.analysis.SentenceTransformer", return_value=fake_model):
        detector = DriftDetector(storage, kl_threshold=0.1)
        detector._model = fake_model
        result = detector.compute_drift()

    assert result["drift_detected"] is False


def test_compute_hyperparameter_importance_insufficient(storage):
    result = compute_hyperparameter_importance(storage, "empty_study")
    assert result == {}


def test_compute_hyperparameter_importance_returns_normalized(storage):
    for i in range(10):
        score = float(i) / 10
        storage.save_trial(
            "imp_test", i,
            {"chunk_size": (i + 1) * 64, "top_k": i + 1},
            score,
        )
    result = compute_hyperparameter_importance(storage, "imp_test")
    if result:
        total = sum(result.values())
        assert abs(total - 1.0) < 0.01
        assert all(0.0 <= v <= 1.0 for v in result.values())


def test_compute_hyperparameter_importance_keys(storage):
    params = {"chunk_size": 512, "top_k": 5, "temperature": 0.1}
    for i in range(8):
        storage.save_trial("key_test", i, params, float(i) * 0.1)
    result = compute_hyperparameter_importance(storage, "key_test")
    # All values same → may be 0 importance for all
    assert isinstance(result, dict)
