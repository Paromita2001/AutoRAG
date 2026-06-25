import json
import os
from unittest.mock import MagicMock, patch
import pytest
from self_improving_rag.config import RAGConfig
from self_improving_rag.scheduler import (
    load_current_config,
    nightly_job,
    save_current_config,
)


@pytest.fixture
def storage(tmp_path):
    from self_improving_rag.storage import RAGStorage
    return RAGStorage(db_path=str(tmp_path / "sched.db"))


def _fill_study(storage, study_name, scores):
    for i, s in enumerate(scores):
        storage.save_query_result(
            f"q{i}", "a", "c", s, s, s, {}, study_name=study_name
        )


def test_save_and_load_config(tmp_path):
    cfg = RAGConfig(chunk_size=256, top_k=3)
    path = str(tmp_path / "current_config.json")
    # Temporarily override path
    import self_improving_rag.scheduler as sched_mod
    original = sched_mod._CURRENT_CONFIG_PATH
    sched_mod._CURRENT_CONFIG_PATH = path
    try:
        save_current_config(cfg)
        loaded = load_current_config()
        assert loaded.chunk_size == 256
        assert loaded.top_k == 3
    finally:
        sched_mod._CURRENT_CONFIG_PATH = original


def test_load_config_returns_default_when_missing(tmp_path):
    import self_improving_rag.scheduler as sched_mod
    original = sched_mod._CURRENT_CONFIG_PATH
    sched_mod._CURRENT_CONFIG_PATH = str(tmp_path / "no_config.json")
    try:
        cfg = load_current_config()
        assert isinstance(cfg, RAGConfig)
    finally:
        sched_mod._CURRENT_CONFIG_PATH = original


def test_nightly_job_no_completed_trials(tmp_path):
    """When optimizer finds nothing, job returns deployed=False."""
    db = str(tmp_path / "j.db")
    with patch("self_improving_rag.scheduler.RAGOptimizer") as MockOpt:
        mock_opt = MagicMock()
        mock_opt.run.return_value = {"best_config": None, "best_score": None, "n_trials": 0}
        MockOpt.return_value = mock_opt
        result = nightly_job(questions=["q1"], db_path=db, chroma_dir="/tmp/c")
    assert result["deployed"] is False
    assert result["ab_result"] is None


def test_nightly_job_deploys_when_ab_passes(tmp_path):
    db = str(tmp_path / "j2.db")
    from self_improving_rag.storage import RAGStorage
    storage = RAGStorage(db_path=db)
    # Seed control study
    _fill_study(storage, "nightly", [0.5] * 10)
    _fill_study(storage, "nightly_treatment", [0.9] * 10)

    best_cfg = RAGConfig(chunk_size=256)
    with patch("self_improving_rag.scheduler.RAGOptimizer") as MockOpt, \
         patch("self_improving_rag.scheduler.save_current_config") as mock_save:
        mock_opt = MagicMock()
        mock_opt.run.return_value = {"best_config": best_cfg, "best_score": 0.9, "n_trials": 5}
        MockOpt.return_value = mock_opt
        result = nightly_job(questions=["q1"], db_path=db, chroma_dir="/tmp/c")

    assert result["deployed"] is True
    mock_save.assert_called_once_with(best_cfg, username=None)


def test_nightly_job_does_not_deploy_when_ab_fails(tmp_path):
    db = str(tmp_path / "j3.db")
    from self_improving_rag.storage import RAGStorage
    storage = RAGStorage(db_path=db)
    # Control same as treatment → no significant difference
    _fill_study(storage, "nightly", [0.5] * 5)
    _fill_study(storage, "nightly_treatment", [0.5] * 5)

    best_cfg = RAGConfig(chunk_size=512)
    with patch("self_improving_rag.scheduler.RAGOptimizer") as MockOpt, \
         patch("self_improving_rag.scheduler.save_current_config") as mock_save:
        mock_opt = MagicMock()
        mock_opt.run.return_value = {"best_config": best_cfg, "best_score": 0.5, "n_trials": 5}
        MockOpt.return_value = mock_opt
        result = nightly_job(questions=["q1"], db_path=db, chroma_dir="/tmp/c")

    assert result["deployed"] is False
    mock_save.assert_not_called()


def test_nightly_job_computes_drift_when_no_config(tmp_path):
    db = str(tmp_path / "j4.db")
    drift_detector = MagicMock()
    drift_detector.compute_drift.return_value = {"drift_detected": False, "kl_divergence": 0.0}
    with patch("self_improving_rag.scheduler.RAGOptimizer") as MockOpt:
        mock_opt = MagicMock()
        mock_opt.run.return_value = {"best_config": None, "best_score": None, "n_trials": 0}
        MockOpt.return_value = mock_opt
        result = nightly_job(
            questions=["q1"], db_path=db, chroma_dir="/tmp/c",
            drift_detector=drift_detector,
        )
    drift_detector.compute_drift.assert_called_once()
    assert result["drift"] is not None


def test_nightly_job_includes_drift_result(tmp_path):
    db = str(tmp_path / "j5.db")
    from self_improving_rag.storage import RAGStorage
    storage = RAGStorage(db_path=db)
    _fill_study(storage, "nightly", [0.5] * 5)
    _fill_study(storage, "nightly_treatment", [0.9] * 10)

    best_cfg = RAGConfig()
    drift_detector = MagicMock()
    drift_detector.compute_drift.return_value = {"drift_detected": True, "kl_divergence": 0.3}
    with patch("self_improving_rag.scheduler.RAGOptimizer") as MockOpt, \
         patch("self_improving_rag.scheduler.save_current_config"):
        mock_opt = MagicMock()
        mock_opt.run.return_value = {"best_config": best_cfg, "best_score": 0.9, "n_trials": 5}
        MockOpt.return_value = mock_opt
        result = nightly_job(
            questions=["q1"], db_path=db, chroma_dir="/tmp/c",
            drift_detector=drift_detector,
        )
    assert result["drift"]["drift_detected"] is True
