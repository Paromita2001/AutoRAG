import pytest
from self_improving_rag.ab_test import ABTest, _cohen_d


@pytest.fixture
def storage(tmp_path):
    from self_improving_rag.storage import RAGStorage
    return RAGStorage(db_path=str(tmp_path / "ab.db"))


def _fill_study(storage, study_name, scores):
    for i, s in enumerate(scores):
        storage.save_query_result(
            question=f"q{i}", answer="a", context="c",
            faithfulness=s, relevance=s, composite=s,
            config={}, study_name=study_name,
        )


def test_cohen_d_basic():
    a = [0.5, 0.5, 0.5]
    b = [0.8, 0.8, 0.8]
    d = _cohen_d(a, b)
    assert d > 0  # b > a


def test_cohen_d_identical():
    a = [0.5, 0.5, 0.5]
    assert _cohen_d(a, a) == 0.0


def test_cohen_d_insufficient():
    assert _cohen_d([0.5], [0.8]) == 0.0


def test_run_ab_test_should_deploy(storage):
    _fill_study(storage, "control", [0.5] * 10)
    _fill_study(storage, "treatment", [0.9] * 10)
    ab = ABTest(storage, control_study="control", treatment_study="treatment")
    result = ab.run()
    assert result["should_deploy"] is True
    assert result["direction"] == "improvement"
    assert result["p_value"] < 0.05
    assert abs(result["cohen_d"]) > 0.2


def test_run_ab_test_should_not_deploy_regression(storage):
    _fill_study(storage, "control2", [0.9] * 10)
    _fill_study(storage, "treatment2", [0.5] * 10)
    ab = ABTest(storage, control_study="control2", treatment_study="treatment2")
    result = ab.run()
    assert result["should_deploy"] is False
    assert result["direction"] == "regression"


def test_run_ab_test_insufficient_data(storage):
    ab = ABTest(storage, control_study="empty_c", treatment_study="empty_t")
    result = ab.run()
    assert result["should_deploy"] is False
    assert result["direction"] == "insufficient_data"


def test_run_ab_test_custom_thresholds(storage):
    # Use real within-group variance so pooled_std is non-trivial
    ctrl_scores = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.6, 0.4, 0.55, 0.45]
    trt_scores  = [0.6, 0.7, 0.5, 0.65, 0.55, 0.6, 0.7, 0.5, 0.65, 0.55]
    _fill_study(storage, "ctrl3", ctrl_scores)
    _fill_study(storage, "trt3", trt_scores)
    ab = ABTest(
        storage,
        control_study="ctrl3",
        treatment_study="trt3",
        p_threshold=0.05,
        d_threshold=100.0,  # Cohen's d will be ~2, far below 100 → should not deploy
    )
    result = ab.run()
    assert result["should_deploy"] is False


def test_result_contains_means(storage):
    _fill_study(storage, "c5", [0.4] * 5)
    _fill_study(storage, "t5", [0.8] * 5)
    ab = ABTest(storage, control_study="c5", treatment_study="t5")
    result = ab.run()
    assert result["control_mean"] == pytest.approx(0.4)
    assert result["treatment_mean"] == pytest.approx(0.8)
