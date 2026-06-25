import logging
import math
from typing import Dict, List

from scipy import stats

from .storage import RAGStorage

logger = logging.getLogger(__name__)


def _cohen_d(group_a: List[float], group_b: List[float]) -> float:
    """Pooled Cohen'"'"'s d effect size (positive = b > a)."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a = sum(group_a) / n_a
    mean_b = sum(group_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (n_b - 1)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (mean_b - mean_a) / pooled_std


class ABTest:
    """
    Compare treatment vs. control composite scores using:
      - Welch'"'"'s t-test (p < p_threshold)
      - Cohen'"'"'s d    (|d| > d_threshold)
      - Direction check (treatment mean > control mean)
    All three must pass for should_deploy=True.
    """

    def __init__(
        self,
        storage: RAGStorage,
        control_study: str,
        treatment_study: str,
        p_threshold: float = 0.05,
        d_threshold: float = 0.2,
    ):
        self.storage = storage
        self.control_study = control_study
        self.treatment_study = treatment_study
        self.p_threshold = p_threshold
        self.d_threshold = d_threshold

    def run(self) -> Dict:
        control_rows = self.storage.get_query_results(study_name=self.control_study)
        treatment_rows = self.storage.get_query_results(study_name=self.treatment_study)
        control_scores = [r["composite"] for r in control_rows if r["composite"] is not None]
        treatment_scores = [r["composite"] for r in treatment_rows if r["composite"] is not None]
        if len(control_scores) < 2 or len(treatment_scores) < 2:
            logger.warning("Insufficient data for A/B test")
            return {
                "should_deploy": False,
                "p_value": 1.0,
                "cohen_d": 0.0,
                "direction": "insufficient_data",
                "control_mean": None,
                "treatment_mean": None,
            }
        _, p_value = stats.ttest_ind(control_scores, treatment_scores, equal_var=False)
        d = _cohen_d(control_scores, treatment_scores)
        control_mean = sum(control_scores) / len(control_scores)
        treatment_mean = sum(treatment_scores) / len(treatment_scores)
        direction = "improvement" if treatment_mean > control_mean else "regression"
        should_deploy = (
            p_value < self.p_threshold
            and abs(d) > self.d_threshold
            and direction == "improvement"
        )
        return {
            "should_deploy": should_deploy,
            "p_value": float(p_value),
            "cohen_d": float(d),
            "direction": direction,
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
        }
