import logging
import math
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from .storage import RAGStorage

logger = logging.getLogger(__name__)


def _kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    """KL divergence D(p||q) via histogram binning with Laplace smoothing."""
    lo = min(float(p.min()), float(q.min()))
    hi = max(float(p.max()), float(q.max()))
    if lo == hi:
        return 0.0
    p_hist, _ = np.histogram(p, bins=bins, range=(lo, hi), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(lo, hi), density=True)
    eps = 1e-10
    p_hist = (p_hist + eps) / (p_hist + eps).sum()
    q_hist = (q_hist + eps) / (q_hist + eps).sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


class DriftDetector:
    """
    Detects query-distribution drift by comparing embedding distributions
    between the older half and newer half of stored queries.
    Uses KL divergence per dimension (first N dims), averaged.
    """

    def __init__(
        self,
        storage: RAGStorage,
        embedding_model: str = "all-MiniLM-L6-v2",
        kl_threshold: float = 0.1,
        window_size: int = 100,
    ):
        self.storage = storage
        self.kl_threshold = kl_threshold
        self.window_size = window_size
        self._model = SentenceTransformer(embedding_model)

    def compute_drift(self) -> Dict[str, Any]:
        results = self.storage.get_query_results(limit=self.window_size * 2)
        questions = [r["question"] for r in results if r.get("question")]
        if len(questions) < 20:
            return {"drift_detected": False, "kl_divergence": 0.0, "reason": "insufficient_data"}
        mid = len(questions) // 2
        old_emb = self._model.encode(questions[mid:], show_progress_bar=False)
        new_emb = self._model.encode(questions[:mid], show_progress_bar=False)
        n_dims = min(old_emb.shape[1], 50)
        kl_values = [_kl_divergence(old_emb[:, d], new_emb[:, d]) for d in range(n_dims)]
        mean_kl = sum(kl_values) / len(kl_values)
        return {
            "drift_detected": mean_kl > self.kl_threshold,
            "kl_divergence": mean_kl,
            "threshold": self.kl_threshold,
            "dims_checked": n_dims,
        }


def compute_hyperparameter_importance(
    storage: RAGStorage,
    study_name: str,
) -> Dict[str, float]:
    """
    Approximate fANOVA importance via Pearson |correlation| between
    each hyperparameter value and trial composite scores.
    Returns a dict normalised to sum=1.
    """
    trials = storage.get_trials(study_name)
    completed = [t for t in trials if t.get("completed") and t.get("score") is not None]
    if len(completed) < 5:
        return {}
    scores = [t["score"] for t in completed]
    param_keys = list(completed[0].get("params", {}).keys())
    importance: Dict[str, float] = {}
    for key in param_keys:
        values = [t["params"].get(key) for t in completed]
        if isinstance(values[0], str):
            unique = sorted(set(values))
            values = [float(unique.index(v)) for v in values]
        try:
            vals = [float(v) for v in values]
            n = len(vals)
            mv = sum(vals) / n
            ms = sum(scores) / n
            cov = sum((v - mv) * (s - ms) for v, s in zip(vals, scores)) / n
            sv = math.sqrt(sum((v - mv) ** 2 for v in vals) / n)
            ss = math.sqrt(sum((s - ms) ** 2 for s in scores) / n)
            importance[key] = abs(cov / (sv * ss)) if sv > 0 and ss > 0 else 0.0
        except Exception:
            importance[key] = 0.0
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}
    return importance
