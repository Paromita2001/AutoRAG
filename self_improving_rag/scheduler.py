import json
import logging
import os
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .ab_test import ABTest
from .analysis import DriftDetector
from .config import RAGConfig
from .optimizer import RAGOptimizer
from .question_store import best_collection, get_questions_for_optimization, question_count
from .storage import RAGStorage

logger = logging.getLogger(__name__)

_CURRENT_CONFIG_PATH = "current_config.json"
_SCHEDULER: Optional[BackgroundScheduler] = None


def load_current_config() -> RAGConfig:
    if os.path.exists(_CURRENT_CONFIG_PATH):
        with open(_CURRENT_CONFIG_PATH) as f:
            return RAGConfig.from_dict(json.load(f))
    return RAGConfig()


def save_current_config(config: RAGConfig) -> None:
    with open(_CURRENT_CONFIG_PATH, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def nightly_job(
    questions: List[str],
    chroma_dir: str = "./chroma_db",
    db_path: str = "./rag_results.db",
    study_name: str = "nightly",
    n_trials: int = 20,
    drift_detector: Optional[DriftDetector] = None,
) -> Dict[str, Any]:
    logger.info("[scheduler] Starting nightly optimization job")

    # Prefer real user questions over hardcoded ones
    user_qs = get_questions_for_optimization(max_n=20)
    if len(user_qs) >= 3:
        questions = [r["question"] for r in user_qs]
        col_name  = best_collection()
        logger.info("[scheduler] Using %d real user questions (collection: %s)",
                    len(questions), col_name)
    else:
        col_name = None
        logger.info("[scheduler] Fewer than 3 user questions recorded — using default questions")

    storage = RAGStorage(db_path=db_path)
    optimizer = RAGOptimizer(
        questions=questions,
        storage=storage,
        chroma_dir=chroma_dir,
        study_name=f"{study_name}_treatment",
        n_trials=n_trials,
        collection_name=col_name,
    )
    opt_result = optimizer.run()
    new_config = opt_result.get("best_config")

    drift_result = None
    if drift_detector is not None:
        drift_result = drift_detector.compute_drift()
        if drift_result.get("drift_detected"):
            logger.warning("[scheduler] Drift detected: %s", drift_result)

    if new_config is None:
        logger.warning("[scheduler] No completed trials -- skipping A/B test")
        return {**opt_result, "deployed": False, "ab_result": None, "drift": drift_result}

    ab = ABTest(
        storage=storage,
        control_study=study_name,
        treatment_study=f"{study_name}_treatment",
    )
    ab_result = ab.run()
    deployed = False
    if ab_result["should_deploy"]:
        save_current_config(new_config)
        logger.info("[scheduler] Deployed new config: %s", new_config.to_dict())
        deployed = True

    return {**opt_result, "deployed": deployed, "ab_result": ab_result, "drift": drift_result}


_AUTO_OPTIMIZE_THRESHOLD = 10  # questions needed to trigger auto-optimization
_LAST_OPTIMIZED_COUNT    = 0   # track how many questions existed at last run


def _smart_job(
    questions: List[str],
    chroma_dir: str,
    db_path: str,
    study_name: str,
    n_trials: int,
) -> None:
    """Runs every hour. Only optimizes when enough NEW questions have been collected."""
    global _LAST_OPTIMIZED_COUNT
    current = question_count()
    new_since_last = current - _LAST_OPTIMIZED_COUNT

    if current < 3:
        logger.info("[scheduler] Only %d questions — need at least 3 to optimize", current)
        return
    if new_since_last < _AUTO_OPTIMIZE_THRESHOLD and _LAST_OPTIMIZED_COUNT > 0:
        logger.info("[scheduler] %d new questions since last run — waiting for %d",
                    new_since_last, _AUTO_OPTIMIZE_THRESHOLD)
        return

    logger.info("[scheduler] %d new questions collected — starting optimization", new_since_last)
    result = nightly_job(questions=questions, chroma_dir=chroma_dir,
                         db_path=db_path, study_name=study_name, n_trials=n_trials)
    if result.get("deployed"):
        _LAST_OPTIMIZED_COUNT = current
        logger.info("[scheduler] New config deployed after %d questions", current)


def start_scheduler(
    questions: List[str],
    chroma_dir: str = "./chroma_db",
    db_path: str = "./rag_results.db",
    study_name: str = "nightly",
    n_trials: int = 20,
) -> BackgroundScheduler:
    global _SCHEDULER
    if _SCHEDULER and _SCHEDULER.running:
        logger.info("[scheduler] Already running")
        return _SCHEDULER
    _SCHEDULER = BackgroundScheduler()

    # Smart trigger: check every hour, only optimize when 10+ new questions collected
    _SCHEDULER.add_job(
        _smart_job,
        trigger=IntervalTrigger(hours=1),
        kwargs={
            "questions": questions,
            "chroma_dir": chroma_dir,
            "db_path": db_path,
            "study_name": study_name,
            "n_trials": n_trials,
        },
        id="smart_optimization",
        replace_existing=True,
    )

    # Keep the 2 AM job as a fallback safety net (runs even if question count is low)
    _SCHEDULER.add_job(
        nightly_job,
        trigger=CronTrigger(hour=2, minute=0),
        kwargs={
            "questions": questions,
            "chroma_dir": chroma_dir,
            "db_path": db_path,
            "study_name": study_name,
            "n_trials": n_trials,
        },
        id="nightly_optimization",
        replace_existing=True,
    )

    _SCHEDULER.start()
    logger.info("[scheduler] Started — optimizes every hour when 10+ new questions, "
                "plus nightly fallback at 02:00")
    return _SCHEDULER


def stop_scheduler() -> None:
    global _SCHEDULER
    if _SCHEDULER and _SCHEDULER.running:
        _SCHEDULER.shutdown(wait=False)
        logger.info("[scheduler] Stopped")
