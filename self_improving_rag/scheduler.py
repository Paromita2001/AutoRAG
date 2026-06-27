import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .ab_test import ABTest
from .analysis import DriftDetector
from .config import RAGConfig
from .optimizer import RAGOptimizer
from .question_store import (best_collection, get_questions_for_optimization,
                             question_count)
from .storage import RAGStorage

logger = logging.getLogger(__name__)

_CURRENT_CONFIG_PATH = "current_config.json"   # fallback / scheduler default
_SCHEDULER: Optional[BackgroundScheduler] = None
# Global lock: prevents scheduler job and "Optimize Now" button from running at the same time.
OPTIMIZER_LOCK = threading.Lock()


def _config_path(username: Optional[str] = None) -> str:
    if username:
        os.makedirs("data", exist_ok=True)
        safe = "".join(c for c in username if c.isalnum() or c == "_")[:32]
        return f"data/config_{safe}.json"
    return _CURRENT_CONFIG_PATH


def load_current_config(username: Optional[str] = None) -> RAGConfig:
    path = _config_path(username)
    if os.path.exists(path):
        with open(path) as f:
            return RAGConfig.from_dict(json.load(f))
    return RAGConfig()


def save_current_config(config: RAGConfig, username: Optional[str] = None) -> None:
    path = _config_path(username)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def nightly_job(
    questions: List[str],
    chroma_dir: str = "./chroma_db",
    db_path: str = "./rag_results.db",
    study_name: str = "nightly",
    n_trials: int = 3,
    drift_detector: Optional[DriftDetector] = None,
    username: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info("[scheduler] Starting optimization job (user=%s)", username or "global")

    # Use this user's real questions if available, else fall back to provided list
    user_qs = get_questions_for_optimization(username, max_n=5) if username else []
    if len(user_qs) >= 3:
        questions = [r["question"] for r in user_qs]
        col_name  = best_collection(username)
        logger.info("[scheduler] Using %d questions for %s (collection: %s)",
                    len(questions), username, col_name)
    else:
        col_name = None
        logger.info("[scheduler] Fewer than 3 questions for %s — using defaults", username)

    # Namespace study by username so each user's trials are separate in SQLite
    effective_study = f"{study_name}_{username}" if username else study_name

    storage = RAGStorage(db_path=db_path)
    optimizer = RAGOptimizer(
        questions=questions,
        storage=storage,
        chroma_dir=chroma_dir,
        study_name=f"{effective_study}_treatment",
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
        logger.warning("[scheduler] No completed trials — skipping A/B test")
        return {**opt_result, "deployed": False, "ab_result": None, "drift": drift_result}

    # Check if there is any control data to compare against
    control_rows = storage.get_query_results(study_name=effective_study)
    deployed = False

    if len(control_rows) < 2:
        # No previous run to compare — deploy the new config directly (first-time or cold start)
        save_current_config(new_config, username=username)
        logger.info(
            "[scheduler] No control data — deploying new config directly for %s: %s",
            username, new_config.to_dict(),
        )
        deployed = True
        ab_result = {"should_deploy": True, "direction": "first_run", "p_value": None, "cohen_d": None}
    else:
        ab = ABTest(
            storage=storage,
            control_study=effective_study,
            treatment_study=f"{effective_study}_treatment",
        )
        ab_result = ab.run()
        if ab_result["should_deploy"]:
            save_current_config(new_config, username=username)
            logger.info("[scheduler] Deployed config for %s: %s", username, new_config.to_dict())
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
    """Runs every hour. Only optimizes when enough NEW questions have been collected.
    Scheduler-level job has no specific user — optimizes for whoever has the most questions."""
    global _LAST_OPTIMIZED_COUNT

    # Find the user with the most new questions across all question banks
    import glob
    user_files = glob.glob("data/questions_*.json")
    best_user  = None
    best_count = 0
    for f in user_files:
        name = os.path.basename(f).replace("questions_", "").replace(".json", "")
        c = question_count(name)
        if c > best_count:
            best_count, best_user = c, name

    current = best_count
    new_since_last = current - _LAST_OPTIMIZED_COUNT

    if current < 3:
        logger.info("[scheduler] Only %d questions total — need at least 3", current)
        return
    if new_since_last < _AUTO_OPTIMIZE_THRESHOLD and _LAST_OPTIMIZED_COUNT > 0:
        logger.info("[scheduler] %d new questions since last run — waiting for %d",
                    new_since_last, _AUTO_OPTIMIZE_THRESHOLD)
        return

    logger.info("[scheduler] %d new questions for %s — starting optimization",
                new_since_last, best_user)
    if not OPTIMIZER_LOCK.acquire(blocking=False):
        logger.info("[scheduler] Optimizer already running — skipping this tick")
        return
    try:
        result = nightly_job(questions=questions, chroma_dir=chroma_dir,
                             db_path=db_path, study_name=study_name, n_trials=n_trials,
                             username=best_user)
        if result.get("deployed"):
            _LAST_OPTIMIZED_COUNT = current
            logger.info("[scheduler] New config deployed for %s", best_user)
    finally:
        OPTIMIZER_LOCK.release()


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
