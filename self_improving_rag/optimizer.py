import logging
from typing import Any, Dict, List, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .config import RAGConfig
from .evaluator import RAGEvaluator
from .pipeline import RAGPipeline
from .storage import RAGStorage

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEARCH_SPACE: Dict[str, Any] = {
    "embedding_model": ["all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5"],
    "chunk_size": (256, 1024),
    "overlap": (32, 256),
    "top_k": (3, 10),
    "temperature": (0.0, 0.5),
    "similarity_threshold": (0.1, 0.7),
}


class RAGOptimizer:
    def __init__(
        self,
        questions: List[str],
        storage: RAGStorage,
        chroma_dir: str = "./chroma_db",
        study_name: str = "rag_optimization",
        n_trials: int = 30,
        groq_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.questions = questions
        self.storage = storage
        self.chroma_dir = chroma_dir
        self.study_name = study_name
        self.n_trials = n_trials
        self.collection_name = collection_name   # None → default collection key
        self._evaluator = RAGEvaluator(groq_api_key=groq_api_key)
        self._best_config: Optional[RAGConfig] = None

    def _suggest_config(self, trial: optuna.Trial) -> RAGConfig:
        return RAGConfig(
            embedding_model=trial.suggest_categorical(
                "embedding_model", SEARCH_SPACE["embedding_model"]
            ),
            chunk_size=trial.suggest_int("chunk_size", *SEARCH_SPACE["chunk_size"], step=64),
            overlap=trial.suggest_int("overlap", *SEARCH_SPACE["overlap"], step=32),
            top_k=trial.suggest_int("top_k", *SEARCH_SPACE["top_k"]),
            temperature=trial.suggest_float("temperature", *SEARCH_SPACE["temperature"]),
            similarity_threshold=trial.suggest_float(
                "similarity_threshold", *SEARCH_SPACE["similarity_threshold"]
            ),
        )

    def _objective(self, trial: optuna.Trial) -> float:
        config = self._suggest_config(trial)
        pipeline = RAGPipeline(config, chroma_dir=self.chroma_dir,
                               collection_name=self.collection_name)
        scores: List[float] = []
        for i, question in enumerate(self.questions):
            try:
                result = pipeline.query(question)
                eval_scores = self._evaluator.evaluate(
                    question=result["question"],
                    answer=result["answer"],
                    context=result["context"],
                )
                composite = eval_scores["composite"]
                scores.append(composite)
                self.storage.save_query_result(
                    question=result["question"],
                    answer=result["answer"],
                    context=result["context"],
                    faithfulness=eval_scores["faithfulness"],
                    relevance=eval_scores["relevance"],
                    composite=composite,
                    config=config.to_dict(),
                    study_name=self.study_name,
                )
                trial.report(sum(scores) / len(scores), step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.warning("Trial %d / question %d failed: %s", trial.number, i, exc)
                scores.append(0.0)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        self.storage.save_trial(
            study_name=self.study_name,
            trial_number=trial.number,
            params=config.to_dict(),
            score=mean_score,
        )
        return mean_score

    def run(self, storage_url: Optional[str] = None) -> Dict[str, Any]:
        study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            storage=storage_url,
            load_if_exists=True,
            sampler=TPESampler(seed=42, n_startup_trials=10),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            logger.warning("No completed trials")
            return {"best_config": None, "best_score": None, "n_trials": 0}
        best = study.best_trial
        best_config = RAGConfig.from_dict(best.params)
        self._best_config = best_config
        return {
            "best_config": best_config,
            "best_score": best.value,
            "n_trials": len(completed),
            "study": study,
        }

    @property
    def best_config(self) -> Optional[RAGConfig]:
        return self._best_config
