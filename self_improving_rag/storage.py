import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class QueryResult(Base):
    __tablename__ = "query_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    context_snippet = Column(String)
    faithfulness = Column(Float)
    relevance = Column(Float)
    composite = Column(Float)
    config_json = Column(JSON)
    study_name = Column(String)


class OptimizationTrial(Base):
    __tablename__ = "optimization_trials"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    study_name = Column(String, nullable=False)
    trial_number = Column(Integer, nullable=False)
    params = Column(JSON)
    score = Column(Float)
    completed = Column(Boolean, default=False)


class RAGStorage:
    def __init__(self, db_path: str = "./rag_results.db"):
        url = f"sqlite:///{db_path}"
        self._engine = create_engine(url, connect_args={"check_same_thread": False})
        with self._engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            conn.commit()
        Base.metadata.create_all(self._engine)

    def save_query_result(
        self,
        question: str,
        answer: str,
        context: str,
        faithfulness: float,
        relevance: float,
        composite: float,
        config: Dict[str, Any],
        study_name: Optional[str] = None,
    ) -> int:
        with Session(self._engine) as session:
            row = QueryResult(
                question=question,
                answer=answer,
                context_snippet=(context or "")[:500],
                faithfulness=faithfulness,
                relevance=relevance,
                composite=composite,
                config_json=config,
                study_name=study_name,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def save_trial(
        self,
        study_name: str,
        trial_number: int,
        params: Dict[str, Any],
        score: float,
        completed: bool = True,
    ) -> int:
        with Session(self._engine) as session:
            row = OptimizationTrial(
                study_name=study_name,
                trial_number=trial_number,
                params=params,
                score=score,
                completed=completed,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def get_query_results(
        self,
        study_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with Session(self._engine) as session:
            q = session.query(QueryResult)
            if study_name:
                q = q.filter(QueryResult.study_name == study_name)
            rows = q.order_by(QueryResult.timestamp.desc()).limit(limit).all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "question": r.question,
                    "answer": r.answer,
                    "context_snippet": r.context_snippet,
                    "faithfulness": r.faithfulness,
                    "relevance": r.relevance,
                    "composite": r.composite,
                    "config_json": r.config_json,
                    "study_name": r.study_name,
                }
                for r in rows
            ]

    def get_best_config(self, study_name: str) -> Optional[Dict[str, Any]]:
        with Session(self._engine) as session:
            row = (
                session.query(QueryResult)
                .filter(QueryResult.study_name == study_name)
                .filter(QueryResult.composite.isnot(None))
                .order_by(QueryResult.composite.desc())
                .first()
            )
            return row.config_json if row else None

    def get_trials(self, study_name: str) -> List[Dict[str, Any]]:
        with Session(self._engine) as session:
            rows = (
                session.query(OptimizationTrial)
                .filter(OptimizationTrial.study_name == study_name)
                .order_by(OptimizationTrial.trial_number)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "trial_number": r.trial_number,
                    "params": r.params,
                    "score": r.score,
                    "completed": r.completed,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in rows
            ]
