"""
AutoRAG — Self-Improving RAG System
Usage:
  python main.py                          # start scheduler + dashboard
  python main.py --documents ./docs       # index docs, then start
  python main.py --optimize-now           # run one optimization cycle immediately
  python main.py --no-scheduler           # dashboard only
  python main.py --no-dashboard           # scheduler only (headless)
"""
import argparse
import json
import logging
import os
import subprocess
import sys

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _load_questions(path: str = "data/test_questions.json") -> list:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return [q["question"] for q in data if q.get("category") != "out_of_scope"]
            return data
        if isinstance(data, dict) and "questions" in data:
            return [
                q["question"] for q in data["questions"]
                if q.get("category") != "out_of_scope"
            ]
    return [
        "What is retrieval-augmented generation?",
        "How does embedding-based search work?",
        "What is the difference between BM25 and dense retrieval?",
    ]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="AutoRAG Self-Improving RAG System")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable nightly scheduler")
    parser.add_argument("--optimize-now", action="store_true", help="Run optimization immediately")
    parser.add_argument("--documents", metavar="PATH", help="Path to documents to index")
    parser.add_argument("--study-name", default="nightly", help="Optuna study name")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--db-path", default="./rag_results.db", help="SQLite database path")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="ChromaDB directory")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable Streamlit dashboard")
    parser.add_argument("--questions", default="data/test_questions.json", help="Path to questions JSON")
    args = parser.parse_args()

    from self_improving_rag.config import RAGConfig
    from self_improving_rag.pipeline import RAGPipeline
    from self_improving_rag.scheduler import (
        load_current_config,
        save_current_config,
        start_scheduler,
        stop_scheduler,
    )
    from self_improving_rag.storage import RAGStorage

    questions = _load_questions(args.questions)
    logger.info("Loaded %d questions", len(questions))

    # Index documents if path provided
    if args.documents:
        from self_improving_rag.file_processor import load_documents

        config = load_current_config()
        pipeline = RAGPipeline(config, chroma_dir=args.chroma_dir)
        docs = load_documents(args.documents, chunk_size=config.chunk_size, overlap=config.overlap)
        n = pipeline.index_documents(docs)
        logger.info("Indexed %d chunks from %s", n, args.documents)

    # Optional immediate optimization
    if args.optimize_now:
        from self_improving_rag.optimizer import RAGOptimizer

        storage = RAGStorage(db_path=args.db_path)
        opt = RAGOptimizer(
            questions=questions,
            storage=storage,
            chroma_dir=args.chroma_dir,
            study_name=args.study_name,
            n_trials=args.n_trials,
        )
        result = opt.run()
        best = result.get("best_config")
        if best:
            save_current_config(best)
            logger.info("Optimized — best score: %.4f  config saved.", result.get("best_score"))
        else:
            logger.warning("Optimization produced no completed trials.")

    # Start background scheduler
    scheduler = None
    if not args.no_scheduler:
        scheduler = start_scheduler(
            questions=questions,
            chroma_dir=args.chroma_dir,
            db_path=args.db_path,
            study_name=args.study_name,
            n_trials=args.n_trials,
        )

    # Launch Streamlit dashboard
    if not args.no_dashboard:
        entry = os.path.join(os.path.dirname(__file__), "self_improving_rag", "dashboard_entry.py")
        dashboard_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", entry, "--server.headless=true"],
        )
        logger.info("Dashboard started at http://localhost:8501")
        try:
            dashboard_proc.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down…")
            dashboard_proc.terminate()
        finally:
            if scheduler:
                stop_scheduler()
    else:
        logger.info("Running headless (no dashboard). Press Ctrl-C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Shutting down…")
            if scheduler:
                stop_scheduler()


if __name__ == "__main__":
    main()
