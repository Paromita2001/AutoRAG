"""Streamlit entry point — import and run dashboard from package."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from self_improving_rag.dashboard import run_dashboard

db_path = os.environ.get("RAG_DB_PATH", "./rag_results.db")
chroma_dir = os.environ.get("RAG_CHROMA_DIR", "./chroma_db")

run_dashboard(db_path=db_path, chroma_dir=chroma_dir)
