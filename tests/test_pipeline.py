import types
from unittest.mock import MagicMock, patch
import pytest
from self_improving_rag.config import RAGConfig
from self_improving_rag.pipeline import RAGPipeline, _COLLECTION_CACHE


def _mock_embedder():
    import numpy as np
    m = MagicMock()
    m.encode.return_value = np.array([[0.1] * 384])
    return m


def _mock_collection(results=None):
    col = MagicMock()
    if results is None:
        results = {
            "documents": [["chunk text"]],
            "metadatas": [[{"source": "test.txt", "chunk_index": 0}]],
            "distances": [[0.1]],
        }
    col.query.return_value = results
    return col


def _make_pipeline():
    cfg = RAGConfig()
    p = RAGPipeline(cfg, chroma_dir="/tmp/test_chroma")
    p._embedder = _mock_embedder()
    p._collection = _mock_collection()
    return p


def test_retrieve_returns_chunks():
    p = _make_pipeline()
    chunks = p.retrieve("test query")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "chunk text"
    assert "score" in chunks[0]


def test_retrieve_filters_below_threshold():
    col = _mock_collection(results={
        "documents": [["low similarity chunk"]],
        "metadatas": [[{"source": "x.txt", "chunk_index": 0}]],
        "distances": [[0.9]],  # score = 0.1, below default threshold 0.3
    })
    cfg = RAGConfig(similarity_threshold=0.3)
    p = RAGPipeline(cfg, chroma_dir="/tmp/test_chroma")
    p._embedder = _mock_embedder()
    p._collection = col
    chunks = p.retrieve("query")
    assert chunks == []


def test_retrieve_empty_results():
    col = _mock_collection(results={"documents": [[]], "metadatas": [[]], "distances": [[]]})
    p = RAGPipeline(RAGConfig(), chroma_dir="/tmp/test_chroma")
    p._embedder = _mock_embedder()
    p._collection = col
    assert p.retrieve("anything") == []


def test_generate_calls_groq():
    p = _make_pipeline()
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=" test answer "))
    ]
    p._groq_client = mock_groq
    answer = p.generate("What is RAG?", [{"text": "RAG is a technique."}])
    assert answer == "test answer"
    mock_groq.chat.completions.create.assert_called_once()


def test_query_returns_all_fields():
    p = _make_pipeline()
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="an answer"))
    ]
    p._groq_client = mock_groq
    result = p.query("test question")
    assert "question" in result
    assert "answer" in result
    assert "context" in result
    assert "chunks" in result
    assert "config" in result


def test_index_documents_returns_count():
    cfg = RAGConfig()
    p = RAGPipeline(cfg, chroma_dir="/tmp/test_chroma")
    p._embedder = _mock_embedder()
    p._collection = MagicMock()
    docs = [{"id": f"doc_{i}", "text": f"text {i}", "source": "x.txt", "chunk_index": i} for i in range(5)]
    n = p.index_documents(docs)
    assert n == 5


def test_index_documents_empty():
    p = RAGPipeline(RAGConfig(), chroma_dir="/tmp/test_chroma")
    assert p.index_documents([]) == 0
