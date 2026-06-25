import pytest
from self_improving_rag.config import RAGConfig


def test_default_values():
    cfg = RAGConfig()
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.chunk_size == 512
    assert cfg.overlap == 64
    assert cfg.top_k == 5
    assert cfg.temperature == 0.1
    assert cfg.groq_model == "llama-3.3-70b-versatile"
    assert cfg.rerank is False
    assert cfg.similarity_threshold == 0.3


def test_collection_key_is_12_chars():
    cfg = RAGConfig()
    key = cfg.collection_key()
    assert len(key) == 12


def test_collection_key_depends_on_model_chunk_overlap():
    a = RAGConfig(embedding_model="all-MiniLM-L6-v2", chunk_size=512, overlap=64)
    b = RAGConfig(embedding_model="BAAI/bge-large-en-v1.5", chunk_size=512, overlap=64)
    assert a.collection_key() != b.collection_key()


def test_collection_key_ignores_top_k_and_temperature():
    a = RAGConfig(top_k=3, temperature=0.0)
    b = RAGConfig(top_k=10, temperature=0.5)
    assert a.collection_key() == b.collection_key()


def test_to_dict_round_trip():
    cfg = RAGConfig(chunk_size=256, overlap=32, top_k=7)
    d = cfg.to_dict()
    restored = RAGConfig.from_dict(d)
    assert restored.chunk_size == 256
    assert restored.overlap == 32
    assert restored.top_k == 7


def test_from_dict_ignores_unknown_keys():
    d = {"chunk_size": 128, "unknown_key": "boom"}
    cfg = RAGConfig.from_dict(d)
    assert cfg.chunk_size == 128


def test_collection_key_stable():
    cfg = RAGConfig()
    assert cfg.collection_key() == cfg.collection_key()


def test_to_dict_has_all_8_fields():
    d = RAGConfig().to_dict()
    expected = {
        "embedding_model", "chunk_size", "overlap", "top_k",
        "temperature", "groq_model", "rerank", "similarity_threshold",
    }
    assert set(d.keys()) == expected


def test_custom_groq_model():
    cfg = RAGConfig(groq_model="llama3-8b-8192")
    assert cfg.groq_model == "llama3-8b-8192"
