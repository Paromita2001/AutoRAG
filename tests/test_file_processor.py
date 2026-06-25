import os
import tempfile
import pytest
from self_improving_rag.file_processor import chunk_text, load_documents


def test_chunk_text_basic():
    text = "a" * 100
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 30 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("", 512, 64) == []
    assert chunk_text("   ", 512, 64) == []


def test_chunk_text_shorter_than_chunk():
    text = "hello world"
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunk_text_overlap():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=6, overlap=2)
    # First: abcdef, second starts at 4: efghij
    assert chunks[0].startswith("abcdef")
    assert len(chunks) >= 2


def test_load_documents_txt(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("This is a test document with enough text to form a chunk.", encoding="utf-8")
    docs = load_documents(str(tmp_path), chunk_size=100, overlap=10)
    assert len(docs) >= 1
    assert docs[0]["source"] == str(f)
    assert "chunk_index" in docs[0]
    assert "id" in docs[0]
    assert "text" in docs[0]


def test_load_documents_nonexistent(capsys):
    docs = load_documents("/nonexistent/path/xyz")
    assert docs == []


def test_load_documents_single_file(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Hello world. " * 20, encoding="utf-8")
    docs = load_documents(str(f), chunk_size=50, overlap=10)
    assert len(docs) >= 1


def test_load_documents_ids_unique(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("word " * 500, encoding="utf-8")
    docs = load_documents(str(f), chunk_size=100, overlap=20)
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids))


def test_load_documents_md_extension(tmp_path):
    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nSome markdown content here.", encoding="utf-8")
    docs = load_documents(str(tmp_path), chunk_size=200, overlap=10)
    assert any("readme" in d["id"] for d in docs)
