import logging
import time
from typing import Any, Dict, List, Optional

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

from .config import RAGConfig
from .groq_client import GroqKeyPool, get_pool

logger = logging.getLogger(__name__)

_COLLECTION_CACHE: Dict[str, Any] = {}


def _get_chroma_client(chroma_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=chroma_dir)


def _get_collection(client: chromadb.PersistentClient, name: str) -> Any:
    if name not in _COLLECTION_CACHE:
        _COLLECTION_CACHE[name] = client.get_or_create_collection(name)
    return _COLLECTION_CACHE[name]


class RAGPipeline:
    def __init__(self, config: RAGConfig, chroma_dir: str = "./chroma_db",
                 collection_name: Optional[str] = None,
                 expand_queries: bool = True):
        self.config = config
        self.chroma_dir = chroma_dir
        self._collection_name_override = collection_name  # per-chat isolation
        self._expand_queries_enabled = expand_queries
        self._embedder: Optional[SentenceTransformer] = None
        self._groq_client: Optional[Groq] = None  # kept for test mocking
        self._pool: Optional[GroqKeyPool] = None
        self._collection: Optional[Any] = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    @property
    def groq_client(self) -> Groq:
        """Kept for test compatibility — tests inject a mock here."""
        return self._groq_client

    def _get_pool(self) -> GroqKeyPool:
        if self._pool is None:
            self._pool = get_pool()
        return self._pool

    def get_collection(self) -> Any:
        if self._collection is None:
            client = _get_chroma_client(self.chroma_dir)
            col_name = self._collection_name_override or f"rag_{self.config.collection_key()}"
            self._collection = _get_collection(client, col_name)
        return self._collection

    def index_documents(self, documents: List[Dict[str, Any]]) -> int:
        if not documents:
            return 0
        collection = self.get_collection()
        texts = [d["text"] for d in documents]
        ids = [d["id"] for d in documents]
        metadatas = [
            {"source": d.get("source", ""), "chunk_index": d.get("chunk_index", 0)}
            for d in documents
        ]
        embeddings = self.embedder.encode(texts, batch_size=32, show_progress_bar=False).tolist()
        batch = 100
        for i in range(0, len(documents), batch):
            collection.upsert(
                ids=ids[i:i+batch],
                documents=texts[i:i+batch],
                embeddings=embeddings[i:i+batch],
                metadatas=metadatas[i:i+batch],
            )
        logger.info("Indexed %d documents", len(documents))
        return len(documents)

    def _expand_queries(self, query: str) -> List[str]:
        """Use Groq to generate 2 alternative phrasings (acronym expansion, rephrasing).
        Falls back to original query only if Groq fails."""
        prompt = (
            f"Given the search query: '{query}'\n"
            "Write 2 alternative search queries that mean the same thing. "
            "Expand any acronyms (e.g. LLM → Large Language Model), use synonyms, and rephrase. "
            "Return ONLY the 2 queries, one per line, no numbering, no explanations."
        )
        try:
            kwargs = dict(
                model=self.config.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            if self._groq_client is not None:
                resp = self._groq_client.chat.completions.create(**kwargs)
            else:
                resp = self._get_pool().chat_completions_create(**kwargs)
            lines = [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]
            return [query] + lines[:2]
        except Exception:
            return [query]

    def _retrieve_single(self, query: str) -> List[Dict[str, Any]]:
        collection = self.get_collection()
        q_emb = self.embedder.encode([query], show_progress_bar=False).tolist()[0]
        results = collection.query(query_embeddings=[q_emb], n_results=self.config.top_k)
        chunks = []
        if results.get("documents") and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = max(0.0, 1.0 - dist)
                if score >= self.config.similarity_threshold:
                    chunks.append({"text": doc, "metadata": meta, "score": score})
        return chunks

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Multi-query retrieval: expand query into alternatives, merge results, return top_k."""
        queries = self._expand_queries(query) if self._expand_queries_enabled else [query]
        seen: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            for chunk in self._retrieve_single(q):
                # deduplicate by first 120 chars of text; keep highest score
                key = chunk["text"][:120]
                if key not in seen or chunk["score"] > seen[key]["score"]:
                    seen[key] = chunk
        ranked = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
        return ranked[:self.config.top_k]

    _SYSTEM_PROMPT = (
        "You are a precise document assistant. "
        "Answer questions using ONLY the context provided. "
        "Rules:\n"
        "1. If the answer is clearly in the context: answer directly and completely.\n"
        "2. If the context partially answers: share what you found and note gaps. "
        "Start with: 'Based on your documents:'\n"
        "3. If the context has nothing relevant: say exactly: "
        "'I don't know based on the provided context.'\n"
        "Never use training knowledge. Never invent facts."
    )

    _FALLBACK_SYSTEM_PROMPT = (
        "You are a helpful, knowledgeable assistant. "
        "Answer the question using your general knowledge. "
        "Be accurate, concise, and honest about uncertainty."
    )

    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        if chunks:
            context = "\n\n".join(c["text"] for c in chunks)
            system = self._SYSTEM_PROMPT
            user_msg = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            system = self._FALLBACK_SYSTEM_PROMPT
            user_msg = (
                f"Question: {query}\n\n"
                "Note: No relevant documents were found. "
                "Answer from general knowledge and start your reply with: "
                "'*(No matching documents — answering from general knowledge)*'\n\nAnswer:"
            )
        kwargs = dict(
            model=self.config.groq_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=self.config.temperature,
            max_tokens=768,
        )
        # Tests inject self._groq_client directly; production uses the key pool.
        if self._groq_client is not None:
            response = self._groq_client.chat.completions.create(**kwargs)
        else:
            response = self._get_pool().chat_completions_create(**kwargs)
        return response.choices[0].message.content.strip()

    def query(self, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        chunks = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        context_text = "\n\n".join(c["text"] for c in chunks)

        t1 = time.perf_counter()
        answer = self.generate(question, chunks)
        generation_ms = (time.perf_counter() - t1) * 1000

        return {
            "question": question,
            "answer": answer,
            "context": context_text,
            "chunks": chunks,
            "config": self.config.to_dict(),
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
        }
