import hashlib
import json
from dataclasses import dataclass


@dataclass
class RAGConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    overlap: int = 64
    top_k: int = 5
    temperature: float = 0.1
    groq_model: str = "llama-3.3-70b-versatile"
    rerank: bool = False
    similarity_threshold: float = 0.3

    def collection_key(self) -> str:
        key_data = {
            "model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "groq_model": self.groq_model,
            "rerank": self.rerank,
            "similarity_threshold": self.similarity_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RAGConfig":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
