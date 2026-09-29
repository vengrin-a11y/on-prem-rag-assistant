"""Environment-only configuration with safe defaults."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3.5:9b"
    ollama_embed_model: str = "nomic-embed-text"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "portfolio_rag_docs"
    min_retrieval_score: float = 0.35

    @classmethod
    def from_env(cls) -> "Settings":
        score = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.35"))
        if not 0 <= score <= 1:
            raise ValueError("MIN_RETRIEVAL_SCORE must be between 0 and 1")
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", cls.ollama_chat_model),
            ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", cls.ollama_embed_model),
            qdrant_url=os.getenv("QDRANT_URL", cls.qdrant_url),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", cls.qdrant_collection),
            min_retrieval_score=score,
        )
