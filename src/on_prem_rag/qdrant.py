"""Qdrant HTTP adapter with stable point IDs for repeatable ingestion."""

import re
import uuid

import httpx

from on_prem_rag.chunking import Chunk


def payload_for(chunk: Chunk) -> dict[str, str]:
    if chunk.document != chunk.document.split("/")[-1] or "\\" in chunk.document or chunk.document.startswith("."):
        raise ValueError("document must be a filename")
    if not chunk.document:
        raise ValueError("invalid document filename")
    return {"document": chunk.document, "chunk_id": chunk.chunk_id, "section": chunk.section, "text": chunk.text}


class QdrantClient:
    def __init__(self, base_url: str, collection: str):
        self.base_url = base_url.rstrip("/")
        self.collection = collection

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", self.collection):
            raise ValueError("invalid collection name")
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("nonempty chunks and vectors must align")
        size = len(vectors[0])
        if not size or any(len(vector) != size for vector in vectors):
            raise ValueError("embedding dimensions must match")
        # Recreate is deliberately scoped to the configured demo collection.
        with httpx.Client(timeout=120) as client:
            url = f"{self.base_url}/collections/{self.collection}"
            existing = client.get(url)
            if existing.status_code == 404:
                response = client.put(url, json={"vectors": {"size": size, "distance": "Cosine"}})
                response.raise_for_status()
            elif existing.status_code == 200:
                configured = existing.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
                if configured.get("size") != size:
                    raise ValueError("existing collection has a different embedding dimension")
            else:
                existing.raise_for_status()
            points = [
                {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.document}:{chunk.chunk_id}")),
                 "vector": vector, "payload": payload_for(chunk)}
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            response = client.put(f"{url}/points?wait=true", json={"points": points})
            response.raise_for_status()

    def search(self, vector: list[float], limit: int = 4) -> list[dict]:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{self.base_url}/collections/{self.collection}/points/search",
                json={"vector": vector, "limit": limit, "with_payload": True, "with_vector": False},
            )
            response.raise_for_status()
            result = response.json().get("result")
        if not isinstance(result, list):
            raise TypeError("invalid search response")
        return result
