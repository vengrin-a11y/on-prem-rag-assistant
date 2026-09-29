"""Idempotent demo ingestion: stable chunks and point IDs."""

import argparse

from on_prem_rag.chunking import chunk_document
from on_prem_rag.config import Settings
from on_prem_rag.document_loader import load_documents
from on_prem_rag.ollama import OllamaClient
from on_prem_rag.qdrant import QdrantClient


def ingest(directory: str, settings: Settings | None = None) -> int:
    config = settings or Settings.from_env()
    chunks = [chunk for document in load_documents(directory) for chunk in chunk_document(document)]
    if not chunks:
        raise ValueError("documents contain no content")
    ollama = OllamaClient(config.ollama_base_url, config.ollama_embed_model, config.ollama_chat_model)
    vectors = [ollama.embed(chunk.text) for chunk in chunks]
    QdrantClient(config.qdrant_url, config.qdrant_collection).upsert(chunks, vectors)
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    print(f"Ingested {ingest(args.directory)} chunks")
