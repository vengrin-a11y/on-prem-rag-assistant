"""Minimal Ollama HTTP adapter."""

import httpx


class OllamaClient:
    def __init__(self, base_url: str, embed_model: str, chat_model: str):
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model

    def embed(self, text: str) -> list[float]:
        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{self.base_url}/api/embed", json={"model": self.embed_model, "input": [text]}
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != 1 or not isinstance(vectors[0], list):
            raise ValueError("invalid embedding response")
        vector = vectors[0]
        if not vector or not all(isinstance(value, (int, float)) for value in vector):
            raise ValueError("invalid embedding vector")
        return [float(value) for value in vector]

    def generate(self, system: str, user: str) -> str:
        with httpx.Client(timeout=180) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.chat_model, "stream": False, "format": "json", "think": False,
                      "options": {"temperature": 0, "num_predict": 256},
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content")
        if not isinstance(content, str):
            raise TypeError("invalid chat response")
        return content
