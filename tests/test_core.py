import json

import pytest

from on_prem_rag.chunking import chunk_document
from on_prem_rag.config import Settings
from on_prem_rag.document_loader import Document
from on_prem_rag.pipeline import Pipeline
from on_prem_rag.qdrant import payload_for
from on_prem_rag.retrieval import Evidence, parse_hits
from on_prem_rag.verifier import FALLBACK, verify


def test_config(monkeypatch):
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "local-model")
    assert Settings.from_env().ollama_chat_model == "local-model"
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "2")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_chunking_is_deterministic_and_payload_has_no_path():
    document = Document("example.md", "# Title\n\n" + "word " * 100)
    first = chunk_document(document, max_chars=120, overlap_chars=20)
    assert first == chunk_document(document, max_chars=120, overlap_chars=20)
    assert len(first) > 1
    assert set(payload_for(first[0])) == {"document", "chunk_id", "section", "text"}
    with pytest.raises(ValueError):
        payload_for(chunk_document(Document("../bad.md", "hello"))[0])


def test_retrieval_parsing_filters_invalid_evidence():
    hits = [
        {"score": 0.8, "payload": {"document": "faq.md", "chunk_id": "a", "text": "Fact"}},
        {"score": 0.1, "payload": {"document": "faq.md", "chunk_id": "b", "text": "No"}},
        {"score": 0.9, "payload": {"document": "/private/file", "chunk_id": "c", "text": "No"}},
    ]
    assert parse_hits(hits, 0.35) == [Evidence("faq.md", "a", "Fact", 0.8)]


def test_verifier_citations_and_fallback():
    evidence = [Evidence("returns.md", "id", "Produkt možno vrátiť do 14 dní.", 0.8)]
    assert verify(json.dumps({"answer": "Vrátiť do 14 dní.", "source_ids": ["id"]}), evidence)["sources"][0]["document"] == "returns.md"
    assert verify(json.dumps({"answer": "Vrátiť do 30 dní.", "source_ids": ["id"]}), evidence)["answer"] == FALLBACK
    assert verify(json.dumps({"answer": "Vrátiť do 14 dní.", "source_ids": ["unknown"]}), evidence)["sources"] == []


class FakeOllama:
    def embed(self, text):
        return [0.1, 0.2]

    def generate(self, system, user):
        return json.dumps({"answer": "Vrátiť do 14 dní.", "source_ids": ["id"]})


class FakeQdrant:
    def search(self, vector):
        return [{"score": 0.8, "payload": {"document": "returns.md", "chunk_id": "id", "text": "Produkt možno vrátiť do 14 dní."}}]


def test_pipeline_happy_path_and_safe_fallback():
    pipeline = Pipeline(Settings(), FakeOllama(), FakeQdrant())
    assert pipeline.ask("Lehota?")["confidence"] == "high"
    class EmptyQdrant:
        def search(self, vector):
            return []
    assert Pipeline(Settings(), FakeOllama(), EmptyQdrant()).ask("Neznáme?")["answer"] == FALLBACK


def test_20_evaluation_cases_are_synthetic_and_complete():
    from pathlib import Path

    cases = [json.loads(line) for line in (Path(__file__).parents[1] / "eval/questions.jsonl").read_text().splitlines()]
    assert len(cases) == 20
    assert sum(case["abstain"] for case in cases) == 5
    assert all(set(case) == {"question", "expected_keywords", "expected_source", "abstain"} for case in cases)
