from fastapi.testclient import TestClient

from on_prem_rag import api


def test_health():
    assert TestClient(api.app).get("/health").json() == {"status": "ok"}


def test_ask_happy_and_invalid(monkeypatch):
    class FakePipeline:
        def __init__(self, settings):
            pass

        def ask(self, question):
            return {"answer": "Áno", "confidence": "high", "sources": []}

    monkeypatch.setattr(api, "Pipeline", FakePipeline)
    client = TestClient(api.app)
    assert client.post("/ask", json={"question": "Test otázka?"}).status_code == 200
    assert client.post("/ask", json={"question": "  "}).status_code == 422
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_ask_dependency_error_is_generic(monkeypatch):
    class BrokenPipeline:
        def __init__(self, settings):
            raise RuntimeError("private details")

    monkeypatch.setattr(api, "Pipeline", BrokenPipeline)
    response = TestClient(api.app).post("/ask", json={"question": "Test otázka?"})
    assert response.status_code == 503
    assert "private details" not in response.text
