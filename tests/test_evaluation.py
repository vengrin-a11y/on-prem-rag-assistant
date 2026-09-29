import importlib.util
import json
from pathlib import Path

import httpx


def test_evaluation_continues_after_timeout(monkeypatch, tmp_path):
    path = Path(__file__).parents[1] / "eval/evaluate.py"
    spec = importlib.util.spec_from_file_location("portfolio_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeClient:
        def __init__(self, timeout):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ReadTimeout("temporary runtime stall")
            return httpx.Response(200, json={"answer": "14 dní", "confidence": "high",
                                             "sources": [{"document": "returns.md"}]},
                                  request=httpx.Request("POST", url))

    monkeypatch.setattr(module.httpx, "Client", FakeClient)
    cases = [
        {"question": "Unknown?", "abstain": True, "expected_keywords": [], "expected_source": None},
        {"question": "Return?", "abstain": False, "expected_keywords": ["14"], "expected_source": "returns.md"},
    ]
    details = tmp_path / "details.jsonl"
    result = module.evaluate(cases, "http://localhost:8000", details)
    assert result["questions"] == 2
    assert result["errors"] == 1
    assert result["abstention_total"] == 1
    assert result["abstention_correct"] == 0
    assert result["answer_correct"] == result["citation_correct"] == 1
    records = [json.loads(line) for line in details.read_text().splitlines()]
    assert records[0]["error"] == "ReadTimeout"
    assert records[1]["answer_ok"] is True
