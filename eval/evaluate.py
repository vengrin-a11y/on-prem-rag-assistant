"""Run measured end-to-end evaluation against the configured local API."""

import argparse
import json
import time
from pathlib import Path

import httpx


def evaluate(cases: list[dict], endpoint: str) -> dict:
    answer_correct = citation_correct = abstention_correct = 0
    abstention_total = 0
    elapsed: list[float] = []
    with httpx.Client(timeout=180) as client:
        for case in cases:
            start = time.perf_counter()
            response = client.post(f"{endpoint.rstrip('/')}/ask", json={"question": case["question"]})
            response.raise_for_status()
            result = response.json()
            elapsed.append(time.perf_counter() - start)
            abstained = result.get("confidence") == "low" and result.get("sources") == []
            if case["abstain"]:
                abstention_total += 1
                abstention_correct += abstained
                answer_correct += abstained
                citation_correct += abstained
            else:
                answer_correct += not abstained and all(
                    word.casefold() in result.get("answer", "").casefold()
                    for word in case["expected_keywords"]
                )
                citation_correct += any(
                    source.get("document") == case["expected_source"]
                    for source in result.get("sources", [])
                )
    return {"questions": len(cases), "answer_correct": answer_correct,
            "citation_correct": citation_correct, "abstention_correct": abstention_correct,
            "abstention_total": abstention_total, "average_latency_seconds": sum(elapsed) / len(elapsed)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.jsonl"))
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != 20:
        raise SystemExit("evaluation requires exactly 20 cases")
    print(json.dumps(evaluate(cases, args.api), indent=2))
