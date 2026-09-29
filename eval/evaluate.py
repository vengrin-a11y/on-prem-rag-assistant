"""Run measured end-to-end evaluation against the configured local API."""

import argparse
import json
import time
from pathlib import Path

import httpx


def evaluate(cases: list[dict], endpoint: str, details: Path | None = None) -> dict:
    answer_correct = citation_correct = abstention_correct = 0
    abstention_total = 0
    errors = 0
    elapsed: list[float] = []
    records: list[dict] = []
    with httpx.Client(timeout=210) as client:
        for index, case in enumerate(cases, start=1):
            start = time.perf_counter()
            abstention_total += bool(case["abstain"])
            try:
                response = client.post(f"{endpoint.rstrip('/')}/ask", json={"question": case["question"]})
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise TypeError("invalid API response")
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                errors += 1
                elapsed.append(time.perf_counter() - start)
                records.append({"index": index, "question": case["question"],
                                "error": type(exc).__name__, "seconds": round(elapsed[-1], 3)})
                continue
            elapsed.append(time.perf_counter() - start)
            abstained = result.get("confidence") == "low" and result.get("sources") == []
            if case["abstain"]:
                answer_ok = citation_ok = abstention_ok = abstained
                abstention_correct += abstention_ok
            else:
                answer_ok = not abstained and all(
                    word.casefold() in result.get("answer", "").casefold()
                    for word in case["expected_keywords"]
                )
                citation_ok = any(
                    source.get("document") == case["expected_source"]
                    for source in result.get("sources", [])
                )
            answer_correct += answer_ok
            citation_correct += citation_ok
            records.append({"index": index, "question": case["question"], "answer_ok": bool(answer_ok),
                            "citation_ok": bool(citation_ok), "seconds": round(elapsed[-1], 3),
                            "response": result})
    if details is not None:
        details.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
    return {"questions": len(cases), "answer_correct": answer_correct,
            "citation_correct": citation_correct, "abstention_correct": abstention_correct,
            "abstention_total": abstention_total, "average_latency_seconds": sum(elapsed) / len(elapsed),
            "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.jsonl"))
    parser.add_argument("--details", type=Path, help="optional per-question diagnostics; keep outside Git")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != 20:
        raise SystemExit("evaluation requires exactly 20 cases")
    print(json.dumps(evaluate(cases, args.api, args.details), indent=2))
