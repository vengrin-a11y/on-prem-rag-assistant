"""Parse and score retrieval results before exposing evidence."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    document: str
    chunk_id: str
    text: str
    score: float


def parse_hits(hits: list[dict], minimum_score: float) -> list[Evidence]:
    evidence: list[Evidence] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        score = hit.get("score")
        payload = hit.get("payload")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or score < minimum_score:
            continue
        if not isinstance(payload, dict):
            continue
        document, chunk_id, content = (payload.get(key) for key in ("document", "chunk_id", "text"))
        if not all(isinstance(value, str) and value for value in (document, chunk_id, content)):
            continue
        if "/" in document or "\\" in document or document.startswith("."):
            continue
        evidence.append(Evidence(document, chunk_id, content, float(score)))
    return evidence
