"""Conservative output gate adapted from the source verifier's evidence checks."""

import json
import re

from on_prem_rag.citations import citations_for
from on_prem_rag.retrieval import Evidence

FALLBACK = "Informáciu som v dostupných dokumentoch nenašiel."


def abstain() -> dict:
    return {"answer": FALLBACK, "confidence": "low", "sources": []}


def verify(raw: str, evidence: list[Evidence]) -> dict:
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or parsed.get("abstain") is True:
            return abstain()
        answer = parsed.get("answer")
        ids = parsed.get("source_ids")
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 1000:
            return abstain()
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            return abstain()
        sources = citations_for(evidence, ids)
        if re.search(r"(?:https?://|/Users/|/home/|PRIVATE KEY)", answer, re.IGNORECASE):
            return abstain()
        # Grounding gate: every substantive answer token must appear in cited evidence.
        cited_text = " ".join(item.text for item in evidence if item.chunk_id in ids).casefold()
        tokens = re.findall(r"[\w]+", answer.casefold())
        substantive = [token for token in tokens if len(token) >= 5 or token.isdigit()]
        if not substantive or any(token not in cited_text for token in substantive):
            return abstain()
        return {"answer": answer.strip(), "confidence": "high", "sources": sources}
    except (ValueError, TypeError, KeyError):
        return abstain()
