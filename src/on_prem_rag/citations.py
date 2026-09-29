"""Return citations only from retrieved evidence."""

from on_prem_rag.retrieval import Evidence


def citations_for(evidence: list[Evidence], ids: list[str]) -> list[dict[str, str]]:
    by_id = {item.chunk_id: item for item in evidence}
    if not ids or len(ids) != len(set(ids)) or any(key not in by_id for key in ids):
        raise ValueError("citations must reference retrieved chunks")
    return [
        {"document": by_id[key].document, "chunk_id": key, "snippet": by_id[key].text[:220]}
        for key in ids
    ]
