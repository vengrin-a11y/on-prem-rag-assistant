"""Deterministic heading-first chunking adapted from the private source core."""

import hashlib
import re
from dataclasses import dataclass

from on_prem_rag.document_loader import Document


@dataclass(frozen=True)
class Chunk:
    document: str
    chunk_id: str
    section: str
    text: str


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = "Document"
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,4}\s+(.+?)\s*$", line)
        if match:
            if "\n".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = match.group(1), []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def chunk_document(document: Document, max_chars: int = 1200, overlap_chars: int = 120) -> list[Chunk]:
    if max_chars < 100 or not 0 <= overlap_chars < max_chars:
        raise ValueError("invalid chunk size or overlap")
    result: list[Chunk] = []
    for heading, body in _sections(document.text):
        start = 0
        while start < len(body):
            part = body[start:start + max_chars].strip()
            if part:
                digest = hashlib.sha256(f"{document.name}\0{heading}\0{part}".encode()).hexdigest()[:12]
                result.append(Chunk(document.name, digest, heading, part))
            end = min(start + max_chars, len(body))
            if end == len(body):
                break
            start = end - overlap_chars
    return result
