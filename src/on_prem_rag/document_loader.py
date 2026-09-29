"""Load only direct Markdown and text files from an explicit directory."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    name: str
    text: str


def load_documents(directory: str | Path) -> list[Document]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError("document directory does not exist")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise ValueError("document directory contains no Markdown or text files")
    return [Document(path.name, path.read_text(encoding="utf-8")) for path in files]
